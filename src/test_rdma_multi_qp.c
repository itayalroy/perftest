/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * Test program for RDMA Multi-QP API
 * Tests bandwidth with variable number of QPs and threads
 */

#include "config.h"
#include "rdma_multi_qp.h"
#include "nvlink_transfer.h"
#include "get_clock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <pthread.h>

#ifdef HAVE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#define DEFAULT_BUFFER_SIZE (1024 * 1024 * 1024)  /* 1 GB */
#define DEFAULT_ITERATIONS 1
#define WARMUP_ITERATIONS 0
#define MAX_THREADS 32

struct thread_args {
    rdma_multi_qp_context_t *ctx;
    int qp_index;
    int iterations;
    int is_server;         /* 1 if server (just wait), 0 if client (send) */
    struct nvlink_context *nvlink_ctx;  /* NVLink context for threads > 0 */
    int gpu0_id;           /* Source GPU ID for NVLink (GPU 0) */
    int num_qps;           /* Total number of QPs for buffer splitting */
    pthread_barrier_t *start_barrier;  /* Barrier before starting work */
    pthread_barrier_t *end_barrier;    /* Barrier after finishing work */
    double *bandwidth;  /* Output: bandwidth in GB/s */
    int *status;        /* Output: 0 on success */
    bool nics_only;     /* Whether to only use NICs for buffer allocation */
};

/* Parse comma-separated string into array */
static int parse_comma_separated(const char *str, char **out, int max_count)
{
    char *str_copy = strdup(str);
    char *token;
    int count = 0;
    
    if (!str_copy) return -1;
    
    token = strtok(str_copy, ",");
    while (token && count < max_count) {
        /* Trim whitespace */
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') *end-- = '\0';
        
        out[count++] = strdup(token);
        token = strtok(NULL, ",");
    }
    
    free(str_copy);
    return count;
}

/* Parse comma-separated integers into array */
static int parse_comma_separated_ints(const char *str, int *out, int max_count)
{
    char *str_copy = strdup(str);
    char *token;
    int count = 0;
    
    if (!str_copy) return -1;
    
    token = strtok(str_copy, ",");
    while (token && count < max_count) {
        /* Trim whitespace */
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') *end-- = '\0';
        
        out[count++] = atoi(token);
        token = strtok(NULL, ",");
    }
    
    free(str_copy);
    return count;
}

static void *write_thread(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    rdma_multi_qp_context_t *ctx = args->ctx;
    int qp_index = args->qp_index;
    size_t buffer_size = rdma_get_buffer_size(ctx);
    size_t part_size = buffer_size / args->num_qps;
    size_t offset = qp_index * part_size;
    cycles_t start_cycles, end_cycles;
    double total_time;
    double cpu_mhz;
    int i;
    
    /* Server just waits - no sending */
    if (args->is_server) {
        printf("Server thread for QP %d: Waiting for client to complete...\n", qp_index);
        sleep(300);  /* 5 minutes should be enough */
        *args->bandwidth = 0.0;
        *args->status = 0;
        printf("Server thread for QP %d: Done waiting\n", qp_index);
        return NULL;
    }
    
    /* Threads > 0 use NVLink then RDMA if NVLink context is set */
    if (qp_index > 0 && args->nvlink_ctx && !args->nics_only) {
        void *src_buffer = rdma_get_local_buffer(ctx, 0);  /* Use QP 0's buffer as NVLink source */
        void *nvlink_dst = rdma_get_local_buffer(ctx, qp_index);  /* Use QP i's buffer as NVLink destination */
        
        /* Calculate buffer part pointers for this GPU */
        void *src_part = (char *)src_buffer + offset;
        void *dst_part = (char *)nvlink_dst + offset;
        
        /* Get CPU frequency */
        cpu_mhz = get_cpu_mhz(0);
        if (cpu_mhz <= 0) {
            fprintf(stderr, "Failed to get CPU frequency\n");
            *args->status = -1;
            return NULL;
        }
        
        printf("Thread %d: Starting NVLink warmup (GPU %d -> GPU %d, part %zu bytes at offset %zu)...\n",
               qp_index, args->gpu0_id, args->nvlink_ctx->gpu1_id, part_size, offset);
        
        /* NVLink warmup */
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (nvlink_copy_gpu_to_gpu_async(args->nvlink_ctx, dst_part, 
                                              src_part, part_size) != 0) {
                fprintf(stderr, "NVLink warmup copy %d failed\n", i);
                *args->status = -1;
                return NULL;
            }
        }
        if (nvlink_synchronize(args->nvlink_ctx) != 0) {
            fprintf(stderr, "NVLink warmup sync failed\n");
            *args->status = -1;
            return NULL;
        }
        
        printf("Thread %d: NVLink warmup completed, starting RDMA warmup...\n", qp_index);
        
        /* RDMA warmup */
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
                fprintf(stderr, "QP %d: RDMA warmup write %d failed\n", qp_index, i);
                *args->status = -1;
                return NULL;
            }
        }
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "QP %d: RDMA warmup completion %d failed\n", qp_index, i);
                *args->status = -1;
                return NULL;
            }
        }
        
        printf("Thread %d: Warmup completed, starting measurements...\n", qp_index);
        
        /* Wait at start barrier before beginning timed work */
        if (args->start_barrier) {
            pthread_barrier_wait(args->start_barrier);
        }
        
        /* Timed transfers: NVLink + RDMA */
        start_cycles = get_cycles();
        
        for (i = 0; i < args->iterations; i++) {
            /* Step 1: NVLink transfer */
            if (nvlink_copy_gpu_to_gpu_async(args->nvlink_ctx, dst_part,
                                              src_part, part_size) != 0) {
                fprintf(stderr, "NVLink copy %d failed\n", i);
                *args->status = -1;
                return NULL;
            }
        }
        
        /* Wait for NVLink to complete */
        if (nvlink_synchronize(args->nvlink_ctx) != 0) {
            fprintf(stderr, "NVLink sync failed\n");
            *args->status = -1;
            return NULL;
        }
        
        for (i = 0; i < args->iterations; i++) {
            /* Step 2: RDMA write */
            if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
                fprintf(stderr, "RDMA write %d failed\n", i);
                *args->status = -1;
                return NULL;
            }
        }

        /* Poll for all RDMA completions */
        for (i = 0; i < args->iterations; i++) {
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "RDMA completion %d failed\n", i);
                *args->status = -1;
                return NULL;
            }
        }
        
        /* Wait at end barrier after completing all work */
        if (args->end_barrier) {
            pthread_barrier_wait(args->end_barrier);
        }

        end_cycles = get_cycles();
        total_time = (double)(end_cycles - start_cycles) / (cpu_mhz * 1e6);
        
        /* Calculate bandwidth (total data transferred: part_size * iterations) */
        *args->bandwidth = (part_size * args->iterations) / (total_time * 1e9);
        *args->status = 0;
        
        printf("Thread %d: Completed %d iterations (NVLink + RDMA) in %.6f seconds\n",
               qp_index, args->iterations, total_time);
        
        return NULL;
    }
    
    /* Thread 0 uses RDMA writes only */
    /* Get CPU frequency */
    cpu_mhz = get_cpu_mhz(0);
    if (cpu_mhz <= 0) {
        fprintf(stderr, "Failed to get CPU frequency\n");
        *args->status = -1;
        return NULL;
    }
    
    printf("Thread for QP %d: Starting warmup (part %zu bytes at offset %zu)...\n", 
           qp_index, part_size, offset);
    
    /* Warmup */
    for (i = 0; i < WARMUP_ITERATIONS; i++) {
        if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
            fprintf(stderr, "QP %d: Warmup write %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }

    /* Poll for all warmup completions */
    for (i = 0; i < WARMUP_ITERATIONS; i++) {
        if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
            fprintf(stderr, "QP %d: Warmup completion %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }
    
    printf("Thread for QP %d: Warmup completed, starting measurements...\n", qp_index);
    
    /* Wait at start barrier before beginning timed work */
    if (args->start_barrier) {
        pthread_barrier_wait(args->start_barrier);
    }
    
    /* Timed transfers */
    start_cycles = get_cycles();
    
    for (i = 0; i < args->iterations; i++) {
        char range_name[64];
        snprintf(range_name, sizeof(range_name), "rdma_write_QP%d_iter%d", qp_index, i);
        #ifdef HAVE_CUDA
        nvtxRangePushA(range_name);
        #endif
        if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
            #ifdef HAVE_CUDA
            nvtxRangePop();
            #endif
            fprintf(stderr, "QP %d: Write %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
        #ifdef HAVE_CUDA
        nvtxRangePop();
        #endif
    }

    /* Poll for all timed test completions */
    for (i = 0; i < args->iterations; i++) {
        if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
            fprintf(stderr, "QP %d: Completion %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }
    
    /* Wait at end barrier after completing all work */
    if (args->end_barrier) {
        pthread_barrier_wait(args->end_barrier);
    }

    end_cycles = get_cycles();
    total_time = (double)(end_cycles - start_cycles) / (cpu_mhz * 1e6);
    
    /* Calculate bandwidth (total data transferred: part_size * iterations) */
    *args->bandwidth = (part_size * args->iterations) / (total_time * 1e9);
    *args->status = 0;
    
    printf("Thread for QP %d: Completed %d iterations in %.6f seconds\n",
           qp_index, args->iterations, total_time);
    
    return NULL;
}

static void print_usage(const char *prog_name)
{
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -n, --nics LIST       Comma-separated NIC device names (e.g., mlx5_0,mlx5_1)\n");
    printf("  -s, --server          Run as server (default: client)\n");
    printf("  -a, --addr ADDR       Server IP address (default: 127.0.0.1)\n");
    printf("  -p, --port PORT       Base port (default: 18515)\n");
    printf("  -b, --buffer SIZE     Buffer size in bytes (default: 64MB)\n");
    printf("  -i, --iterations NUM  Number of iterations (default: 256)\n");
    printf("  -g, --gpus LIST       Comma-separated GPU IDs (-1 for host, default: -1)\n");
    printf("  -h, --help            Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  Server: %s -s -n mlx5_0,mlx5_1 -g 0,1\n", prog_name);
    printf("  Client: %s -n mlx5_0,mlx5_1 -g 0,1 -a 192.168.1.100\n", prog_name);
}

int main(int argc, char *argv[])
{
    struct rdma_multi_qp_config config = {0};
    rdma_multi_qp_context_t *ctx = NULL;
    pthread_t *threads = NULL;
    struct thread_args *args = NULL;
    double *bandwidths = NULL;
    int *statuses = NULL;
    char **nic_names = NULL;
    int *gpu_ids = NULL;
    struct nvlink_context *nvlink_ctxs = NULL;
    int num_qps = 0;
    int opt;
    int ret = 0;
    int i;
    char *nics_str = NULL;
    char *gpus_str = NULL;
    
    /* Defaults */
    config.base_port = 18515;
    config.buffer_size = DEFAULT_BUFFER_SIZE;
    config.is_server = 0;
    config.server_addr = NULL;
    
    /* Command line parsing */
    static struct option long_options[] = {
        {"nics", required_argument, 0, 'n'},
        {"server", no_argument, 0, 's'},
        {"addr", required_argument, 0, 'a'},
        {"port", required_argument, 0, 'p'},
        {"buffer", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'i'},
        {"gpus", required_argument, 0, 'g'},
        {"help", no_argument, 0, 'h'},
        {"nics-only", no_argument, 0, 'c'},
        {0, 0, 0, 0}
    };
    
    while ((opt = getopt_long(argc, argv, "n:sa:p:b:i:g:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'n':
            nics_str = optarg;
            break;
        case 's':
            config.is_server = 1;
            break;
        case 'a':
            config.server_addr = optarg;
            break;
        case 'c':
            config.nics_only = 1;
            break;
        case 'p':
            config.base_port = atoi(optarg);
            break;
        case 'b':
            config.buffer_size = strtoull(optarg, NULL, 0);
            break;
        case 'i':
            /* Will be set per thread */
            break;
        case 'g':
            gpus_str = optarg;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!nics_str) {
        fprintf(stderr, "Error: NIC names must be specified with -n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    /* Parse NIC names */
    nic_names = calloc(MAX_THREADS, sizeof(char *));
    if (!nic_names) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    num_qps = parse_comma_separated(nics_str, nic_names, MAX_THREADS);
    if (num_qps <= 0) {
        fprintf(stderr, "Error: Failed to parse NIC names\n");
        free(nic_names);
        return 1;
    }
    
    /* Parse GPU IDs */
    gpu_ids = calloc(num_qps, sizeof(int));
    if (!gpu_ids) {
        fprintf(stderr, "Failed to allocate memory\n");
        for (i = 0; i < num_qps; i++) free(nic_names[i]);
        free(nic_names);
        return 1;
    }
    
    if (gpus_str) {
        int gpu_count = parse_comma_separated_ints(gpus_str, gpu_ids, num_qps);
        if (gpu_count != num_qps) {
            fprintf(stderr, "Error: Number of GPUs (%d) must match number of NICs (%d)\n", 
                    gpu_count, num_qps);
            for (i = 0; i < num_qps; i++) free(nic_names[i]);
            free(nic_names);
            free(gpu_ids);
            return 1;
        }
    } else {
        /* Default: all host memory */
        for (i = 0; i < num_qps; i++) {
            gpu_ids[i] = -1;
        }
    }
    
    /* Setup config */
    config.nic_names = (const char **)nic_names;
    config.gpu_id = gpu_ids;
    config.num_qps = num_qps;
    
    printf("\n========================================\n");
    printf("RDMA Multi-QP Bandwidth Test\n");
    printf("========================================\n");
    printf("Mode: %s\n", config.is_server ? "Server" : "Client");
    printf("Number of QPs: %d\n", num_qps);
    for (i = 0; i < num_qps; i++) {
        printf("NIC %d: %s, GPU %d: %d\n", i, nic_names[i], i, gpu_ids[i]);
    }
    printf("Base port: %d\n", config.base_port);
    printf("Buffer size: %zu bytes (%.2f MB)\n", 
           config.buffer_size, config.buffer_size / (1024.0 * 1024.0));
    if (config.server_addr) {
        printf("Server address: %s\n", config.server_addr);
    }
    printf("Iterations per thread: %d\n", DEFAULT_ITERATIONS);
    printf("========================================\n\n");
    
    /* Initialize RDMA context */
    printf("Initializing RDMA context...\n");
    if (rdma_multi_qp_init(&config, &ctx, config.nics_only) != 0) {
        fprintf(stderr, "Failed to initialize RDMA context\n");
        ret = 1;
        goto cleanup;
    }
    printf("RDMA context initialized\n\n");
    
    /* Connect QPs */
    printf("Connecting QPs...\n");
    if (rdma_multi_qp_connect(ctx) != 0) {
        fprintf(stderr, "Failed to connect QPs\n");
        rdma_multi_qp_cleanup(ctx);
        ret = 1;
        goto cleanup;
    }
    printf("QPs connected\n\n");
    
    /* Allocate thread arrays */
    threads = calloc(num_qps, sizeof(pthread_t));
    args = calloc(num_qps, sizeof(struct thread_args));
    bandwidths = calloc(num_qps, sizeof(double));
    statuses = calloc(num_qps, sizeof(int));
    nvlink_ctxs = calloc(num_qps, sizeof(struct nvlink_context));
    
    if (!threads || !args || !bandwidths || !statuses || !nvlink_ctxs) {
        fprintf(stderr, "Failed to allocate memory\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Setup NVLink contexts for threads > 0 */
    if (!config.is_server && gpu_ids[0] >= 0) {
        for (i = 1; i < num_qps; i++) {
            if (gpu_ids[i] >= 0 && gpu_ids[i] != gpu_ids[0]) {
                printf("Initializing NVLink (GPU %d -> GPU %d) for thread %d...\n", 
                       gpu_ids[0], gpu_ids[i], i);
                if (nvlink_init_context(&nvlink_ctxs[i], gpu_ids[0], gpu_ids[i]) != 0) {
                    fprintf(stderr, "Failed to initialize NVLink context for thread %d\n", i);
                    ret = 1;
                    goto cleanup;
                }
            }
        }
        printf("NVLink contexts initialized\n\n");
    }
    
    /* Initialize barriers for synchronizing thread start/end */
    pthread_barrier_t start_barrier, end_barrier;
    int barriers_initialized = 0;
    if (!config.is_server) {
        /* Barriers need num_qps + 1 (threads + main thread) */
        if (pthread_barrier_init(&start_barrier, NULL, num_qps + 1) != 0) {
            fprintf(stderr, "Failed to initialize start barrier\n");
            ret = 1;
            goto cleanup;
        }
        if (pthread_barrier_init(&end_barrier, NULL, num_qps + 1) != 0) {
            fprintf(stderr, "Failed to initialize end barrier\n");
            pthread_barrier_destroy(&start_barrier);
            ret = 1;
            goto cleanup;
        }
        barriers_initialized = 1;
    }
    
    /* Setup thread arguments */
    for (i = 0; i < num_qps; i++) {
        args[i].ctx = ctx;
        args[i].qp_index = i;
        args[i].iterations = DEFAULT_ITERATIONS;
        args[i].is_server = config.is_server;
        args[i].nvlink_ctx = (i > 0 && gpu_ids[0] >= 0 && gpu_ids[i] >= 0 && 
                              gpu_ids[i] != gpu_ids[0]) ? &nvlink_ctxs[i] : NULL;
        args[i].gpu0_id = gpu_ids[0];
        args[i].num_qps = num_qps;
        args[i].start_barrier = config.is_server ? NULL : &start_barrier;
        args[i].end_barrier = config.is_server ? NULL : &end_barrier;
        args[i].bandwidth = &bandwidths[i];
        args[i].status = &statuses[i];
        args[i].nics_only = config.nics_only;
        statuses[i] = -1;
    }
    
    /* Get CPU frequency for wall-clock time measurement */
    double cpu_mhz = get_cpu_mhz(0);
    if (cpu_mhz <= 0) {
        fprintf(stderr, "Failed to get CPU frequency\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Measure wall-clock time for total bandwidth calculation */
    cycles_t total_start_cycles, total_end_cycles;
    size_t total_buffer_size = rdma_get_buffer_size(ctx);
    
    /* Create threads */
    printf("Creating %d threads...\n", num_qps);
    for (i = 0; i < num_qps; i++) {
        if (pthread_create(&threads[i], NULL, write_thread, &args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            ret = 1;
            goto cleanup;
        }
    }
    
    /* Wait for all threads to reach start barrier, then take start time */
    if (!config.is_server) {
        pthread_barrier_wait(&start_barrier);
        total_start_cycles = get_cycles();
    }
    
    /* Wait for all threads to reach end barrier, then take end time */
    if (!config.is_server) {
        pthread_barrier_wait(&end_barrier);
        total_end_cycles = get_cycles();
    }
    
    /* Wait for threads */
    printf("Waiting for threads to complete...\n\n");
    for (i = 0; i < num_qps; i++) {
        pthread_join(threads[i], NULL);
        if (statuses[i] != 0) {
            fprintf(stderr, "Thread %d failed\n", i);
            ret = 1;
        }
    }
    
    /* Calculate total wall-clock time */
    double total_time = (double)(total_end_cycles - total_start_cycles) / (cpu_mhz * 1e6);
    
    /* Print results */
    printf("\n========================================\n");
    printf("Results:\n");
    printf("========================================\n");
    for (i = 0; i < num_qps; i++) {
        if (statuses[i] == 0) {
            if (i > 0 && args[i].nvlink_ctx) {
                printf("Thread %d (NVLink + RDMA) bandwidth: %.2f GB/s\n", i, bandwidths[i]);
            } else {
                printf("QP %d bandwidth: %.2f GB/s\n", i, bandwidths[i]);
            }
        } else {
            printf("QP %d: FAILED\n", i);
        }
    }
    
    /* Calculate total bandwidth based on wall-clock time */
    double total_bw = 0.0;
    int success_count = 0;
    for (i = 0; i < num_qps; i++) {
        if (statuses[i] == 0) {
            success_count++;
        }
    }
    if (success_count > 0 && total_time > 0.0) {
        /* Total bandwidth = (total buffer size * iterations) / total wall-clock time */
        total_bw = (total_buffer_size * DEFAULT_ITERATIONS) / (total_time * 1e9);
        printf("Total bandwidth (wall-clock): %.2f GB/s (measured over %.6f seconds)\n", 
               total_bw, total_time);
    }
    printf("========================================\n\n");
    
cleanup:
    /* Cleanup */
    printf("Cleaning up...\n");
    if (barriers_initialized) {
        pthread_barrier_destroy(&start_barrier);
        pthread_barrier_destroy(&end_barrier);
    }
    if (nvlink_ctxs) {
        for (i = 1; i < num_qps; i++) {
            if (args && args[i].nvlink_ctx) {
                nvlink_cleanup_context(&nvlink_ctxs[i]);
            }
        }
    }
    if (ctx) rdma_multi_qp_cleanup(ctx);
    if (threads) free(threads);
    if (args) free(args);
    if (bandwidths) free(bandwidths);
    if (statuses) free(statuses);
    if (nvlink_ctxs) free(nvlink_ctxs);
    if (nic_names) {
        for (i = 0; i < num_qps; i++) {
            if (nic_names[i]) free(nic_names[i]);
        }
        free(nic_names);
    }
    if (gpu_ids) free(gpu_ids);
    
    return ret;
}

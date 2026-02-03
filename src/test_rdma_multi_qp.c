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
#define MAX_THREADS 64

struct thread_args {
    rdma_multi_qp_context_t *ctx;
    int qp_index;
    int iterations;
    int is_server;         /* 1 if server (just wait), 0 if client (send) */
    struct nvlink_context *nvlink_ctx;  /* NVLink context for threads > 0 */
    int source_gpu_id;     /* Source GPU ID for this QP (for NVLink transfers) */
    int target_gpu_id;     /* Target GPU ID for this QP */
    int num_qps;           /* Total number of QPs */
    int num_source_gpus;   /* Number of source GPUs (for multi-source NVLink) */
    int source_gpu_index;  /* Index of source GPU (0, 1, 2, ...) */
    void *source_buffer;   /* Pointer to source buffer (full size, on source GPU) */
    pthread_barrier_t *start_barrier;  /* Barrier before starting work */
    pthread_barrier_t *end_barrier;    /* Barrier after finishing work */
    double *bandwidth;  /* Output: bandwidth in GB/s */
    int *status;        /* Output: 0 on success */
    bool nics_only;     /* Whether to only use NICs for buffer allocation */
    bool direct_mode;   /* Whether in direct mode (each GPU sends via own NIC) */
    int source_qp_index;   /* QP index that has the source GPU buffer */
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
    size_t part_size, offset;
    
    /* In direct mode, each GPU has its own full buffer */
    if (args->direct_mode) {
        part_size = buffer_size;
        offset = 0;
    } else if (args->num_source_gpus > 1) {
        /* Multi-source NVLink mode:
         * - buffer_size (QP buffer) = (original_buffer × num_sources) / num_qps
         * - Each source writes: original_buffer / num_qps
         * - Offset within QP buffer: source_index × (original_buffer / num_qps)
         */
        part_size = buffer_size / args->num_source_gpus;
        offset = args->source_gpu_index * part_size;
    } else if (args->num_source_gpus == 1) {
        /* Single-source NVLink mode: each thread sends full QP buffer */
        part_size = buffer_size;
        offset = 0;
    } else {
        /* NICs-only mode: buffer is partitioned by QP */
        part_size = buffer_size / args->num_qps;
        offset = qp_index * part_size;
    }
    
    cycles_t start_cycles, end_cycles;
    double total_time;
    double cpu_mhz;
    int i;
    
    /* Server just waits - no sending */
    if (args->is_server) {
        sleep(300);  /* 5 minutes should be enough */
        *args->bandwidth = 0.0;
        *args->status = 0;
        return NULL;
    }
    
    /* Multi-source NVLink mode: copy from source buffer to QP buffer, then RDMA */
    if (!args->nics_only && args->source_buffer && args->num_source_gpus > 0) {
        void *src_buffer = args->source_buffer;  /* Use separate source buffer (full size) */
        void *qp_buffer = rdma_get_local_buffer(ctx, qp_index);  /* QP buffer for RDMA */
        
        /* Calculate buffer part pointers
         * Multi-source mode:
         *   - Source offset: depends on target QP (which slice of source data to send)
         *   - Dest offset: depends on source GPU index (where to write in target)
         * Single-source mode:
         *   - Both read/write from start (full buffer)
         */
        void *src_part, *dst_part;
        size_t src_offset, dst_offset;
        if (args->num_source_gpus > 1) {
            /* Multi-source: read 128MB slice from 1GB source, write to target at offset */
            size_t source_buffer_size = DEFAULT_BUFFER_SIZE;  /* 1GB */
            size_t slice_size = source_buffer_size / args->num_qps;  /* 128MB */
            src_offset = qp_index * slice_size;
            dst_offset = offset;  /* = source_gpu_index * part_size */
            src_part = (char *)src_buffer + src_offset;
            dst_part = (char *)qp_buffer + dst_offset;
        } else {
            /* Single-source: send full buffer */
            src_offset = 0;
            dst_offset = 0;
            src_part = (char *)src_buffer;
            dst_part = (char *)qp_buffer;
        }
        
        /* Get CPU frequency */
        cpu_mhz = get_cpu_mhz(1);  /* Suppress CPU frequency warnings */
        if (cpu_mhz <= 0) {
            fprintf(stderr, "Failed to get CPU frequency\n");
            *args->status = -1;
            return NULL;
        }
        
        /* Warmup: Copy from source buffer to QP buffer */
        if (args->source_gpu_id != args->target_gpu_id && args->nvlink_ctx) {
            /* Cross-GPU: use NVLink */
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
        } else if (args->source_gpu_id == args->target_gpu_id) {
            /* Same GPU: use local cudaMemcpy */
            cudaError_t err = cudaSetDevice(args->source_gpu_id);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", args->source_gpu_id, cudaGetErrorString(err));
                *args->status = -1;
                return NULL;
            }
            for (i = 0; i < WARMUP_ITERATIONS; i++) {
                err = cudaMemcpy(dst_part, src_part, part_size, cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Local GPU copy warmup %d failed: %s\n", i, cudaGetErrorString(err));
                    *args->status = -1;
                    return NULL;
                }
            }
        }
        
        /* RDMA warmup */
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_write(ctx, qp_index, dst_offset, part_size, dst_offset) != 0) {
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
        
        /* Wait at start barrier before beginning timed work */
        if (args->start_barrier) {
            pthread_barrier_wait(args->start_barrier);
        }
        
        /* Timed transfers: NVLink + RDMA */
        start_cycles = get_cycles();
        
        /* Step 1: Copy from source buffer to QP buffer */
        if (args->source_gpu_id != args->target_gpu_id && args->nvlink_ctx) {
            /* Cross-GPU: use NVLink async */
            for (i = 0; i < args->iterations; i++) {
                if (nvlink_copy_gpu_to_gpu_async(args->nvlink_ctx, dst_part,
                                                  src_part, part_size) != 0) {
                    fprintf(stderr, "NVLink copy %d failed\n", i);
                    *args->status = -1;
                    return NULL;
                }
            }
            // printf("NVLink: GPU%d -> GPU%d, src=%p (off=%zu), dst=%p (off=%zu), size=%zu\n",
            //        args->nvlink_ctx->gpu0_id, args->nvlink_ctx->gpu1_id,
            //        src_part, src_offset, dst_part, dst_offset, part_size);
            
            /* Wait for NVLink to complete */
            if (nvlink_synchronize(args->nvlink_ctx) != 0) {
                fprintf(stderr, "NVLink sync failed\n");
                *args->status = -1;
                return NULL;
            }
        } else if (args->source_gpu_id == args->target_gpu_id) {
            /* Same GPU: local copy */
            cudaError_t err = cudaSetDevice(args->source_gpu_id);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", args->source_gpu_id, cudaGetErrorString(err));
                *args->status = -1;
                return NULL;
            }
            for (i = 0; i < args->iterations; i++) {
                err = cudaMemcpy(dst_part, src_part, part_size, cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Local GPU copy %d failed: %s\n", i, cudaGetErrorString(err));
                    *args->status = -1;
                    return NULL;
                }
            }
            // printf("Local copy: GPU%d src=%p (off=%zu), dst=%p (off=%zu), size=%zu\n",
            //        args->source_gpu_id, src_part, src_offset, dst_part, dst_offset, part_size);
        }
        
        for (i = 0; i < args->iterations; i++) {
            /* Step 2: RDMA write */
            if (rdma_write(ctx, qp_index, dst_offset, part_size, dst_offset) != 0) {
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
        
        return NULL;
    }
    
    /* Thread 0 uses RDMA writes only */
    /* Get CPU frequency */
    cpu_mhz = get_cpu_mhz(1);  /* Suppress CPU frequency warnings */
    if (cpu_mhz <= 0) {
        fprintf(stderr, "Failed to get CPU frequency\n");
        *args->status = -1;
        return NULL;
    }
    
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
    printf("  -S, --source-gpus LIST Comma-separated source GPU IDs for multi-source NVLink\n");
    printf("  -D, --direct          Each GPU sends directly via its own NIC (no NVLink)\n");
    printf("  -h, --help            Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  Server: %s -s -n mlx5_0,mlx5_1 -g 0,1\n", prog_name);
    printf("  Client: %s -n mlx5_0,mlx5_1 -g 0,1 -a 192.168.1.100\n", prog_name);
    printf("  Direct: %s -n mlx5_0,mlx5_1 -g 0,1 -a 192.168.1.100 --direct\n", prog_name);
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
    void **source_buffers = NULL;  /* Separate buffers on source GPUs (full size) */
    int num_qps = 0;
    int opt;
    int ret = 0;
    int i;
    char *nics_str = NULL;
    char *gpus_str = NULL;
    char *source_gpus_filter = NULL; /* filter the GPU to use as data source */
    
    /* Defaults */
    config.base_port = 18515;
    config.buffer_size = DEFAULT_BUFFER_SIZE;
    config.is_server = 0;
    config.server_addr = NULL;
    config.direct_mode = 0;
    
    /* Command line parsing */
    static struct option long_options[] = {
        {"nics", required_argument, 0, 'n'},
        {"server", no_argument, 0, 's'},
        {"addr", required_argument, 0, 'a'},
        {"port", required_argument, 0, 'p'},
        {"buffer", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'i'},
        {"gpus", required_argument, 0, 'g'},
        {"source-gpus", required_argument, 0, 'S'},
        {"help", no_argument, 0, 'h'},
        {"nics-only", no_argument, 0, 'c'},
        {"direct", no_argument, 0, 'D'},
        {0, 0, 0, 0}
    };
    
    while ((opt = getopt_long(argc, argv, "n:sa:p:b:i:g:S:h:D", long_options, NULL)) != -1) {
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
        case 'D':
            config.direct_mode = 1;
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
        case 'S':
            source_gpus_filter = optarg;
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
    
    int *target_gpu_ids = NULL;  /* Target GPU IDs (from --gpus) curreently all the GPUs are targets in non direct mode */
    int *source_gpu_ids_list = NULL;  /* Source GPUs (from --source-gpus) */
    int num_source_gpus = 0;
    int num_threads = 0;  /* Total number of threads to create */
    
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
        
        /* Save target GPU IDs */
        target_gpu_ids = calloc(num_qps, sizeof(int));
        memcpy(target_gpu_ids, gpu_ids, num_qps * sizeof(int));
        
        /* Parse source GPUs if specified */
        if (source_gpus_filter) {
            source_gpu_ids_list = calloc(MAX_THREADS, sizeof(int));
            num_source_gpus = parse_comma_separated_ints(source_gpus_filter, source_gpu_ids_list, MAX_THREADS);
            
            if (num_source_gpus <= 0) {
                fprintf(stderr, "Error: Invalid source-gpus\n");
                goto cleanup;
            }
            
            if (config.direct_mode) {
                /* Direct mode: Use only source GPU QPs and send data from source GPU to target GPU directly threw their own NICs */
                /* Filter NICs and GPUs to match source GPUs */
                if (num_source_gpus > num_qps) {
                    fprintf(stderr, "Error: More source GPUs (%d) than NICs (%d)\n", 
                            num_source_gpus, num_qps);
                    goto cleanup;
                }
                
                /* Update gpu_ids to only include source GPUs */
                for (i = 0; i < num_source_gpus; i++) {
                    gpu_ids[i] = source_gpu_ids_list[i];
                }
                /* Update num_qps to only use source GPU count */
                num_qps = num_source_gpus;
                num_threads = num_source_gpus;
                
                printf("Direct mode: %d source GPUs, %d QPs\n", num_source_gpus, num_qps);
            } else if (!config.nics_only) {
                /* NVLink mode: M source GPUs × N target QPs threads */
                num_threads = num_source_gpus * num_qps;
                
                /* QP buffers: allocate on target GPUs (for RDMA)
                 * Separate source buffers (full size) will be allocated below
                 */
                for (i = 0; i < num_qps; i++) {
                    gpu_ids[i] = target_gpu_ids[i];
                    printf("QP %d buffer on target GPU %d (for RDMA)\n", i, gpu_ids[i]);
                }
                
                printf("Multi-source NVLink mode: %d source GPUs × %d QPs = %d threads\n", 
                       num_source_gpus, num_qps, num_threads);
            } else {
                /* NICs-only with source-gpus: just use first source GPU */
                for (i = 0; i < num_qps; i++) {
                    gpu_ids[i] = target_gpu_ids[0];
                }
                num_threads = num_qps;
            }
        } else {
            /* No source-gpus filter */
            num_threads = num_qps;
        }
    } else {
        /* Default: all host memory */
        for (i = 0; i < num_qps; i++) {
            gpu_ids[i] = -1;
        }
        num_threads = num_qps;
    }
    
    /* Adjust buffer size for multi-source NVLink mode */
    size_t actual_buffer_size = config.buffer_size;
    if (!config.nics_only && !config.direct_mode && num_source_gpus > 0) {
        /* Each QP buffer = (buffer_size / num_qps) × num_source_gpus */
        /* This holds data from ALL source GPUs */
        actual_buffer_size = (config.buffer_size * num_source_gpus) / num_qps;
        printf("Multi-source NVLink: %d sources × %zu buffer / %d QPs = %zu bytes per QP\n", 
               num_source_gpus, config.buffer_size, num_qps, actual_buffer_size);
    }
    
    /* Setup config */
    config.nic_names = (const char **)nic_names;
    config.gpu_id = gpu_ids;
    config.num_qps = num_qps;
    config.buffer_size = actual_buffer_size;
    
    /* Initialize RDMA context */
    if (rdma_multi_qp_init(&config, &ctx, config.nics_only) != 0) {
        fprintf(stderr, "Failed to initialize RDMA context\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Allocate separate source buffers for multi-source NVLink mode */
    if (!config.direct_mode && !config.nics_only && num_source_gpus > 0) {
        source_buffers = calloc(num_source_gpus, sizeof(void*));
        if (!source_buffers) {
            fprintf(stderr, "Failed to allocate source buffers array\n");
            ret = 1;
            goto cleanup;
        }
        
        for (i = 0; i < num_source_gpus; i++) {
            int src_gpu = source_gpu_ids_list[i];
            size_t source_buffer_size = DEFAULT_BUFFER_SIZE;  /* Full 1GB per source */
            
            cudaError_t err = cudaSetDevice(src_gpu);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", src_gpu, cudaGetErrorString(err));
                ret = 1;
                goto cleanup;
            }
            
            err = cudaMalloc(&source_buffers[i], source_buffer_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to allocate source buffer on GPU %d: %s\n",
                        src_gpu, cudaGetErrorString(err));
                ret = 1;
                goto cleanup;
            }
            
            printf("Allocated source buffer on GPU %d: %zu bytes\n", src_gpu, source_buffer_size);
        }
    }
    
    /* Connect QPs */
    if (rdma_multi_qp_connect(ctx) != 0) {
        fprintf(stderr, "Failed to connect QPs\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Initialize buffers with test data (client/sender only) */
    if (!config.is_server) {
        printf("\n=== Initializing test data in buffers ===\n");
        
        if (config.direct_mode) {
            /* Direct mode: write test string at start of each QP buffer */
            for (i = 0; i < num_qps; i++) {
                cudaError_t err = cudaSetDevice(gpu_ids[i]);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", gpu_ids[i], cudaGetErrorString(err));
                    continue;
                }
                
                void *qp_buf = rdma_get_local_buffer(ctx, i);
                size_t buf_size = rdma_get_buffer_size(ctx);
                char test_str[128];
                snprintf(test_str, sizeof(test_str), "one buffer of size %zu on GPU_%d\n", buf_size, gpu_ids[i]);
                
                err = cudaMemcpy(qp_buf, test_str, strlen(test_str) + 1, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to write test data to GPU %d: %s\n", gpu_ids[i], cudaGetErrorString(err));
                }
                printf("Initialized QP %d buffer on GPU %d with test string\n", i, gpu_ids[i]);
            }
        } else if (source_buffers && num_source_gpus > 0) {
            /* NVLink mode: write test strings to slices of source buffers */
            for (i = 0; i < num_source_gpus; i++) {
                int src_gpu = source_gpu_ids_list[i];
                cudaError_t err = cudaSetDevice(src_gpu);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", src_gpu, cudaGetErrorString(err));
                    continue;
                }
                
                size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;  /* 128MB per slice */
                for (int slice_idx = 0; slice_idx < num_qps; slice_idx++) {
                    char test_str[128];
                    snprintf(test_str, sizeof(test_str), "buffer number %d on source gpu_%d\n", slice_idx, src_gpu);
                    
                    size_t offset = slice_idx * slice_size;
                    void *dst = (char *)source_buffers[i] + offset;
                    
                    err = cudaMemcpy(dst, test_str, strlen(test_str) + 1, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to write slice %d to GPU %d: %s\n", slice_idx, src_gpu, cudaGetErrorString(err));
                    }
                }
                printf("Initialized source buffer on GPU %d with %d test slices\n", src_gpu, num_qps);
            }
        }
        printf("=== Test data initialization complete ===\n\n");
    }
    
    /* Allocate thread arrays based on actual thread count */
    threads = calloc(num_threads, sizeof(pthread_t));
    args = calloc(num_threads, sizeof(struct thread_args));
    bandwidths = calloc(num_threads, sizeof(double));
    statuses = calloc(num_threads, sizeof(int));
    nvlink_ctxs = calloc(num_threads, sizeof(struct nvlink_context));
    
    if (!threads || !args || !bandwidths || !statuses || !nvlink_ctxs) {
        fprintf(stderr, "Failed to allocate memory\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Setup NVLink contexts (skip in direct mode) */
    printf("num_source_gpus: %d and config.direct_mode: %d\n", num_source_gpus, config.direct_mode);
    if (!config.is_server && !config.direct_mode && num_source_gpus > 0) {
        /* Multi-source NVLink: Create contexts for each source GPU × target GPU combination */
        printf("Creating NVLink contexts for %d source GPUs and %d target GPUs\n", num_source_gpus, num_qps);
        for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
            int src_gpu = source_gpu_ids_list[src_idx];
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                int tgt_gpu = target_gpu_ids[qp_idx];
                int thread_idx = src_idx * num_qps + qp_idx;
                
                if (src_gpu >= 0 && tgt_gpu >= 0 && src_gpu != tgt_gpu) {
                    if (nvlink_init_context(&nvlink_ctxs[thread_idx], src_gpu, tgt_gpu) != 0) {
                        fprintf(stderr, "Failed to initialize NVLink context thread %d (GPU%d->GPU%d)\n", 
                                thread_idx, src_gpu, tgt_gpu);
                        ret = 1;
                        goto cleanup;
                    } else {
                        printf("Successfully initialized NVLink context thread %d (GPU%d->GPU%d)\n",
                               thread_idx, src_gpu, tgt_gpu);
                    }
                }
            }
        }
    }
    /* Print paths being used */
    if (!config.is_server) {
        printf("Paths:\n");
        if (!config.direct_mode && num_source_gpus > 0) {
            /* Multi-source NVLink mode: show all source → target combinations */
            for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
                int src_gpu = source_gpu_ids_list[src_idx];
                printf("  Source GPU %d:\n", src_gpu);
                for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                    int tgt_gpu = target_gpu_ids[qp_idx];
                    if (src_gpu == tgt_gpu) {
                        printf("    Thread %d: GPU%d -> %s -> RDMA\n", 
                               src_idx * num_qps + qp_idx, src_gpu, nic_names[qp_idx]);
                    } else {
                        printf("    Thread %d: GPU%d -> NVLink -> GPU%d -> %s -> RDMA\n", 
                               src_idx * num_qps + qp_idx, src_gpu, tgt_gpu, nic_names[qp_idx]);
                    }
                }
            }
        } else {
            /* Direct or NICs-only mode: show per-QP paths */
            for (i = 0; i < num_qps; i++) {
                int src_gpu = gpu_ids[i];
                
                if (config.direct_mode) {
                    if (src_gpu >= 0) {
                        printf("  Thread %d: GPU%d -> %s -> RDMA (direct)\n", i, src_gpu, nic_names[i]);
                    } else {
                        printf("  Thread %d: Host -> %s -> RDMA (direct)\n", i, nic_names[i]);
                    }
                } else if (config.nics_only) {
                    if (src_gpu >= 0) {
                        printf("  Thread %d: GPU%d -> %s -> RDMA\n", i, src_gpu, nic_names[i]);
                    } else {
                        printf("  Thread %d: Host -> %s -> RDMA\n", i, nic_names[i]);
                    }
                }
            }
        }
        printf("\n");
    }
    
    /* Initialize barriers for synchronizing thread start/end */
    pthread_barrier_t start_barrier, end_barrier;
    int barriers_initialized = 0;
    if (!config.is_server) {
        /* Barriers need num_threads + 1 (threads + main thread) */
        if (pthread_barrier_init(&start_barrier, NULL, num_threads + 1) != 0) {
            fprintf(stderr, "Failed to initialize start barrier\n");
            ret = 1;
            goto cleanup;
        }
        if (pthread_barrier_init(&end_barrier, NULL, num_threads + 1) != 0) {
            fprintf(stderr, "Failed to initialize end barrier\n");
            pthread_barrier_destroy(&start_barrier);
            ret = 1;
            goto cleanup;
        }
        barriers_initialized = 1;
    }
    
    /* Setup thread arguments */
    if (!config.direct_mode && num_source_gpus > 0) {
        /* Multi-source NVLink mode: M source GPUs × N QPs threads */
        for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
            int src_gpu = source_gpu_ids_list[src_idx];
            int src_qp_idx = src_idx;  /* QP that has this source GPU's buffer */
            
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                int thread_idx = src_idx * num_qps + qp_idx;
                int tgt_gpu = target_gpu_ids[qp_idx];
                
                args[thread_idx].ctx = ctx;
                args[thread_idx].qp_index = qp_idx;  /* Which QP to use for RDMA */
                args[thread_idx].iterations = DEFAULT_ITERATIONS;
                args[thread_idx].is_server = config.is_server;
                args[thread_idx].source_gpu_id = src_gpu;
                args[thread_idx].target_gpu_id = tgt_gpu;
                args[thread_idx].source_qp_index = src_qp_idx;  /* QP with source buffer */
                args[thread_idx].num_source_gpus = num_source_gpus;
                args[thread_idx].source_gpu_index = src_idx;
                args[thread_idx].source_buffer = source_buffers ? source_buffers[src_idx] : NULL;
                
                /* Set NVLink context if source != target */
                args[thread_idx].nvlink_ctx = (src_gpu >= 0 && tgt_gpu >= 0 && src_gpu != tgt_gpu) 
                                               ? &nvlink_ctxs[thread_idx] : NULL;
                
                args[thread_idx].num_qps = num_qps;
                args[thread_idx].start_barrier = config.is_server ? NULL : &start_barrier;
                args[thread_idx].end_barrier = config.is_server ? NULL : &end_barrier;
                args[thread_idx].bandwidth = &bandwidths[thread_idx];
                args[thread_idx].status = &statuses[thread_idx];
                args[thread_idx].nics_only = config.nics_only;
                args[thread_idx].direct_mode = config.direct_mode;
                statuses[thread_idx] = -1;
            }
        }
    } else {
        /* Direct mode or single-source: One thread per QP */
        for (i = 0; i < num_threads; i++) {
            int src_gpu = gpu_ids[i];
            int tgt_gpu = target_gpu_ids ? target_gpu_ids[i] : gpu_ids[i];
            
            args[i].ctx = ctx;
            args[i].qp_index = i;
            args[i].iterations = DEFAULT_ITERATIONS;
            args[i].is_server = config.is_server;
            args[i].source_gpu_id = src_gpu;
            args[i].target_gpu_id = tgt_gpu;
            args[i].source_qp_index = i;
            args[i].num_source_gpus = (num_source_gpus > 0) ? 1 : 0;
            args[i].source_gpu_index = 0;
            args[i].source_buffer = NULL;  /* No separate source buffer in direct/nics-only */
            
            args[i].nvlink_ctx = NULL;  /* No NVLink in direct/nics-only mode */
            
            args[i].num_qps = num_qps;
            args[i].start_barrier = config.is_server ? NULL : &start_barrier;
            args[i].end_barrier = config.is_server ? NULL : &end_barrier;
            args[i].bandwidth = &bandwidths[i];
            args[i].status = &statuses[i];
            args[i].nics_only = config.nics_only;
            args[i].direct_mode = config.direct_mode;
            statuses[i] = -1;
        }
    }
    
    /* Get CPU frequency for wall-clock time measurement */
    double cpu_mhz = get_cpu_mhz(1);  /* Suppress CPU frequency warnings */
    if (cpu_mhz <= 0) {
        fprintf(stderr, "Failed to get CPU frequency\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Measure wall-clock time for total bandwidth calculation */
    cycles_t total_start_cycles, total_end_cycles;
    size_t total_buffer_size = rdma_get_buffer_size(ctx);
    
    /* Create threads */
    for (i = 0; i < num_threads; i++) {
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
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        if (statuses[i] != 0) {
            fprintf(stderr, "Thread %d failed\n", i);
            ret = 1;
        }
    }
    
    /* Calculate total wall-clock time */
    double total_time = (double)(total_end_cycles - total_start_cycles) / (cpu_mhz * 1e6);
    
    printf("Total elapsed time: %.6f seconds\n", total_time);
    
    /* Calculate total bandwidth based on wall-clock time */
    double total_bw = 0.0;
    int success_count = 0;
    for (i = 0; i < num_threads; i++) {
        if (statuses[i] == 0) {
            success_count++;
        }
    }
    if (success_count > 0 && total_time > 0.0) {
        /* Calculate total data transferred */
        size_t total_data;
        if (config.direct_mode) {
            /* Direct mode: each source GPU sends full buffer */
            printf("default initial buffer_size: %d\n", DEFAULT_BUFFER_SIZE);
            printf("total_buffer_size: %zu and num_threads: %d\n", total_buffer_size, num_threads);
            total_data = total_buffer_size * num_threads;
        } else if (!config.nics_only && num_source_gpus > 0) {
            /* Multi-source NVLink: total = buffer_per_QP × num_QPs  = (initial buffer_size * num_source_gpus) / num_qps × num_QPs  = initial buffer_size * num_source_gpus */
            printf("default initial buffer_size: %d\n", DEFAULT_BUFFER_SIZE);
            printf("total_buffer_size: %zu and num_qps: %d and num_source_gpus: %d\n", total_buffer_size, num_qps, num_source_gpus);
            total_data = total_buffer_size * num_qps;
        } else {
            /* NICs-only or single-source: partitioned buffer */
            total_data = total_buffer_size;
        }
        total_bw = (total_data * DEFAULT_ITERATIONS) / (total_time * 1e9);
        /* Print total bandwidth in green */
        printf("\033[0;32mTotal bandwidth: %.2f GB/s\033[0m\n", total_bw);
    }
    
    /* Verify transferred data */
    printf("\n=== Verifying transferred data ===\n");
    
    if (config.direct_mode) {
        /* Direct mode: read test string from each QP buffer */
        for (i = 0; i < num_qps; i++) {
            cudaError_t err = cudaSetDevice(gpu_ids[i]);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", gpu_ids[i], cudaGetErrorString(err));
                continue;
            }
            
            void *qp_buf = rdma_get_local_buffer(ctx, i);
            char host_buf[256] = {0};
            
            err = cudaMemcpy(host_buf, qp_buf, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to read from GPU %d: %s\n", gpu_ids[i], cudaGetErrorString(err));
                continue;
            }
            
            host_buf[sizeof(host_buf) - 1] = '\0';
            printf("GPU_%d read from buffer: %s", gpu_ids[i], host_buf);
        }
    } else if (num_source_gpus > 0) {
        /* NVLink mode: read all sections from each target QP buffer */
        size_t qp_buf_size = rdma_get_buffer_size(ctx);
        size_t section_size = qp_buf_size / num_source_gpus;
        
        for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
            int tgt_gpu = target_gpu_ids[qp_idx];
            cudaError_t err = cudaSetDevice(tgt_gpu);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", tgt_gpu, cudaGetErrorString(err));
                continue;
            }
            
            void *qp_buf = rdma_get_local_buffer(ctx, qp_idx);
            printf("Target GPU_%d (QP %d) sections:\n", tgt_gpu, qp_idx);
            
            for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
                char host_buf[256] = {0};
                size_t offset = src_idx * section_size;
                void *src = (char *)qp_buf + offset;
                
                err = cudaMemcpy(host_buf, src, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    fprintf(stderr, "  Failed to read section %d: %s\n", src_idx, cudaGetErrorString(err));
                    continue;
                }
                
                host_buf[sizeof(host_buf) - 1] = '\0';
                printf("  Section %d: %s", src_idx, host_buf);
            }
        }
    }
    
    printf("=== Data verification complete ===\n\n");
    
    /* Skip cleanup to avoid segfault - just return */
    return ret;
    
cleanup:
    /* Free source buffers if allocated */
    if (source_buffers) {
        for (i = 0; i < num_source_gpus; i++) {
            if (source_buffers[i]) {
                cudaSetDevice(source_gpu_ids_list[i]);
                cudaFree(source_buffers[i]);
            }
        }
        free(source_buffers);
    }
    
    /* Minimal cleanup for error paths - skip problematic cleanup calls */
    if (barriers_initialized) {
        pthread_barrier_destroy(&start_barrier);
        pthread_barrier_destroy(&end_barrier);
    }
    /* Skip nvlink_cleanup_context and rdma_multi_qp_cleanup to avoid segfault */
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
    if (target_gpu_ids) free(target_gpu_ids);
    if (source_gpu_ids_list) free(source_gpu_ids_list);
    
    return ret;
}

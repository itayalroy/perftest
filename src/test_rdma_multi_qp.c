/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * Test program for RDMA Multi-QP API
 * Tests bandwidth with 2 QPs and 2 threads
 */

#include "config.h"
#include "rdma_multi_qp.h"
#include "get_clock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <pthread.h>

#define DEFAULT_BUFFER_SIZE (1024 * 1024 * 64)  /* 64 MB */
#define DEFAULT_ITERATIONS 256
#define WARMUP_ITERATIONS 256

struct thread_args {
    rdma_multi_qp_context_t *ctx;
    int qp_index;
    int iterations;
    int is_server;         /* 1 if server (just wait), 0 if client (send) */
    double *bandwidth;  /* Output: bandwidth in GB/s */
    int *status;        /* Output: 0 on success */
};

static void *write_thread(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    rdma_multi_qp_context_t *ctx = args->ctx;
    int qp_index = args->qp_index;
    size_t buffer_size = rdma_get_buffer_size(ctx);
    cycles_t start_cycles, end_cycles;
    double total_time;
    double cpu_mhz;
    int i;
    
    /* Server just waits - no sending */
    if (args->is_server) {
        printf("Server thread for QP %d: Waiting for client to complete...\n", qp_index);
        /* Wait for a long time - client will finish first */
        sleep(300);  /* 5 minutes should be enough */
        *args->bandwidth = 0.0;
        *args->status = 0;
        printf("Server thread for QP %d: Done waiting\n", qp_index);
        return NULL;
    }
    
    /* Client sends RDMA writes */
    /* Get CPU frequency */
    cpu_mhz = get_cpu_mhz(0);
    if (cpu_mhz <= 0) {
        fprintf(stderr, "Failed to get CPU frequency\n");
        *args->status = -1;
        return NULL;
    }
    
    printf("Thread for QP %d: Starting warmup...\n", qp_index);
    
    /* Warmup */
    for (i = 0; i < WARMUP_ITERATIONS; i++) {
        if (rdma_write(ctx, qp_index, 0, buffer_size, 0) != 0) {
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
    
    /* Timed transfers */
    start_cycles = get_cycles();
    
    for (i = 0; i < args->iterations; i++) {
        if (rdma_write(ctx, qp_index, 0, buffer_size, 0) != 0) {
            fprintf(stderr, "QP %d: Write %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }

    /* Poll for all timed test completions */
    for (i = 0; i < args->iterations; i++) {
        if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
            fprintf(stderr, "QP %d: Completion %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }
    
    end_cycles = get_cycles();
    total_time = (double)(end_cycles - start_cycles) / (cpu_mhz * 1e6);  /* Convert cycles to seconds */
    
    /* Calculate bandwidth */
    *args->bandwidth = (buffer_size * args->iterations) / (total_time * 1e9);  /* GB/s */
    *args->status = 0;
    
    printf("Thread for QP %d: Completed %d iterations in %.6f seconds\n",
           qp_index, args->iterations, total_time);
    
    return NULL;
}

static void print_usage(const char *prog_name)
{
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -n, --nic0 NAME       First NIC device name (e.g., mlx5_0)\n");
    printf("  -N, --nic1 NAME       Second NIC device name (e.g., mlx5_1)\n");
    printf("  -s, --server          Run as server (default: client)\n");
    printf("  -a, --addr ADDR       Server IP address (default: 127.0.0.1)\n");
    printf("  -p, --port PORT       Base port (default: 18515)\n");
    printf("  -b, --buffer SIZE     Buffer size in bytes (default: 64MB)\n");
    printf("  -i, --iterations NUM  Number of iterations (default: 100)\n");
    printf("  -g, --gpu ID          GPU ID for buffer allocation (-1 for host, default: -1)\n");
    printf("  -h, --help            Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  Server: %s -s -n mlx5_0 -N mlx5_1\n", prog_name);
    printf("  Client: %s -n mlx5_0 -N mlx5_1 -a 192.168.1.100\n", prog_name);
}

int main(int argc, char *argv[])
{
    struct rdma_multi_qp_config config = {0};
    rdma_multi_qp_context_t *ctx = NULL;
    pthread_t threads[2];
    struct thread_args args[2];
    double bandwidths[2];
    int statuses[2];
    int opt;
    int ret = 0;
    int i;
    
    /* Defaults */
    config.base_port = 18515;
    config.buffer_size = DEFAULT_BUFFER_SIZE;
    config.is_server = 0;
    config.gpu_id = -1;
    config.server_addr = NULL;
    
    /* Command line parsing */
    static struct option long_options[] = {
        {"nic0", required_argument, 0, 'n'},
        {"nic1", required_argument, 0, 'N'},
        {"server", no_argument, 0, 's'},
        {"addr", required_argument, 0, 'a'},
        {"port", required_argument, 0, 'p'},
        {"buffer", required_argument, 0, 'b'},
        {"iterations", required_argument, 0, 'i'},
        {"gpu", required_argument, 0, 'g'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    while ((opt = getopt_long(argc, argv, "n:N:sa:p:b:i:g:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'n':
            config.nic_names[0] = optarg;
            break;
        case 'N':
            config.nic_names[1] = optarg;
            break;
        case 's':
            config.is_server = 1;
            break;
        case 'a':
            config.server_addr = optarg;
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
            config.gpu_id = atoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!config.nic_names[0] || !config.nic_names[1]) {
        fprintf(stderr, "Error: Both NIC names must be specified\n");
        print_usage(argv[0]);
        return 1;
    }
    
    printf("\n========================================\n");
    printf("RDMA Multi-QP Bandwidth Test\n");
    printf("========================================\n");
    printf("Mode: %s\n", config.is_server ? "Server" : "Client");
    printf("NIC 0: %s\n", config.nic_names[0]);
    printf("NIC 1: %s\n", config.nic_names[1]);
    printf("Base port: %d\n", config.base_port);
    printf("Buffer size: %zu bytes (%.2f MB)\n", 
           config.buffer_size, config.buffer_size / (1024.0 * 1024.0));
    printf("GPU ID: %d\n", config.gpu_id);
    if (config.server_addr) {
        printf("Server address: %s\n", config.server_addr);
    }
    printf("Iterations per thread: %d\n", DEFAULT_ITERATIONS);
    printf("========================================\n\n");
    
    /* Initialize RDMA context */
    printf("Initializing RDMA context...\n");
    if (rdma_multi_qp_init(&config, &ctx) != 0) {
        fprintf(stderr, "Failed to initialize RDMA context\n");
        return 1;
    }
    printf("RDMA context initialized\n\n");
    
    /* Connect QPs */
    printf("Connecting QPs...\n");
    if (rdma_multi_qp_connect(ctx) != 0) {
        fprintf(stderr, "Failed to connect QPs\n");
        rdma_multi_qp_cleanup(ctx);
        return 1;
    }
    printf("QPs connected\n\n");
    
    /* Setup thread arguments */
    for (i = 0; i < 2; i++) {
        args[i].ctx = ctx;
        args[i].qp_index = i;
        args[i].iterations = DEFAULT_ITERATIONS;
        args[i].is_server = config.is_server;
        args[i].bandwidth = &bandwidths[i];
        args[i].status = &statuses[i];
        statuses[i] = -1;
    }
    
    /* Create threads */
    printf("Creating threads...\n");
    for (i = 0; i < 2; i++) {
        if (pthread_create(&threads[i], NULL, write_thread, &args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            ret = 1;
            goto cleanup;
        }
    }
    
    /* Wait for threads */
    printf("Waiting for threads to complete...\n\n");
    for (i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
        if (statuses[i] != 0) {
            fprintf(stderr, "Thread %d failed\n", i);
            ret = 1;
        }
    }
    
    /* Print results */
    printf("\n========================================\n");
    printf("Results:\n");
    printf("========================================\n");
    for (i = 0; i < 2; i++) {
        if (statuses[i] == 0) {
            printf("QP %d bandwidth: %.2f GB/s\n", i, bandwidths[i]);
        } else {
            printf("QP %d: FAILED\n", i);
        }
    }
    if (statuses[0] == 0 && statuses[1] == 0) {
        printf("Total bandwidth: %.2f GB/s\n", bandwidths[0] + bandwidths[1]);
    }
    printf("========================================\n\n");
    
cleanup:
    /* Cleanup */
    printf("Cleaning up...\n");
    rdma_multi_qp_cleanup(ctx);
    
    return ret;
}


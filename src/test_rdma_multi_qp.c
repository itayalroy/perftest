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
#define DEFAULT_ITERATIONS 10
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
    bool reassembly;    /* Whether reassembly mode is enabled (allow-nvlink only) */
    int source_qp_index;   /* QP index that has the source GPU buffer */
    void **reassembly_buffers;  /* Array of reassembly buffers (receiver only, allow-nvlink) */
    struct nvlink_context **reassembly_nvlink_ctxs;  /* NVLink contexts for reassembly */
    pthread_barrier_t *reassembly_barrier;  /* Barrier after reassembly complete */
    pthread_mutex_t *qp_mutex;  /* Mutex for serializing QP access (multi-source mode) */
    pthread_barrier_t *iteration_barrier;  /* Barrier to sync between iterations */
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

/* Receiver thread for allow-nvlink mode with reassembly */
static void *receive_thread_with_reassembly(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    rdma_multi_qp_context_t *ctx = args->ctx;
    int qp_index = args->qp_index;
    size_t slice_size = DEFAULT_BUFFER_SIZE / args->num_qps;  /* 128MB per slice */
    void *qp_buffer = rdma_get_local_buffer(ctx, qp_index);
    int src_idx;
    
    /* Wait at start barrier */
    if (args->start_barrier) {
        pthread_barrier_wait(args->start_barrier);
    }
    
    /* Loop over iterations */
    for (int iter = 0; iter < args->iterations; iter++) {
        /* Post M receive WQEs (one for each source GPU) for this iteration */
        for (src_idx = 0; src_idx < args->num_source_gpus; src_idx++) {
            if (rdma_post_receive(ctx, qp_index, 0, 1) != 0) {
                fprintf(stderr, "QP %d: Failed to post receive %d for iteration %d\n", qp_index, src_idx, iter);
                *args->status = -1;
                return NULL;
            }
        }
        
        /* Poll M times (once for each source GPU) and reassemble */
        for (src_idx = 0; src_idx < args->num_source_gpus; src_idx++) {
            /* Poll for completion with immediate data */
            uint32_t source_gpu_idx = 0;
        if (rdma_poll_completion_with_imm(ctx, qp_index, -1, &source_gpu_idx) != 0) {
            fprintf(stderr, "QP %d: Failed to poll completion with imm %d\n", qp_index, src_idx);
            *args->status = -1;
            return NULL;
        }
        
            /* Reassemble: Copy from QP buffer to reassembly buffer */
            if (source_gpu_idx < args->num_source_gpus && args->reassembly_buffers) {
                /* Calculate source and destination pointers */
                void *dst_buffer = args->reassembly_buffers[source_gpu_idx];
                size_t dst_offset = qp_index * slice_size;  /* Where in reassembly buffer */
                void *dst_ptr = (char *)dst_buffer + dst_offset;
                
                /* QP buffer contains data from ALL source GPUs - extract the section for this source */
                size_t src_offset_in_qp = source_gpu_idx * slice_size;  /* Which section in QP buffer */
                void *src_ptr = (char *)qp_buffer + src_offset_in_qp;
                
                /* Use NVLink context for reassembly if available */
                /* Context index: qp_index * num_source_gpus + source_gpu_idx */
                int ctx_idx = qp_index * args->num_source_gpus + source_gpu_idx;
                struct nvlink_context *nvlink_ctx = args->reassembly_nvlink_ctxs ? 
                                                     args->reassembly_nvlink_ctxs[ctx_idx] : NULL;
                
                if (nvlink_ctx) {
                    /* Cross-GPU copy using NVLink */
                    if (nvlink_copy_gpu_to_gpu_async(nvlink_ctx, dst_ptr, src_ptr, slice_size) != 0) {
                        fprintf(stderr, "QP %d: NVLink reassembly failed for source GPU %u\n", 
                                qp_index, source_gpu_idx);
                        *args->status = -1;
                        return NULL;
                    }
                    if (nvlink_synchronize(nvlink_ctx) != 0) {
                        fprintf(stderr, "QP %d: NVLink sync failed\n", qp_index);
                        *args->status = -1;
                        return NULL;
                    }
                } else {
                    /* Local GPU copy */
                    cudaError_t err = cudaSetDevice(args->target_gpu_id);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "QP %d: Failed to set device %d: %s\n", 
                                qp_index, args->target_gpu_id, cudaGetErrorString(err));
                        *args->status = -1;
                        return NULL;
                    }
                    err = cudaMemcpy(dst_ptr, src_ptr, slice_size, cudaMemcpyDeviceToDevice);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "QP %d: Reassembly copy failed: %s\n", 
                                qp_index, cudaGetErrorString(err));
                        *args->status = -1;
                        return NULL;
                    }
                }
            }
        }  /* End for (src_idx...) loop */
        
        /* Wait for all QP threads to finish reassembly for this iteration */
        if (args->reassembly_barrier) {
            pthread_barrier_wait(args->reassembly_barrier);
        }
        
        /* Wait at iteration barrier (synchronize before next iteration) */
        if (args->iteration_barrier) {
            pthread_barrier_wait(args->iteration_barrier);
        }
    }
    
    /* Wait at end barrier after all iterations complete */
    if (args->end_barrier) {
        pthread_barrier_wait(args->end_barrier);
    }
    
    *args->bandwidth = 0.0;  /* Bandwidth calculated on sender side */
    *args->status = 0;
    return NULL;
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
    
    /* Calculate total data transferred per iteration (across all threads) */
    size_t total_data_per_iter;
    if (args->nics_only && args->num_source_gpus == 0) {
        /* NICs-only mode: total data = buffer_size (partitioned across QPs) */
        total_data_per_iter = buffer_size;
    } else {
        /* Direct, single-source, multi-source modes: total = buffer_size * num_qps */
        total_data_per_iter = buffer_size * args->num_qps;
    }
    
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
            /* Single-source: each QP sends its corresponding slice from the source buffer */
            size_t source_buffer_size = DEFAULT_BUFFER_SIZE;  /* 1GB */
            size_t slice_size = source_buffer_size / args->num_qps;  /* 128MB per slice */
            src_offset = qp_index * slice_size;  /* Each QP reads different slice */
            dst_offset = 0;  /* Write to start of QP buffer */
            src_part = (char *)src_buffer + src_offset;
            dst_part = (char *)qp_buffer + dst_offset;
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
            /* Same GPU: use local async copy + sync (match NVLink path) */
            cudaError_t err = cudaSetDevice(args->source_gpu_id);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", args->source_gpu_id, cudaGetErrorString(err));
                *args->status = -1;
                return NULL;
            }
            for (i = 0; i < WARMUP_ITERATIONS; i++) {
                err = cudaMemcpyAsync(dst_part, src_part, part_size, cudaMemcpyDeviceToDevice, 0);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Local GPU copy warmup %d failed: %s\n", i, cudaGetErrorString(err));
                    *args->status = -1;
                    return NULL;
                }
            }
            /* Synchronize to ensure GPU writes are visible to RDMA NIC */
            err = cudaStreamSynchronize(0);
            if (err != cudaSuccess) {
                fprintf(stderr, "Local GPU warmup sync failed: %s\n", cudaGetErrorString(err));
                *args->status = -1;
                return NULL;
            }
        }
        
        /* RDMA warmup (serialize QP access in multi-source mode) */
        if (args->qp_mutex) {
            pthread_mutex_lock(args->qp_mutex);
        }
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_write(ctx, qp_index, dst_offset, part_size, dst_offset) != 0) {
                fprintf(stderr, "QP %d: RDMA warmup write %d failed\n", qp_index, i);
                *args->status = -1;
                if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                return NULL;
            }
        }
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "QP %d: RDMA warmup completion %d failed\n", qp_index, i);
                *args->status = -1;
                if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                return NULL;
            }
        }
        if (args->qp_mutex) {
            pthread_mutex_unlock(args->qp_mutex);
        }
        
        /* Wait at start barrier before beginning timed work */
        if (args->start_barrier) {
            pthread_barrier_wait(args->start_barrier);
        }
        
        /* Timed transfers: NVLink + RDMA in a loop over iterations */
        cycles_t total_start_cycles = get_cycles();
        
        for (i = 0; i < args->iterations; i++) {
            start_cycles = get_cycles();  /* Start timing this iteration */
            /* Step 1: Copy from source buffer to QP buffer */
            if (args->source_gpu_id != args->target_gpu_id && args->nvlink_ctx) {
                /* Cross-GPU: use NVLink async */
                if (nvlink_copy_gpu_to_gpu_async(args->nvlink_ctx, dst_part,
                                                  src_part, part_size) != 0) {
                    fprintf(stderr, "NVLink copy failed (iteration %d)\n", i);
                    *args->status = -1;
                    return NULL;
                }
                /* Wait for NVLink to complete */
                if (nvlink_synchronize(args->nvlink_ctx) != 0) {
                    fprintf(stderr, "NVLink sync failed (iteration %d)\n", i);
                    *args->status = -1;
                    return NULL;
                }
            } else if (args->source_gpu_id == args->target_gpu_id) {
                /* Same GPU: async copy + sync to ensure visibility to RDMA NIC */
                cudaError_t err = cudaSetDevice(args->source_gpu_id);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s (iteration %d)\n", args->source_gpu_id, cudaGetErrorString(err), i);
                    *args->status = -1;
                    return NULL;
                }
                err = cudaMemcpyAsync(dst_part, src_part, part_size, cudaMemcpyDeviceToDevice, 0);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Local GPU copy failed: %s (iteration %d)\n", cudaGetErrorString(err), i);
                    *args->status = -1;
                    return NULL;
                }
                /* Synchronize to ensure GPU writes are visible to RDMA NIC */
                err = cudaStreamSynchronize(0);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Local GPU sync failed: %s (iteration %d)\n", cudaGetErrorString(err), i);
                    *args->status = -1;
                    return NULL;
                }
            }

            /* Step 2: RDMA write (serialize QP access in multi-source mode) */
            if (args->qp_mutex) {
                pthread_mutex_lock(args->qp_mutex);
            }
            if (rdma_write(ctx, qp_index, dst_offset, part_size, dst_offset) != 0) {
                fprintf(stderr, "RDMA write failed (iteration %d)\n", i);
                *args->status = -1;
                if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                return NULL;
            }
            /* Poll for completion of RDMA write */
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "RDMA completion failed (iteration %d)\n", i);
                *args->status = -1;
                if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                return NULL;
            }
            
            /* Step 3: Send completion signal with immediate (only if reassembly is enabled) */
            if (args->reassembly) {
                if (rdma_write_with_imm(ctx, qp_index, dst_offset, 0, dst_offset, args->source_gpu_index) != 0) {
                    fprintf(stderr, "RDMA write_with_imm failed (iteration %d)\n", i);
                    *args->status = -1;
                    if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                    return NULL;
                }
                if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                    fprintf(stderr, "RDMA write_with_imm completion failed (iteration %d)\n", i);
                    *args->status = -1;
                    if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                    return NULL;
                }
            }
            if (args->qp_mutex) {
                pthread_mutex_unlock(args->qp_mutex);
            }
            
            /* End timing for this iteration */
            /* Wait at iteration barrier (synchronize all threads before next iteration) */
            if (args->iteration_barrier) {
                pthread_barrier_wait(args->iteration_barrier);
            }
        }

        /* End total timing */
        cycles_t total_end_cycles = get_cycles();
        total_time = (double)(total_end_cycles - total_start_cycles) / (cpu_mhz * 1e6);
        
        /* Calculate bandwidth (total data transferred: total_data_per_iter * iterations) */
        *args->bandwidth = (total_data_per_iter * args->iterations) / (total_time * 1e9);
        *args->status = 0;
        
        /* Wait at end barrier before signal-back */
        if (args->end_barrier) {
            pthread_barrier_wait(args->end_barrier);
        }
        
        return NULL;
    }
    
    /* Direct/NICs-only mode */
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
    
    /* Timed transfers with per-iteration timing */
    cycles_t total_start_cycles = get_cycles();
    
    for (i = 0; i < args->iterations; i++) {
        start_cycles = get_cycles();
        
        #ifdef HAVE_CUDA
        nvtxRangePushA("rdma_write");
        #endif
        if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
            #ifdef HAVE_CUDA
            nvtxRangePop();
            #endif
            fprintf(stderr, "QP %d: Write failed\n", qp_index);
            *args->status = -1;
            return NULL;
        }
        #ifdef HAVE_CUDA
        nvtxRangePop();
        #endif
        
        if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
            fprintf(stderr, "QP %d: Completion failed\n", qp_index);
            *args->status = -1;
            return NULL;
        }
        
        /* Wait at iteration barrier (synchronize all threads between iterations) */
        if (args->iteration_barrier) {
            pthread_barrier_wait(args->iteration_barrier);
        }
    }

    /* Calculate bandwidth (total data transferred: total_data_per_iter * iterations) */
    cycles_t total_end_cycles = get_cycles();
    total_time = (double)(total_end_cycles - total_start_cycles) / (cpu_mhz * 1e6);
    
    /* Calculate bandwidth (total data transferred: total_data_per_iter * iterations) */
    *args->bandwidth = (total_data_per_iter * args->iterations) / (total_time * 1e9);
    *args->status = 0;
    
    /* Wait at end barrier after all iterations complete */
    if (args->end_barrier) {
        pthread_barrier_wait(args->end_barrier);
    }
    
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
    printf("  -R, --reassembly          Enable reassembly of data from multiple source GPUs only when using NVLink\n");
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
    config.nics_only = 0;
    config.direct_mode = 0;
    config.reassembly = 0;
    
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
        {"reassembly", no_argument, 0, 'R'},
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
        case 'R':
            config.reassembly = 1;
            if (config.direct_mode || config.nics_only) {
                fprintf(stderr, "Error: --reassembly is not supported in direct/nics-only mode\n");
                return 1;
            }
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
                /* NVLink mode: M source GPUs × N target QPs threads (sender only) */
                if (config.is_server) {
                    /* Receiver: only need N QP threads (one per QP) */
                    num_threads = num_qps;
                } else {
                    /* Sender: need M × N threads (one per source GPU × QP combination) */
                    num_threads = num_source_gpus * num_qps;
                }
                
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
    
    /* Allocate separate source buffers for multi-source NVLink mode (sender only) */
    if (!config.is_server && !config.direct_mode && !config.nics_only && num_source_gpus > 0) {
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
    
    /* Declare reassembly variables (used conditionally throughout main) */
    void **reassembly_buffers = NULL;
    struct nvlink_context **reassembly_nvlink_ctxs = NULL;
    pthread_barrier_t reassembly_barrier;
    int reassembly_barrier_initialized = 0;
    
    /* Per-QP mutexes for serializing RDMA operations in multi-source mode */
    pthread_mutex_t *qp_mutexes = NULL;
    int qp_mutexes_initialized = 0;
    
    /* Allocate reassembly buffers for receiver in allow-nvlink mode */
    if (config.reassembly) {
        if (config.is_server && !config.direct_mode && !config.nics_only && num_source_gpus > 0) {
            printf("\n=== Allocating reassembly buffers on receiver ===\n");
            reassembly_buffers = calloc(num_source_gpus, sizeof(void*));
            if (!reassembly_buffers) {
                fprintf(stderr, "Failed to allocate reassembly buffers array\n");
                ret = 1;
                goto cleanup;
            }
            
            size_t reassembly_buffer_size = DEFAULT_BUFFER_SIZE;  /* 1GB per source GPU */
            for (i = 0; i < num_source_gpus; i++) {
                int reassembly_gpu = source_gpu_ids_list[i];  /* Use same GPU IDs as source */
                
                cudaError_t err = cudaSetDevice(reassembly_gpu);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d for reassembly: %s\n",
                            reassembly_gpu, cudaGetErrorString(err));
                    ret = 1;
                    goto cleanup;
                }
                
                err = cudaMalloc(&reassembly_buffers[i], reassembly_buffer_size);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to allocate reassembly buffer on GPU %d: %s\n",
                            reassembly_gpu, cudaGetErrorString(err));
                    ret = 1;
                    goto cleanup;
                }
                
                printf("Allocated reassembly buffer on GPU %d: %zu bytes\n", reassembly_gpu, reassembly_buffer_size);
            }
            
            /* Create NVLink contexts for reassembly (N QPs × M source GPUs) */
            printf("Creating NVLink contexts for reassembly (all-to-all GPU pairs)\n");
            reassembly_nvlink_ctxs = calloc(num_qps * num_source_gpus, sizeof(struct nvlink_context*));
            if (!reassembly_nvlink_ctxs) {
                fprintf(stderr, "Failed to allocate reassembly NVLink contexts array\n");
                ret = 1;
                goto cleanup;
            }

            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                int qp_gpu = target_gpu_ids[qp_idx];  /* QP buffer GPU */
                
                for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
                    int reassembly_gpu = source_gpu_ids_list[src_idx];  /* Reassembly buffer GPU */
                    int ctx_idx = qp_idx * num_source_gpus + src_idx;
                    
                    if (qp_gpu != reassembly_gpu) {
                        reassembly_nvlink_ctxs[ctx_idx] = malloc(sizeof(struct nvlink_context));
                        if (!reassembly_nvlink_ctxs[ctx_idx]) {
                            fprintf(stderr, "Failed to allocate reassembly NVLink context [QP%d->GPU%d]\n", 
                                    qp_idx, src_idx);
                            ret = 1;
                            goto cleanup;
                        }
                        
                        if (nvlink_init_context(reassembly_nvlink_ctxs[ctx_idx], qp_gpu, reassembly_gpu) != 0) {
                            fprintf(stderr, "Failed to init NVLink context for reassembly: GPU%d -> GPU%d\n",
                                    qp_gpu, reassembly_gpu);
                            ret = 1;
                            goto cleanup;
                        }
                        printf("Reassembly NVLink [QP%d->SrcGPU%d]: GPU%d -> GPU%d\n", 
                               qp_idx, src_idx, qp_gpu, reassembly_gpu);
                    } else {
                        reassembly_nvlink_ctxs[ctx_idx] = NULL;  /* Same GPU, no NVLink needed */
                    }
                }
            }
            
            /* Initialize reassembly barrier (for N QP threads + main thread on receiver) */
            if (pthread_barrier_init(&reassembly_barrier, NULL, num_qps + 1) != 0) {
                fprintf(stderr, "Failed to initialize reassembly barrier\n");
                ret = 1;
                goto cleanup;
            }
            reassembly_barrier_initialized = 1;
        }
    }
    
    /* Initialize per-QP mutexes for multi-source NVLink mode on sender */
    if (!config.is_server && !config.direct_mode && !config.nics_only && num_source_gpus > 1) {
        qp_mutexes = calloc(num_qps, sizeof(pthread_mutex_t));
        if (!qp_mutexes) {
            fprintf(stderr, "Failed to allocate QP mutexes\n");
            ret = 1;
            goto cleanup;
        }
        for (i = 0; i < num_qps; i++) {
            if (pthread_mutex_init(&qp_mutexes[i], NULL) != 0) {
                fprintf(stderr, "Failed to initialize mutex for QP %d\n", i);
                ret = 1;
                goto cleanup;
            }
        }
        qp_mutexes_initialized = num_qps;
        printf("Initialized %d QP mutexes for serialized RDMA access\n", num_qps);
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
    pthread_barrier_t start_barrier, end_barrier, iteration_barrier;
    int barriers_initialized = 0;
    int iteration_barrier_initialized = 0;
    /* Barriers needed for client, and also for server in allow-nvlink mode (for timing) */
    int needs_barriers = !config.is_server || (!config.direct_mode && !config.nics_only && num_source_gpus > 0 && config.reassembly);
    if (needs_barriers) {
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
        /* Iteration barrier for synchronizing between iterations */
        if (pthread_barrier_init(&iteration_barrier, NULL, num_threads + 1) != 0) {
            fprintf(stderr, "Failed to initialize iteration barrier\n");
            pthread_barrier_destroy(&start_barrier);
            pthread_barrier_destroy(&end_barrier);
            ret = 1;
            goto cleanup;
        }
        barriers_initialized = 1;
        iteration_barrier_initialized = 1;
    }
    
    /* Setup thread arguments */
    if (!config.direct_mode && num_source_gpus > 0) {
        if (config.is_server && config.reassembly) {
            /* Receiver: N threads (one per QP) */
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                int tgt_gpu = target_gpu_ids[qp_idx];
                
                args[qp_idx].ctx = ctx;
                args[qp_idx].qp_index = qp_idx;
                args[qp_idx].iterations = DEFAULT_ITERATIONS;
                args[qp_idx].is_server = config.is_server;
                args[qp_idx].source_gpu_id = -1;  /* Not used on receiver */
                args[qp_idx].target_gpu_id = tgt_gpu;
                args[qp_idx].source_qp_index = -1;
                args[qp_idx].num_source_gpus = num_source_gpus;
                args[qp_idx].source_gpu_index = -1;
                args[qp_idx].source_buffer = NULL;
                args[qp_idx].nvlink_ctx = NULL;
                
                args[qp_idx].num_qps = num_qps;
                args[qp_idx].start_barrier = &start_barrier;
                args[qp_idx].end_barrier = &end_barrier;
                args[qp_idx].iteration_barrier = iteration_barrier_initialized ? &iteration_barrier : NULL;
                args[qp_idx].bandwidth = &bandwidths[qp_idx];
                args[qp_idx].status = &statuses[qp_idx];
                args[qp_idx].nics_only = config.nics_only;
                args[qp_idx].direct_mode = config.direct_mode;
                args[qp_idx].reassembly = config.reassembly;
                args[qp_idx].reassembly_buffers = reassembly_buffers;
                args[qp_idx].reassembly_nvlink_ctxs = reassembly_nvlink_ctxs;
                args[qp_idx].reassembly_barrier = reassembly_barrier_initialized ? &reassembly_barrier : NULL;
                args[qp_idx].qp_mutex = NULL;  /* Receiver doesn't need mutex */
                statuses[qp_idx] = -1;
            }
        } else if (!config.is_server) {
            /* Sender: M×N threads (one per source GPU × QP combination) */
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
                    args[thread_idx].start_barrier = &start_barrier;
                    args[thread_idx].end_barrier = &end_barrier;
                    args[thread_idx].iteration_barrier = iteration_barrier_initialized ? &iteration_barrier : NULL;
                    args[thread_idx].bandwidth = &bandwidths[thread_idx];
                    args[thread_idx].status = &statuses[thread_idx];
                    args[thread_idx].nics_only = config.nics_only;
                    args[thread_idx].direct_mode = config.direct_mode;
                    args[thread_idx].reassembly = config.reassembly;
                    args[thread_idx].reassembly_buffers = NULL;  /* Not used on sender */
                    args[thread_idx].reassembly_nvlink_ctxs = NULL;
                    args[thread_idx].reassembly_barrier = NULL;
                    args[thread_idx].qp_mutex = (qp_mutexes && qp_idx < qp_mutexes_initialized) ? &qp_mutexes[qp_idx] : NULL;
                    statuses[thread_idx] = -1;
                }
            }
        } else {
            /* Receiver without reassembly: N threads (basic receive, no reassembly) */
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                int tgt_gpu = target_gpu_ids[qp_idx];
                
                args[qp_idx].ctx = ctx;
                args[qp_idx].qp_index = qp_idx;
                args[qp_idx].iterations = DEFAULT_ITERATIONS;
                args[qp_idx].is_server = config.is_server;
                args[qp_idx].source_gpu_id = -1;
                args[qp_idx].target_gpu_id = tgt_gpu;
                args[qp_idx].source_qp_index = -1;
                args[qp_idx].num_source_gpus = num_source_gpus;
                args[qp_idx].source_gpu_index = -1;
                args[qp_idx].source_buffer = NULL;
                args[qp_idx].nvlink_ctx = NULL;
                
                args[qp_idx].num_qps = num_qps;
                args[qp_idx].start_barrier = NULL;
                args[qp_idx].end_barrier = NULL;
                args[qp_idx].iteration_barrier = NULL;
                args[qp_idx].bandwidth = &bandwidths[qp_idx];
                args[qp_idx].status = &statuses[qp_idx];
                args[qp_idx].nics_only = config.nics_only;
                args[qp_idx].direct_mode = config.direct_mode;
                args[qp_idx].reassembly = config.reassembly;
                args[qp_idx].reassembly_buffers = NULL;
                args[qp_idx].reassembly_nvlink_ctxs = NULL;
                args[qp_idx].reassembly_barrier = NULL;
                args[qp_idx].qp_mutex = NULL;
                statuses[qp_idx] = -1;
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
            args[i].iteration_barrier = (config.is_server || !iteration_barrier_initialized) ? NULL : &iteration_barrier;
            args[i].bandwidth = &bandwidths[i];
            args[i].status = &statuses[i];
            args[i].nics_only = config.nics_only;
            args[i].direct_mode = config.direct_mode;
            args[i].reassembly = config.reassembly;
            args[i].reassembly_buffers = NULL;  /* Not used in direct/nics-only */
            args[i].reassembly_nvlink_ctxs = NULL;
            args[i].reassembly_barrier = NULL;
            args[i].qp_mutex = NULL;  /* No mutex needed for direct/single-source */
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
    /* Use receive_thread for server in allow-nvlink mode with reassembly, write_thread otherwise */
    void *(*thread_func)(void *) = write_thread;
    if (config.is_server && !config.direct_mode && !config.nics_only && config.reassembly) {
        thread_func = receive_thread_with_reassembly;
    }
    
    for (i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, thread_func, &args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            ret = 1;
            goto cleanup;
        }
    }
    
    /* Post receive WQEs for signal-back on sender (one per iteration) */
    if (!config.is_server && !config.direct_mode && !config.nics_only && config.reassembly && num_source_gpus > 0) {
        for (i = 0; i < DEFAULT_ITERATIONS; i++) {
            if (rdma_post_receive(ctx, 0, 0, 1) != 0) {
                fprintf(stderr, "Failed to post receive for signal-back iteration %d\n", i);
                ret = 1;
                goto cleanup;
            }
        }
        printf("Sender: posted %d receive WQEs for signal-back from receiver\n", DEFAULT_ITERATIONS);
    }
    
    /* Wait for all threads to reach start barrier, then take start time */
    if (needs_barriers) {
        pthread_barrier_wait(&start_barrier);
        total_start_cycles = get_cycles();
    }
    
    /* Loop over iterations: handle iteration barriers and signal-back per iteration */
    cycles_t iter_start_cycles = total_start_cycles;
    for (i = 0; i < DEFAULT_ITERATIONS; i++) {
        /* Allow-nvlink mode with reassembly: signal-back per iteration */
        if (!config.direct_mode && !config.nics_only && config.reassembly && num_source_gpus > 0) {
            if (config.is_server) {
                /* Receiver: wait for all reassembly threads, then send signal back, then iteration barrier */
                pthread_barrier_wait(&reassembly_barrier);
                
                if (rdma_write_with_imm(ctx, 0, 0, 1, 0, 0xDEADBEEF) != 0) {
                    fprintf(stderr, "Failed to send signal back for iteration %d\n", i);
                    ret = 1;
                    goto cleanup;
                }
                if (rdma_poll_completion(ctx, 0, -1) != 0) {
                    fprintf(stderr, "Failed to poll signal send completion for iteration %d\n", i);
                    ret = 1;
                    goto cleanup;
                }
                
                /* Wait at iteration barrier after sending signal */
                if (needs_barriers) {
                    pthread_barrier_wait(&iteration_barrier);
                }
            } else {
                /* Sender: wait for workers first, then wait for signal-back from receiver */
                if (needs_barriers) {
                    pthread_barrier_wait(&iteration_barrier);
                }
                
                uint32_t dummy_imm = 0;
                if (rdma_poll_completion_with_imm(ctx, 0, -1, &dummy_imm) != 0) {
                    fprintf(stderr, "Failed to receive signal-back for iteration %d\n", i);
                    ret = 1;
                    goto cleanup;
                }
            }
        } else {
            /* Non-reassembly modes: just sync at iteration barrier */
            if (needs_barriers) {
                pthread_barrier_wait(&iteration_barrier);
            }
        }
        
        
        /* Calculate and print per-iteration bandwidth (sender only) */
        if (!config.is_server && needs_barriers) {
            cycles_t iter_end_cycles = get_cycles();
            double iter_time = (double)(iter_end_cycles - iter_start_cycles) / (cpu_mhz * 1e6);
            
            /* Calculate total data transferred per iteration */
            size_t total_data_per_iter;
            if (config.direct_mode) {
                total_data_per_iter = total_buffer_size * num_threads;
            } else if (!config.nics_only && num_source_gpus > 0) {
                total_data_per_iter = total_buffer_size * num_qps;
            } else {
                total_data_per_iter = total_buffer_size;
            }
            
            double iter_bw = total_data_per_iter / (iter_time * 1e9);  /* GB/s - aggregate bandwidth */
            printf("Iteration %d: %.2f GB/s (%.6f seconds)\n", i+1, iter_bw, iter_time);
            
            /* Update start time for next iteration */
            iter_start_cycles = iter_end_cycles;
        }
    }
    
    /* Wait for all threads to reach end barrier after all iterations */
    if (needs_barriers) {
        pthread_barrier_wait(&end_barrier);
    }
    
    /* End timer after signal-back (or immediately if not allow-nvlink) */
    if (needs_barriers) {
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
    
    /* Calculate total bandwidth based on wall-clock time (SENDER ONLY) */
    double total_bw = 0.0;
    int success_count = 0;
    for (i = 0; i < num_threads; i++) {
        if (statuses[i] == 0) {
            success_count++;
        }
    }
    if (!config.is_server && success_count > 0 && total_time > 0.0) {
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
        double avg_iter_time = total_time / DEFAULT_ITERATIONS;
        double avg_iter_bw = total_data / (avg_iter_time * 1e9);
        
        /* Print summary */
        printf("Average per iteration: %.2f GB/s (%.6f seconds)\n", avg_iter_bw, avg_iter_time);
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
    
    /* Verify reassembly buffers on receiver */
    if (config.is_server && config.reassembly && reassembly_buffers && num_source_gpus > 0) {
        printf("\n=== Verifying reassembled data on receiver ===\n");
        size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;  /* 128MB per slice */
        
        for (i = 0; i < num_source_gpus; i++) {
            int gpu = source_gpu_ids_list[i];
            cudaError_t err = cudaSetDevice(gpu);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", gpu, cudaGetErrorString(err));
                continue;
            }
            
            printf("Reassembly buffer[%d] (GPU %d) slices:\n", i, gpu);
            
            /* Check each slice (one from each QP) */
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                char host_buf[256] = {0};
                size_t offset = qp_idx * slice_size;
                void *slice_ptr = (char *)reassembly_buffers[i] + offset;
                
                err = cudaMemcpy(host_buf, slice_ptr, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    fprintf(stderr, "  Failed to read slice %d: %s\n", qp_idx, cudaGetErrorString(err));
                    continue;
                }
                
                host_buf[sizeof(host_buf) - 1] = '\0';
                printf("  Slice %d (from QP %d): %s", qp_idx, qp_idx, host_buf);
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
    
    /* Free reassembly buffers if allocated */
    if (reassembly_buffers) {
        for (i = 0; i < num_source_gpus; i++) {
            if (reassembly_buffers[i]) {
                cudaSetDevice(source_gpu_ids_list[i]);
                cudaFree(reassembly_buffers[i]);
            }
        }
        free(reassembly_buffers);
    }
    
    /* Free reassembly NVLink contexts if allocated */
    if (reassembly_nvlink_ctxs) {
        for (i = 0; i < num_qps * num_source_gpus; i++) {
            if (reassembly_nvlink_ctxs[i]) {
                free(reassembly_nvlink_ctxs[i]);
            }
        }
        free(reassembly_nvlink_ctxs);
    }
    
    /* Destroy reassembly barrier */
    if (reassembly_barrier_initialized) {
        pthread_barrier_destroy(&reassembly_barrier);
    }
    
    /* Destroy QP mutexes */
    if (qp_mutexes) {
        for (i = 0; i < qp_mutexes_initialized; i++) {
            pthread_mutex_destroy(&qp_mutexes[i]);
        }
        free(qp_mutexes);
    }
    
    /* Minimal cleanup for error paths - skip problematic cleanup calls */
    if (barriers_initialized) {
        pthread_barrier_destroy(&start_barrier);
        pthread_barrier_destroy(&end_barrier);
    }
    if (iteration_barrier_initialized) {
        pthread_barrier_destroy(&iteration_barrier);
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

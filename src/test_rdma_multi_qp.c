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
#include <cuda_runtime.h>

/* Debug: print which device a pointer belongs to (helps diagnose invalid argument on copy) */
static void debug_print_ptr_device(const char *label, void *ptr, int expected_device)
{
    struct cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, ptr);
    if (e != cudaSuccess) {
        fprintf(stderr, "  %s: cudaPointerGetAttributes failed: %s\n", label, cudaGetErrorString(e));
        return;
    }
    if (attr.type == cudaMemoryTypeDevice)
        fprintf(stderr, "  %s: on device %d (expected %d) %s\n", label, attr.device, expected_device,
                attr.device != expected_device ? " <- MISMATCH" : "");
    else if (attr.type == cudaMemoryTypeHost)
        fprintf(stderr, "  %s: host memory (expected device %d) <- WRONG\n", label, expected_device);
    else
        fprintf(stderr, "  %s: type=%d device=%d (expected %d)\n", label, (int)attr.type, attr.device, expected_device);
}
#endif

#define DEFAULT_BUFFER_SIZE (1024 * 1024 * 1024)  /* 1 GB */
// #define DEFAULT_BUFFER_SIZE (1024 * 1024 *128 )  /* 128 MB */
#define DEFAULT_ITERATIONS 50
#define WARMUP_ITERATIONS 5
#define MAX_THREADS 513

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
    pthread_barrier_t *completion_barrier;  /* Barrier after completing iteration work */
    pthread_barrier_t *iteration_barrier;  /* Barrier before starting next iteration */
    double *bandwidth;  /* Output: bandwidth in GB/s */
    int *status;        /* Output: 0 on success */
    bool nics_only;     /* Whether to only use NICs for buffer allocation */
    bool direct_mode;   /* Whether in direct mode (each GPU sends via own NIC) */
    bool reassembly;    /* Whether reassembly mode is enabled (allow-nvlink only) */
    bool all_to_all;    /* All-to-all: each source sends to every target (N*M QPs) */
    int num_target_nics;   /* Number of target/receiving NICs (M); used when all_to_all */
    size_t source_buffer_size;  /* Full size of one source GPU buffer (all-to-all direct); part_size = this / num_target_nics */
    int source_qp_index;   /* QP index that has the source GPU buffer */
    void **reassembly_buffers;  /* Array of reassembly buffers (receiver only, allow-nvlink) */
    struct nvlink_context **reassembly_nvlink_ctxs;  /* NVLink contexts for reassembly */
    pthread_barrier_t *reassembly_barrier;  /* Barrier after reassembly complete */
    pthread_mutex_t *qp_mutex;  /* Mutex for serializing QP access (multi-source mode) */
    /* All-to-all allow-nvlink reassembly: N×M×M threads, sub-slice (i,j) of source n */
    bool all_to_all_reassembly;
    int slice_index_j;         /* Sender: which slice j (0..M-1); sub-slice i = qp_index */
    void **reassembly_buffers_2d;  /* Receiver: [n*M+j] = buffer on GPU j for slice j of source n */
    size_t sub_slice_size;     /* 1GB/M² */
};

/* Environment for setting up thread args (avoids passing many parameters) */
struct thread_setup_env {
    rdma_multi_qp_context_t *ctx;
    const struct rdma_multi_qp_config *config;
    int num_threads;
    int num_qps;
    int num_source_gpus;
    int num_nics;
    int *source_gpu_ids_list;
    int *target_gpu_ids;
    int *gpu_ids;
    int *expanded_gpu_ids;
    void **source_buffers;
    struct nvlink_context *nvlink_ctxs;
    pthread_barrier_t *start_barrier;
    pthread_barrier_t *end_barrier;
    pthread_barrier_t *completion_barrier;
    pthread_barrier_t *iteration_barrier;
    int completion_barrier_initialized;
    int iteration_barrier_initialized;
    void **reassembly_buffers;
    struct nvlink_context **reassembly_nvlink_ctxs;
    pthread_barrier_t *reassembly_barrier;
    int reassembly_barrier_initialized;
    void **reassembly_buffers_2d;
    pthread_mutex_t *qp_mutexes;
    int qp_mutexes_initialized;
    size_t full_source_buffer_size_all_to_all;
    double *bandwidths;
    int *statuses;
};

/* Set fields common to all thread args for this run */
static void set_thread_arg_base(struct thread_args *a, const struct thread_setup_env *env, int thread_idx)
{
    a->ctx = env->ctx;
    a->iterations = DEFAULT_ITERATIONS;
    a->is_server = env->config->is_server;
    a->num_qps = env->num_qps;
    a->start_barrier = env->start_barrier;
    a->end_barrier = env->end_barrier;
    a->completion_barrier = env->completion_barrier_initialized ? env->completion_barrier : NULL;
    a->iteration_barrier = env->iteration_barrier_initialized ? env->iteration_barrier : NULL;
    a->bandwidth = &env->bandwidths[thread_idx];
    a->status = &env->statuses[thread_idx];
    a->nics_only = env->config->nics_only;
    a->direct_mode = env->config->direct_mode;
    a->reassembly = env->config->reassembly;
}

/* Fill thread args and statuses according to run mode. Caller allocates args, statuses, bandwidths. */
static void setup_thread_args(struct thread_args *args, int *statuses, double *bandwidths,
                              const struct thread_setup_env *env)
{
    const struct rdma_multi_qp_config *config = env->config;
    int num_qps = env->num_qps;
    int num_source_gpus = env->num_source_gpus;
    int *source_gpu_ids_list = env->source_gpu_ids_list;
    int *target_gpu_ids = env->target_gpu_ids;

    /* NVLink multi-source / reassembly modes */
    if (!config->direct_mode && num_source_gpus > 0) {
        if (config->is_server && config->reassembly) {
            /* Receiver: M threads (one per QP) */
            size_t sub_slice_sz = config->all_to_all ? (DEFAULT_BUFFER_SIZE / (num_qps * num_qps)) : 0;
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                struct thread_args *a = &args[qp_idx];
                set_thread_arg_base(a, env, qp_idx);
                a->qp_index = qp_idx;
                a->source_gpu_id = -1;
                a->target_gpu_id = target_gpu_ids[qp_idx];
                a->source_qp_index = -1;
                a->num_source_gpus = num_source_gpus;
                a->source_gpu_index = -1;
                a->source_buffer = NULL;
                a->nvlink_ctx = NULL;
                a->reassembly_buffers = config->all_to_all ? NULL : env->reassembly_buffers;
                a->reassembly_nvlink_ctxs = env->reassembly_nvlink_ctxs;
                a->reassembly_barrier = env->reassembly_barrier_initialized ? env->reassembly_barrier : NULL;
                a->qp_mutex = NULL;
                a->all_to_all_reassembly = config->all_to_all;
                a->slice_index_j = 0;
                a->reassembly_buffers_2d = config->all_to_all ? env->reassembly_buffers_2d : NULL;
                a->sub_slice_size = sub_slice_sz;
                statuses[qp_idx] = -1;
            }
            return;
        }
        if (!config->is_server) {
            if (config->all_to_all && config->reassembly) {
                /* Sender: N×M×M threads */
                size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);
                for (int n = 0; n < num_source_gpus; n++) {
                    int src_gpu = source_gpu_ids_list[n];
                    for (int i = 0; i < num_qps; i++) {
                        int tgt_gpu = target_gpu_ids[i];
                        for (int j = 0; j < num_qps; j++) {
                            int thread_idx = n * num_qps * num_qps + i * num_qps + j;
                            struct thread_args *a = &args[thread_idx];
                            set_thread_arg_base(a, env, thread_idx);
                            a->qp_index = i;
                            a->source_gpu_id = src_gpu;
                            a->target_gpu_id = tgt_gpu;
                            a->source_qp_index = n;
                            a->num_source_gpus = num_source_gpus;
                            a->source_gpu_index = n;
                            a->source_buffer = env->source_buffers ? env->source_buffers[n] : NULL;
                            a->nvlink_ctx = (src_gpu >= 0 && tgt_gpu >= 0 && src_gpu != tgt_gpu)
                                             ? &env->nvlink_ctxs[thread_idx] : NULL;
                            a->reassembly_buffers = NULL;
                            a->reassembly_nvlink_ctxs = NULL;
                            a->reassembly_barrier = NULL;
                            a->qp_mutex = (env->qp_mutexes && i < env->qp_mutexes_initialized) ? &env->qp_mutexes[i] : NULL;
                            a->all_to_all_reassembly = true;
                            a->slice_index_j = j;
                            a->reassembly_buffers_2d = NULL;
                            a->sub_slice_size = sub_slice_sz;
                            statuses[thread_idx] = -1;
                        }
                    }
                }
                return;
            }
            /* Sender: M×N threads (one per source GPU × QP) */
            for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
                int src_gpu = source_gpu_ids_list[src_idx];
                int src_qp_idx = src_idx;
                for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                    int thread_idx = src_idx * num_qps + qp_idx;
                    int tgt_gpu = target_gpu_ids[qp_idx];
                    struct thread_args *a = &args[thread_idx];
                    set_thread_arg_base(a, env, thread_idx);
                    a->qp_index = qp_idx;
                    a->source_gpu_id = src_gpu;
                    a->target_gpu_id = tgt_gpu;
                    a->source_qp_index = src_qp_idx;
                    a->num_source_gpus = num_source_gpus;
                    a->source_gpu_index = src_idx;
                    a->source_buffer = env->source_buffers ? env->source_buffers[src_idx] : NULL;
                    a->nvlink_ctx = (src_gpu >= 0 && tgt_gpu >= 0 && src_gpu != tgt_gpu)
                                     ? &env->nvlink_ctxs[thread_idx] : NULL;
                    a->reassembly_buffers = NULL;
                    a->reassembly_nvlink_ctxs = NULL;
                    a->reassembly_barrier = NULL;
                    a->qp_mutex = (env->qp_mutexes && qp_idx < env->qp_mutexes_initialized) ? &env->qp_mutexes[qp_idx] : NULL;
                    a->all_to_all_reassembly = false;
                    statuses[thread_idx] = -1;
                }
            }
            return;
        }
        /* Receiver without reassembly: N threads (no barriers) */
        for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
            struct thread_args *a = &args[qp_idx];
            set_thread_arg_base(a, env, qp_idx);
            a->qp_index = qp_idx;
            a->source_gpu_id = -1;
            a->target_gpu_id = target_gpu_ids[qp_idx];
            a->source_qp_index = -1;
            a->num_source_gpus = num_source_gpus;
            a->source_gpu_index = -1;
            a->source_buffer = NULL;
            a->nvlink_ctx = NULL;
            a->start_barrier = NULL;
            a->end_barrier = NULL;
            a->completion_barrier = NULL;
            a->iteration_barrier = NULL;
            a->reassembly_buffers = NULL;
            a->reassembly_nvlink_ctxs = NULL;
            a->reassembly_barrier = NULL;
            a->qp_mutex = NULL;
            a->all_to_all_reassembly = false;
            statuses[qp_idx] = -1;
        }
        return;
    }

    /* Direct or NICs-only: one thread per QP (includes all-to-all direct) */
    int num_threads = env->num_threads;
    int *expanded_gpu_ids = env->expanded_gpu_ids;
    int *gpu_ids = env->gpu_ids;
    if (config->all_to_all && num_source_gpus > 0) {
        printf("\n=== Setting thread args for all-to-all direct (%d threads, M=%d target NICs) ===\n", num_threads, env->num_nics);
    }
    for (int i = 0; i < num_threads; i++) {
        int src_gpu = expanded_gpu_ids ? expanded_gpu_ids[i] : gpu_ids[i];
        int tgt_gpu = expanded_gpu_ids ? expanded_gpu_ids[i] : (target_gpu_ids ? target_gpu_ids[i] : gpu_ids[i]);
        struct thread_args *a = &args[i];
        set_thread_arg_base(a, env, i);
        a->qp_index = i;
        a->source_gpu_id = src_gpu;
        a->target_gpu_id = tgt_gpu;
        a->source_qp_index = i;
        a->num_source_gpus = (num_source_gpus > 0) ? 1 : 0;
        a->source_gpu_index = 0;
        a->source_buffer = NULL;
        a->nvlink_ctx = NULL;
        a->all_to_all = config->all_to_all;
        a->num_target_nics = config->all_to_all ? env->num_nics : 0;
        a->source_buffer_size = (config->all_to_all && num_source_gpus > 0) ? env->full_source_buffer_size_all_to_all : 0;
        a->start_barrier = config->is_server ? NULL : env->start_barrier;
        a->end_barrier = config->is_server ? NULL : env->end_barrier;
        a->completion_barrier = (config->is_server || !env->completion_barrier_initialized) ? NULL : env->completion_barrier;
        a->iteration_barrier = (config->is_server || !env->iteration_barrier_initialized) ? NULL : env->iteration_barrier;
        a->reassembly_buffers = NULL;
        a->reassembly_nvlink_ctxs = NULL;
        a->reassembly_barrier = NULL;
        a->qp_mutex = NULL;
        statuses[i] = -1;
    }
    if (config->all_to_all && num_source_gpus > 0) {
        printf("=== End thread args (all-to-all direct) ===\n\n");
    }
}

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
            if (rdma_post_receive(ctx, qp_index, 0, 1, 0) != 0) {
                fprintf(stderr, "QP %d: Failed to post receive %d for iteration %d\n", qp_index, src_idx, iter);
                *args->status = -1;
                return NULL;
            }
        }
        
        /* Poll M times (once for each source GPU) and reassemble */
        for (src_idx = 0; src_idx < args->num_source_gpus; src_idx++) {
            /* Poll for completion with immediate data */
            uint32_t source_gpu_idx = 0;
        if (rdma_poll_completion_with_imm(ctx, qp_index, -1, &source_gpu_idx, NULL) != 0) {
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
        
        /* Wait at completion barrier (signal we finished work for this iteration) */
        if (args->completion_barrier) {
            pthread_barrier_wait(args->completion_barrier);
        }
        
        /* Wait at iteration barrier (wait for main to send signal-back before next iteration) */
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

/* Receiver thread for all-to-all NVLink reassembly: N×M sub-slices per QP per iteration */
static void *receive_thread_all_to_all_reassembly(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    rdma_multi_qp_context_t *ctx = args->ctx;
    int qp_index = args->qp_index;  /* QP i */
    size_t sub_slice = args->sub_slice_size;  /* 1GB/M² */
    void *qp_buffer = rdma_get_local_buffer(ctx, qp_index);
    int M = args->num_qps;
    int N = args->num_source_gpus;
    
    if (args->start_barrier) {
        pthread_barrier_wait(args->start_barrier);
    }
    
    for (int iter = 0; iter < args->iterations; iter++) {
        /* Post N×M receives at distinct offsets (no overwrite); wr_id = slot index */
        for (int nj = 0; nj < N * M; nj++) {
            uint64_t offset = (uint64_t)nj * (uint64_t)sub_slice;
            if (rdma_post_receive(ctx, qp_index, offset, (uint64_t)sub_slice, (uint64_t)nj) != 0) {
                fprintf(stderr, "QP %d: Failed to post receive %d for iteration %d\n", qp_index, nj, iter);
                *args->status = -1;
                return NULL;
            }
        }
        /* Poll N×M completions (drain CQ); order is non-deterministic so do not use (wr_id,imm) for placement */
        for (int nj = 0; nj < N * M; nj++) {
            uint32_t imm = 0;
            uint64_t wr_id = 0;
            if (rdma_poll_completion_with_imm(ctx, qp_index, -1, &imm, &wr_id) != 0) {
                fprintf(stderr, "QP %d: Failed to poll completion %d\n", qp_index, nj);
                *args->status = -1;
                return NULL;
            }
        }
        /* Copy by fixed mapping: QP i has (i,j) in slot j (verified by QP buffer dump). Copy slot j -> slice j at offset i. */
        if (args->reassembly_buffers_2d) {
            for (int n = 0; n < N; n++) {
                for (int j = 0; j < M; j++) {
                    void *dst_buffer = args->reassembly_buffers_2d[n * M + j];
                    size_t dst_offset = (size_t)qp_index * sub_slice;  /* sub-slice i=qp_index at offset qp_index */
                    void *dst_ptr = (char *)dst_buffer + dst_offset;
                    size_t slot = (size_t)(n * M + j);
                    void *src_ptr = (char *)qp_buffer + slot * sub_slice;
                    int ctx_idx = qp_index * M + j;
                    struct nvlink_context *nvlink_ctx = args->reassembly_nvlink_ctxs ? args->reassembly_nvlink_ctxs[ctx_idx] : NULL;
                    if (nvlink_ctx) {
                        if (nvlink_copy_gpu_to_gpu_async(nvlink_ctx, dst_ptr, src_ptr, sub_slice) != 0) {
                            fprintf(stderr, "QP %d: NVLink reassembly (n=%d,j=%d) failed\n", qp_index, n, j);
                            *args->status = -1;
                            return NULL;
                        }
                        if (nvlink_synchronize(nvlink_ctx) != 0) {
                            *args->status = -1;
                            return NULL;
                        }
                    } else {
                        cudaError_t err = cudaSetDevice(args->target_gpu_id);
                        if (err != cudaSuccess) {
                            *args->status = -1;
                            return NULL;
                        }
                        err = cudaMemcpy(dst_ptr, src_ptr, sub_slice, cudaMemcpyDeviceToDevice);
                        if (err != cudaSuccess) {
                            fprintf(stderr, "QP %d: Reassembly copy (n=%d,j=%d) failed: %s\n",
                                    qp_index, n, j, cudaGetErrorString(err));
                            *args->status = -1;
                            return NULL;
                        }
                    }
                }
            }
        }
        if (args->reassembly_barrier) {
            pthread_barrier_wait(args->reassembly_barrier);
        }
        if (args->completion_barrier) {
            pthread_barrier_wait(args->completion_barrier);
        }
        if (args->iteration_barrier) {
            pthread_barrier_wait(args->iteration_barrier);
        }
    }
    
    if (args->end_barrier) {
        pthread_barrier_wait(args->end_barrier);
    }
    *args->bandwidth = 0.0;
    *args->status = 0;
    return NULL;
}

/* Background thread to utilize a NIC with write+poll on an extra QP (simulates other traffic on same NIC) */
struct util_nic_thread_args {
    rdma_multi_qp_context_t *ctx;
    volatile int *stop;
    size_t write_size;
};

static void *util_nic_thread(void *arg)
{
    struct util_nic_thread_args *u = (struct util_nic_thread_args *)arg;
    while (!*u->stop) {
        if (rdma_write(u->ctx, 0, 0, u->write_size, 0) != 0)
            break;
        if (rdma_poll_completion(u->ctx, 0, 100) != 0)  /* 100 ms timeout so we can check stop */
            continue;
    }
    return NULL;
}

static void *write_thread(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    rdma_multi_qp_context_t *ctx = args->ctx;
    int qp_index = args->qp_index;
    size_t buffer_size = rdma_get_buffer_size(ctx);
    size_t part_size, offset;
    
    /* In direct mode, each GPU has its own full buffer; all-to-all: each QP sends one slice */
    if (args->direct_mode) {
        if (args->all_to_all && args->num_target_nics > 0) {
            part_size = args->source_buffer_size / (size_t)args->num_target_nics;  /* slice = source_gpu_buffer_size / num_target_nics */
            offset = 0;
        } else {
            part_size = buffer_size;
            offset = 0;
        }
    } else if (args->all_to_all_reassembly) {
        /* All-to-all NVLink reassembly: one sub-slice per thread (must be first so we use sub_slice_size when num_source_gpus==1) */
        part_size = args->sub_slice_size;  /* 1GB/M² */
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
    
    int i;
    
    /* Calculate total data transferred per iteration (across all threads) */
    size_t total_data_per_iter;
    if (args->nics_only && args->num_source_gpus == 0) {
        total_data_per_iter = buffer_size;
    } else {
        /* Direct, single-source, multi-source, all-to-all: total = buffer_size * num_qps */
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
        if (args->all_to_all_reassembly) {
            /* Sub-slice (i,j) of source n: i=qp_index, j=slice_index_j; src at slice j, sub-slice i.
             * Each (n,j) must go to a distinct slot in the QP buffer so we don't overwrite: slot = n*M+j. */
            size_t slice_size = DEFAULT_BUFFER_SIZE / args->num_qps;  /* 1GB/M */
            int j = args->slice_index_j;
            int n = args->source_gpu_index;
            int M = args->num_qps;
            src_offset = (size_t)j * slice_size + (size_t)qp_index * args->sub_slice_size;
            dst_offset = (size_t)(n * M + j) * args->sub_slice_size;
            src_part = (char *)src_buffer + src_offset;
            dst_part = (char *)qp_buffer + dst_offset;
        } else if (args->num_source_gpus > 1) {
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
        
        /* Timed transfers: NVLink + RDMA in a loop over iterations (main thread measures time) */
        for (i = 0; i < args->iterations; i++) {
            /* Step 1: Copy from source buffer to QP buffer */
            if (args->source_gpu_id != args->target_gpu_id && args->nvlink_ctx) {
                /* Cross-GPU: use NVLink async */
                if (nvlink_copy_gpu_to_gpu_async(args->nvlink_ctx, dst_part,
                                                  src_part, part_size) != 0) {
                    fprintf(stderr, "NVLink copy failed (iteration %d) qp=%d\n", i, qp_index);

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
                    fprintf(stderr, "Local GPU copy failed: %s (iteration %d) qp=%d\n", cudaGetErrorString(err), i, qp_index);

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
            if (args->reassembly) {
                /* Send data and imm in one operation so receiver gets correct (n,j) with payload */
                uint32_t imm_val = (uint32_t)args->source_gpu_index;
                if (args->all_to_all_reassembly)
                    imm_val = (uint32_t)(args->source_gpu_index * args->num_qps + args->slice_index_j);
                if (rdma_write_with_imm(ctx, qp_index, dst_offset, part_size, dst_offset, imm_val) != 0) {
                    fprintf(stderr, "RDMA write_with_imm failed (iteration %d)\n", i);
                    *args->status = -1;
                    if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                    return NULL;
                }
            } else {
                if (rdma_write(ctx, qp_index, dst_offset, part_size, dst_offset) != 0) {
                    fprintf(stderr, "RDMA write failed (iteration %d)\n", i);
                    *args->status = -1;
                    if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                    return NULL;
                }
            }
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "RDMA completion failed (iteration %d)\n", i);
                *args->status = -1;
                if (args->qp_mutex) pthread_mutex_unlock(args->qp_mutex);
                return NULL;
            }
            if (args->qp_mutex) {
                pthread_mutex_unlock(args->qp_mutex);
            }
            
            /* Wait at completion barrier (signal we finished work for this iteration) */
            if (args->completion_barrier) {
                pthread_barrier_wait(args->completion_barrier);
            }
            
            /* Wait at iteration barrier (wait for main to receive signal-back before starting next iteration) */
            if (args->iteration_barrier) {
                pthread_barrier_wait(args->iteration_barrier);
            }
        }

        *args->bandwidth = 0.0;  /* Bandwidth reported by main thread from wall-clock time */
        *args->status = 0;
        
        /* Wait at end barrier before signal-back */
        if (args->end_barrier) {
            pthread_barrier_wait(args->end_barrier);
        }
        
        return NULL;
    }
    
    /* Direct/NICs-only mode (including all-to-all direct: copy slice to QP buffer then RDMA) */
    /* All-to-all direct with external_buffers: QP buffer is already a slice of source buffer; no copy, just RDMA write */
    printf("direct mode: %d, all-to-all: %d, num-target-nics: %d\n", args->direct_mode, args->all_to_all, args->num_target_nics);
    if (args->direct_mode && args->all_to_all && args->num_target_nics > 0) {
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_write(ctx, qp_index, 0, part_size, 0) != 0) {
                fprintf(stderr, "QP %d: All-to-all direct warmup write failed\n", qp_index);
                *args->status = -1;
                return NULL;
            }
        }
        for (i = 0; i < WARMUP_ITERATIONS; i++) {
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "QP %d: All-to-all direct warmup completion failed\n", qp_index);
                *args->status = -1;
                return NULL;
            }
        }
        printf("all-to-all direct waiting to start\n");
        if (args->start_barrier) pthread_barrier_wait(args->start_barrier);
        for (i = 0; i < args->iterations; i++) {
            if (rdma_write(ctx, qp_index, 0, part_size, 0) != 0) {
                fprintf(stderr, "QP %d: Write failed\n", qp_index);
                *args->status = -1;
                return NULL;
            }
            printf("all-to-all direct wrote partsize: %zu\n", part_size);
            if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
                fprintf(stderr, "QP %d: Completion failed\n", qp_index);
                *args->status = -1;
                return NULL;
            }
            printf("all-to-all direct completed polled successfully from qp %d\n", qp_index);
            if (args->completion_barrier) pthread_barrier_wait(args->completion_barrier);
            if (args->iteration_barrier) pthread_barrier_wait(args->iteration_barrier);
        }
        *args->bandwidth = 0.0;  /* Bandwidth reported by main thread */
        *args->status = 0;
        if (args->end_barrier) pthread_barrier_wait(args->end_barrier);
        return NULL;
    }
    
    /* Warmup (non all-to-all direct) */
    for (i = 0; i < WARMUP_ITERATIONS; i++) {
        if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
            fprintf(stderr, "QP %d: Warmup write %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }

    for (i = 0; i < WARMUP_ITERATIONS; i++) {
        if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
            fprintf(stderr, "QP %d: Warmup completion %d failed\n", qp_index, i);
            *args->status = -1;
            return NULL;
        }
    }
    
    if (args->start_barrier) pthread_barrier_wait(args->start_barrier);
    
    for (i = 0; i < args->iterations; i++) {
        if (rdma_write(ctx, qp_index, offset, part_size, offset) != 0) {
            fprintf(stderr, "QP %d: Write failed\n", qp_index);
            *args->status = -1;
            return NULL;
        }
        if (rdma_poll_completion(ctx, qp_index, -1) != 0) {
            fprintf(stderr, "QP %d: Completion failed\n", qp_index);
            *args->status = -1;
            return NULL;
        }
        if (args->completion_barrier) pthread_barrier_wait(args->completion_barrier);
        if (args->iteration_barrier) pthread_barrier_wait(args->iteration_barrier);
    }

    *args->bandwidth = 0.0;  /* Bandwidth reported by main thread */
    *args->status = 0;
    if (args->end_barrier) pthread_barrier_wait(args->end_barrier);
    return NULL;
}

/**
 * Print the results on the buffers to the console
 *
 * @param ctx: Context handle
 * @param expanded_gpu_ids: Expanded GPU IDs
 * @param target_gpu_ids: Target GPU IDs
 * @param num_source_gpus: Number of source GPUs
 * @param num_qps: Number of QPs
 * @param source_buffers: Source buffers
 * @param reassembly_buffers: Reassembly buffers
 * @param reassembly_buffers_2d: Reassembly buffers 2D
 */
static void print_results(struct rdma_multi_qp_config config, rdma_multi_qp_context_t *ctx, int *expanded_gpu_ids,int *target_gpu_ids, int num_source_gpus, int *gpu_ids, int num_qps, void **source_buffers, int *source_gpu_ids_list, void **reassembly_buffers, void **reassembly_buffers_2d){

    /* Verify transferred data on QP*/
    printf("\n=== Verifying transferred data on QP buffers ===\n");
    
    if (config.direct_mode) {
        /* Direct mode: read test string from each QP buffer (use same GPU array as buffer placement) */
        int *verify_gpu_ids = expanded_gpu_ids ? expanded_gpu_ids : gpu_ids;
        for (int i = 0; i < num_qps; i++) {
            int buf_gpu = verify_gpu_ids[i];
            cudaError_t err = cudaSetDevice(buf_gpu);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", buf_gpu, cudaGetErrorString(err));
                continue;
            }
            
            void *qp_buf = rdma_get_local_buffer(ctx, i);
            char host_buf[256] = {0};
            
            err = cudaMemcpy(host_buf, qp_buf, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to read from GPU %d: %s\n", buf_gpu, cudaGetErrorString(err));
                continue;
            }
            
            host_buf[sizeof(host_buf) - 1] = '\0';
            printf("QP %d (GPU_%d) read from buffer: %s", i, buf_gpu, host_buf);
        }
    } else if (!config.is_server && config.all_to_all && config.reassembly && source_buffers && num_source_gpus > 0) {
        /* All-to-all NVLink reassembly (sender only): QP buffers overwritten per send; verify source buffers show all sub-slices */
        size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;
        size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);
        printf("Sender: all-to-all reassembly - verifying source buffers (all sub-slices):\n");
        for (int n = 0; n < num_source_gpus; n++) {
            int src_gpu = source_gpu_ids_list[n];
            cudaError_t err = cudaSetDevice(src_gpu);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", src_gpu, cudaGetErrorString(err));
                continue;
            }
            printf("  Source %d (GPU %d) :\n", n, src_gpu, num_qps - 1, num_qps - 1);
            for (int j = 0; j < num_qps; j++) {
                for (int i = 0; i < num_qps; i++) {
                    char host_buf[128] = {0};
                    size_t offset = (size_t)j * slice_size + (size_t)i * sub_slice_sz;
                    err = cudaMemcpy(host_buf, (char *)source_buffers[n] + offset, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) break;
                    host_buf[sizeof(host_buf) - 1] = '\0';
                    /* Trim newline for inline */
                    for (char *p = host_buf; *p; p++) { if (*p == '\n') { *p = '\0'; break; } }
                    printf(" %s\n", host_buf);
                }
            }
        }
    } else if (config.is_server && config.all_to_all && config.reassembly && num_source_gpus > 0) {
            /* All-to-all NVLink reassembly (receiver only): each QP buffer has N*M slots (no overwrite); print all slots */
            size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);
            int num_slots_per_qp = num_source_gpus * num_qps;
            printf("\n=== QP buffers after receive (before reassembly copy) - all %d slots per QP ===\n", num_slots_per_qp);
            for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                int tgt_gpu = target_gpu_ids[qp_idx];
                cudaError_t err = cudaSetDevice(tgt_gpu);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", tgt_gpu, cudaGetErrorString(err));
                    continue;
                }
                void *qp_buf = rdma_get_local_buffer(ctx, qp_idx);
                printf("QP %d (GPU %d) slots 0..%d:\n", qp_idx, tgt_gpu, num_slots_per_qp - 1);
                for (int slot = 0; slot < num_slots_per_qp; slot++) {
                    char host_buf[256] = {0};
                    size_t offset = (size_t)slot * sub_slice_sz;
                    err = cudaMemcpy(host_buf, (char *)qp_buf + offset, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "  slot %d: read failed: %s\n", slot, cudaGetErrorString(err));
                        continue;
                    }
                    host_buf[sizeof(host_buf) - 1] = '\0';
                    printf("  slot %d: %s", slot, host_buf);
                }
            }
    } else if (num_source_gpus > 0) {
        /* NVLink mode (non all-to-all): read all sections from each target QP buffer */
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
    
    /* Verify reassembly buffers on receiver (after copy from QP buffers: slice j lives on GPU j) */
    if (config.is_server && config.reassembly && num_source_gpus > 0) {
        if (config.all_to_all && reassembly_buffers_2d) {
            size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);
            printf("\n=== Reassembled data: slice j on GPU j (expected) ===\n");
            for (int j = 0; j < num_qps; j++) {
                int gpu_j = target_gpu_ids[j];
                cudaError_t err = cudaSetDevice(gpu_j);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", gpu_j, cudaGetErrorString(err));
                    continue;
                }
                printf("Slice j=%d on GPU %d (reassembly_buffers_2d[*,%d]):\n", j, gpu_j, j);
                for (int n = 0; n < num_source_gpus; n++) {
                    void *buf = reassembly_buffers_2d[n * num_qps + j];
                    if (!buf) continue;
                    for (int i = 0; i < num_qps; i++) {
                        char host_buf[256] = {0};
                        size_t offset = (size_t)i * sub_slice_sz;
                        err = cudaMemcpy(host_buf, (char *)buf + offset, sizeof(host_buf) - 1, cudaMemcpyDeviceToHost);
                        if (err != cudaSuccess) {
                            fprintf(stderr, "  subslice (%d,%d): read failed: %s\n", i, j, cudaGetErrorString(err));
                            continue;
                        }
                        host_buf[sizeof(host_buf) - 1] = '\0';
                        printf("  subslice (i=%d,j=%d): %s", i, j, host_buf);
                    }
                }
                printf("\n");
            }
        } else if (reassembly_buffers) {
            printf("\n=== Verifying reassembled data on receiver ===\n");
            size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;  /* 128MB per slice */
            for (int i = 0; i < num_source_gpus; i++) {
                int gpu = source_gpu_ids_list[i];
                cudaError_t err = cudaSetDevice(gpu);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", gpu, cudaGetErrorString(err));
                    continue;
                }
                printf("Reassembly buffer[%d] (GPU %d) slices:\n", i, gpu);
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
    }
    
    printf("=== Data verification complete ===\n\n");
    
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
    printf("  -R, --reassembly      Enable reassembly of data from multiple source GPUs only when using NVLink\n");
    printf("  -A, --all-to-all      Spread from each source to every target (N*M QPs); use with --direct or --allow-nvlink --reassembly\n");
    printf("  -U, --utilize-nic NIC Run a background thread that does RDMA write+poll on an extra QP on NIC (same as one of -n) to simulate NIC load\n");
    printf("  -h, --help            Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  Server: %s -s -n mlx5_0,mlx5_1 -g 0,1\n", prog_name);
    printf("  Client: %s -n mlx5_0,mlx5_1 -g 0,1 -a 192.168.1.100\n", prog_name);
    printf("  Direct: %s -n mlx5_0,mlx5_1 -g 0,1 -a 192.168.1.100 --direct\n", prog_name);
}

static size_t prepare_data_buffers(struct rdma_multi_qp_config config, int num_qps, int num_source_gpus, int num_nics)
{
    /* setting up the data buffers on source GPUs (if needed - not on one to one direct mode)*/
    size_t actual_buffer_size = config.buffer_size;
    size_t full_source_buffer_size_all_to_all = 0;  /* full buffer per source GPU  */
    if (config.direct_mode && config.all_to_all && num_source_gpus > 0) {
        full_source_buffer_size_all_to_all = config.buffer_size;  /* save before we use per-QP size */
        actual_buffer_size = config.buffer_size / num_nics; /* split the buffer into num_nics parts to send each part to another nic*/
    } else if (config.all_to_all && !config.direct_mode && config.reassembly && num_source_gpus > 0) {
        /* All-to-all NVLink reassembly: QP buffer = (slice size / num_subslices) * num_slices * num_source_gpus = num_source_gpus *slice size (N*M slots for subslices per QP) */
        printf("\nDEBUG: All-to-all direct: buffer size per QP = %zu, %d target NICs = %zu bytes per QP\n", actual_buffer_size, num_nics, actual_buffer_size);
        size_t slice_size_at = config.buffer_size / num_nics;
        actual_buffer_size = slice_size_at * (size_t)num_source_gpus;
        printf("\nDEBUG: All-to-all NVLink reassembly: buffer size per QP = %zu, %d sources × slice_size (%zu) = %zu bytes per QP (no overwrite)\n",actual_buffer_size, num_source_gpus, slice_size_at, actual_buffer_size);
    } else if (!config.nics_only && !config.direct_mode && num_source_gpus > 0) {
        /* NVlink mode: Each QP buffer = (buffer_size / num_qps) × num_source_gpus */
        actual_buffer_size = (config.buffer_size * num_source_gpus) / num_qps;
        printf("\nDEBUG: Multi-source NVLink: buffer size per QP = %zu, %d sources × %zu buffer / %d QPs = %zu bytes per QP\n",
            actual_buffer_size,num_source_gpus, config.buffer_size, num_qps, actual_buffer_size);
    }
    return actual_buffer_size;
}

int main(int argc, char *argv[])
{
    struct rdma_multi_qp_config config = {0};
    rdma_multi_qp_context_t *ctx = NULL;
    rdma_multi_qp_context_t *util_ctx = NULL;  /* Extra QP on utilize_nic for background load */
    pthread_t util_thread;
    volatile int util_stop = 0;
    struct util_nic_thread_args util_args = {0};
    int util_thread_started = 0;
    pthread_t *threads = NULL;
    struct thread_args *args = NULL;
    double *bandwidths = NULL;
    int *statuses = NULL;
    char **nic_names = NULL;
    int *gpu_ids = NULL;
    char **expanded_nic_names = NULL;  /* N*M for all-to-all; points into nic_names */
    int *expanded_gpu_ids = NULL;     /* N*M for all-to-all */
    void **external_buffers = NULL;   /* Per-QP pointers into source buffers (all-to-all direct sender); no copy */
    struct nvlink_context *nvlink_ctxs = NULL;
    void **source_buffers = NULL;  /* Separate buffers on source GPUs (full size) */
    int num_qps = 0;
    int opt;
    int ret = 0;
    int i;
    char *nics_str = NULL;
    char *gpus_str = NULL;
    char *source_gpus_filter = NULL; /* filter the GPU to use as data source */
    const char *utilize_nic = NULL;  /* If set, run util thread on this NIC (must be in -n list) */
    
    /* Defaults */
    config.base_port = 18515;
    config.buffer_size = DEFAULT_BUFFER_SIZE;
    config.is_server = 0;
    config.server_addr = NULL;
    config.nics_only = 0;
    config.direct_mode = 0;
    config.reassembly = 0;
    config.all_to_all = 0;
    
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
        {"all-to-all", no_argument, 0, 'A'},
        {"utilize-nic", required_argument, 0, 'U'},
        {0, 0, 0, 0}
    };
    
    /* parsing arguments*/
    while ((opt = getopt_long(argc, argv, "n:sa:p:b:i:g:S:h:DAU:", long_options, NULL)) != -1) {
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
        case 'A':
            config.all_to_all = 1;
            break;
        case 'U':
            utilize_nic = optarg;
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
    
    if (config.all_to_all) {
        if (!config.direct_mode && !(config.reassembly && !config.nics_only)) {
            fprintf(stderr, "Error: --all-to-all requires --direct or --allow-nvlink --reassembly\n");
            return 1;
        }
        if (!source_gpus_filter) {
            fprintf(stderr, "Error: --all-to-all requires --source-gpus\n");
            return 1;
        }
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
    if (utilize_nic) {
        int found = 0;
        for (i = 0; i < num_qps; i++) {
            if (strcmp(nic_names[i], utilize_nic) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Error: --utilize-nic %s is not in the NIC list (-n)\n", utilize_nic);
            for (i = 0; i < num_qps; i++) free(nic_names[i]);
            free(nic_names);
            return 1;
        }
        printf("Utilize-NIC: will run background write+poll on NIC %s\n", utilize_nic);
    }
    int num_nics = num_qps;  /* M = number of NICs (target NICs for all-to-all) */
    
    /* Parse GPU IDs */
    gpu_ids = calloc(num_qps, sizeof(int));
    if (!gpu_ids) {
        fprintf(stderr, "Failed to allocate memory\n");
        for (i = 0; i < num_qps; i++) free(nic_names[i]);
        free(nic_names);
        return 1;
    }
    
    int *target_gpu_ids = NULL;  /* Target GPU IDs (from --gpus) curreently all the GPUs are targets in non direct mode */
    int *source_gpu_ids_list = NULL;  /* Source GPUs (from --source-gpus) currently source gpus are also target gpu in reassembly mode */
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
                if (config.all_to_all) {
                    /* All-to-all direct: N source GPUs × M target NICs -> N*M QPs */
                    int N = num_source_gpus;
                    int M = num_nics;
                    int total_qps = N * M;
                    if (config.is_server) {
                        printf("\n=== Receiver: all-to-all direct - expanding to N*M QPs ===\n");
                    } else {
                        printf("\n=== Sender: all-to-all direct - expanding to N*M QPs ===\n");
                    }
                    printf("  N = %d source GPUs, M = %d target NICs -> %d QPs total\n", N, M, total_qps);
                    expanded_nic_names = calloc(total_qps, sizeof(char *));
                    expanded_gpu_ids = calloc(total_qps, sizeof(int));
                    if (!expanded_nic_names || !expanded_gpu_ids) {
                        fprintf(stderr, "Failed to allocate expanded arrays for all-to-all\n");
                        goto cleanup;
                    }
                    for (i = 0; i < total_qps; i++) {
                        if (config.is_server) {
                            expanded_nic_names[i] = nic_names[i % M];
                            expanded_gpu_ids[i] = target_gpu_ids[i % M];
                            printf("QP %d: target GPU %d on NIC %s\n", i, expanded_gpu_ids[i], expanded_nic_names[i]);
                        } else {
                            expanded_nic_names[i] = nic_names[i / M];
                            expanded_gpu_ids[i] = source_gpu_ids_list[i / M];
                            printf("QP %d: target GPU %d on NIC %s\n", i, expanded_gpu_ids[i], expanded_nic_names[i]);

                        }
                    }
                    num_qps = total_qps;
                    num_threads = total_qps;
                    printf("  QP layout: sender QP i on NIC (i/%d), receiver QP i on NIC (i%%%d)\n", M, M);
                    printf("=== End all-to-all direct expansion ===\n\n");
                } else {
                    /* Direct mode (no all-to-all): Use only source GPU QPs */
                    if (num_source_gpus > num_qps) {
                        fprintf(stderr, "Error: More source GPUs (%d) than NICs (%d)\n", 
                                num_source_gpus, num_qps);
                        goto cleanup;
                    }
                    for (i = 0; i < num_source_gpus; i++) {
                        gpu_ids[i] = source_gpu_ids_list[i];
                    }
                    num_qps = num_source_gpus;
                    num_threads = num_source_gpus;
                    printf("Direct mode: %d source GPUs, %d QPs\n", num_source_gpus, num_qps);
                }
            } else if (!config.nics_only) { /* allow-nvlink mode */
                /* NVLink mode: M source GPUs × N target QPs threads (sender only) */
                /* All-to-all reassembly: N×M×M threads (one per source × sub-slice i × slice j) */
                if (config.is_server) {
                    /* Receiver: only need M QP threads (one per QP) */
                    num_threads = num_qps;
                } else {
                    if (config.all_to_all && config.reassembly) {
                        num_threads = num_source_gpus * num_qps * num_qps;  /* N×M×M */
                        printf("All-to-all NVLink reassembly: %d × %d × %d = %d sender threads\n",
                               num_source_gpus, num_qps, num_qps, num_threads);
                    } else {
                        num_threads = num_source_gpus * num_qps;
                    }
                }
                
                /* QP buffers: allocate on target GPUs (for RDMA)
                 * Separate source buffers (full size) will be allocated below
                 */
                for (i = 0; i < num_qps; i++) {
                    gpu_ids[i] = target_gpu_ids[i];
                    printf("QP %d buffer on target GPU %d (for RDMA)\n", i, gpu_ids[i]);
                }
                
                if (!(config.all_to_all && config.reassembly))
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

    /* finished parsing arguments*/

    /* setting up the data buffers on source GPUs (if needed - not on one to one direct mode)*/
    // size_t actual_buffer_size = prepare_data_buffers(config, num_qps, num_source_gpus, num_nics);
    size_t actual_buffer_size = config.buffer_size;
    size_t full_source_buffer_size_all_to_all = 0;  /* full buffer per source GPU  */
    if (config.direct_mode && config.all_to_all && num_source_gpus > 0) {
        full_source_buffer_size_all_to_all = config.buffer_size;  /* save before we use per-QP size */
        actual_buffer_size = config.buffer_size / num_nics; /* split the buffer into num_nics parts to send each part to another nic*/
    } else if (config.all_to_all && !config.direct_mode && config.reassembly && num_source_gpus > 0) {
        /* All-to-all NVLink reassembly: QP buffer = (slice size / num_subslices) * num_slices * num_source_gpus = num_source_gpus *slice size (N*M slots for subslices per QP) */
        printf("\nDEBUG: All-to-all direct: buffer size per QP = %zu, %d target NICs = %zu bytes per QP\n", actual_buffer_size, num_nics, actual_buffer_size);
        size_t slice_size_at = config.buffer_size / num_nics;
        actual_buffer_size = slice_size_at * (size_t)num_source_gpus;
        printf("\nDEBUG: All-to-all NVLink reassembly: buffer size per QP = %zu, %d sources × slice_size (%zu) = %zu bytes per QP (no overwrite)\n",actual_buffer_size, num_source_gpus, slice_size_at, actual_buffer_size);
    } else if (!config.nics_only && !config.direct_mode && num_source_gpus > 0) {
        /* NVlink mode: Each QP buffer = (buffer_size / num_qps) × num_source_gpus */
        actual_buffer_size = (config.buffer_size * num_source_gpus) / num_qps;
        printf("\nDEBUG: Multi-source NVLink: buffer size per QP = %zu, %d sources × %zu buffer / %d QPs = %zu bytes per QP\n",
            actual_buffer_size,num_source_gpus, config.buffer_size, num_qps, actual_buffer_size);
    }
    
    /* All-to-all direct sender: allocate source buffers and set external_buffers so each QP uses a slice (no copy) */
    if (!config.is_server && config.direct_mode && config.all_to_all && num_source_gpus > 0) {
#ifdef HAVE_CUDA
        printf("\n=== Sender: all-to-all direct - one buffer per source GPU, each slice registered per QP (no copy) ===\n");
        source_buffers = calloc(num_source_gpus, sizeof(void*));
        if (!source_buffers) {
            fprintf(stderr, "Failed to allocate source buffers array\n");
            ret = 1;
            goto cleanup;
        }
        for (i = 0; i < num_source_gpus; i++) {
            cudaError_t err = cudaSetDevice(source_gpu_ids_list[i]);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to set device %d: %s\n", source_gpu_ids_list[i], cudaGetErrorString(err));
                ret = 1;
                goto cleanup;
            }
            err = cudaMalloc(&source_buffers[i], config.buffer_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to allocate source buffer on GPU %d: %s\n", source_gpu_ids_list[i], cudaGetErrorString(err));
                ret = 1;
                goto cleanup;
            }
            printf("  Source buffer on GPU %d: %zu bytes\n", source_gpu_ids_list[i], (size_t)config.buffer_size);
        }
        external_buffers = calloc(num_qps, sizeof(void*));
        if (!external_buffers) {
            fprintf(stderr, "Failed to allocate external_buffers array\n");
            ret = 1;
            goto cleanup;
        }
        for (i = 0; i < num_qps; i++) {
            external_buffers[i] = (char *)source_buffers[i / num_nics] + (size_t)(i % num_nics) * actual_buffer_size;
        }
        config.external_buffers = external_buffers;
        printf("  %d QPs use slices of source buffers (no copy)\n", num_qps);
        printf("=== End all-to-all direct buffer setup ===\n\n");
#else
        fprintf(stderr, "Error: All-to-all direct (no-copy) requires CUDA for GPU source buffers\n");
        ret = 1;
        goto cleanup;
#endif
    }
    
    /* Setup config */
    config.nic_names = (const char **)(expanded_nic_names ? expanded_nic_names : nic_names);
    config.gpu_id = expanded_gpu_ids ? expanded_gpu_ids : gpu_ids;
    config.num_qps = num_qps;
    config.buffer_size = actual_buffer_size;
    
    /* Initialize RDMA context */
    if (rdma_multi_qp_init(&config, &ctx, config.nics_only) != 0) {
        fprintf(stderr, "Failed to initialize RDMA context\n");
        ret = 1;
        goto cleanup;
    }
    
    /* Allocate separate source buffers for multi-source NVLink*/
    if (!config.is_server && num_source_gpus > 0 && !source_buffers &&
        (!config.direct_mode && !config.nics_only || config.direct_mode && config.all_to_all)) {
        if (config.direct_mode && config.all_to_all) {
            printf("\n=== Sender: allocating source buffers for all-to-all direct  %d buffers (one per source GPU), %zu bytes each\n", num_source_gpus, (size_t)config.buffer_size);
        }
        source_buffers = calloc(num_source_gpus, sizeof(void*));
        if (!source_buffers) {
            fprintf(stderr, "Failed to allocate source buffers array\n");
            ret = 1;
            goto cleanup;
        }
        
        for (i = 0; i < num_source_gpus; i++) {
            int src_gpu = source_gpu_ids_list[i];
            size_t source_buffer_size = (config.direct_mode && config.all_to_all) ? config.buffer_size : DEFAULT_BUFFER_SIZE;
            
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
    void **reassembly_buffers_2d = NULL;  /* All-to-all: [n*M+j] = buffer on GPU j for slice j of source n */
    struct nvlink_context **reassembly_nvlink_ctxs = NULL;
    pthread_barrier_t reassembly_barrier;
    int reassembly_barrier_initialized = 0;
    
    /* Per-QP mutexes for serializing RDMA operations in multi-source mode */
    pthread_mutex_t *qp_mutexes = NULL;
    int qp_mutexes_initialized = 0;
    
    /* Allocate reassembly buffers for receiver in allow-nvlink mode */
    if (config.reassembly) {
        if (config.is_server && !config.direct_mode && !config.nics_only && num_source_gpus > 0) {
            if (config.all_to_all) {
                /* All-to-all NVLink reassembly: slice_size * *num_source_GPU */
                size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;  /* 1GB/M per slice */
                reassembly_buffers_2d = calloc(num_source_gpus * num_qps, sizeof(void*));
                if (!reassembly_buffers_2d) {
                    fprintf(stderr, "Failed to allocate reassembly_buffers_2d\n");
                    ret = 1;
                    goto cleanup;
                }
                for (int j = 0; j < num_qps; j++) {
                    int gpu_j = target_gpu_ids[j];
                    cudaError_t err = cudaSetDevice(gpu_j);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Failed to set device %d for reassembly: %s\n", gpu_j, cudaGetErrorString(err));
                        ret = 1;
                        goto cleanup;
                    }
                    for (int n = 0; n < num_source_gpus; n++) {
                        int idx = n * num_qps + j;
                        err = cudaMalloc(&reassembly_buffers_2d[idx], slice_size);
                        if (err != cudaSuccess) {
                            fprintf(stderr, "Failed to allocate reassembly 2d [src%d][slice%d] on GPU %d: %s\n",
                                    n, j, gpu_j, cudaGetErrorString(err));
                            ret = 1;
                            goto cleanup;
                        }
                    }
                    printf("Allocated %d slice buffers on GPU %d (%zu bytes each)\n", num_source_gpus, gpu_j, (size_t)slice_size);
                }
                /* NVLink contexts: QP i (GPU i) -> GPU j, index i*M+j */
                printf("Creating NVLink contexts for all-to-all reassembly (QP GPU -> slice GPU)\n");
                reassembly_nvlink_ctxs = calloc(num_qps * num_qps, sizeof(struct nvlink_context*));
                if (!reassembly_nvlink_ctxs) {
                    fprintf(stderr, "Failed to allocate reassembly NVLink contexts\n");
                    ret = 1;
                    goto cleanup;
                }
                for (int i = 0; i < num_qps; i++) {
                    int qp_gpu = target_gpu_ids[i];
                    for (int j = 0; j < num_qps; j++) {
                        int slice_gpu = target_gpu_ids[j];
                        int ctx_idx = i * num_qps + j;
                        if (qp_gpu != slice_gpu) {
                            reassembly_nvlink_ctxs[ctx_idx] = malloc(sizeof(struct nvlink_context));
                            if (!reassembly_nvlink_ctxs[ctx_idx]) {
                                fprintf(stderr, "Failed to allocate reassembly context [%d][%d]\n", i, j);
                                ret = 1;
                                goto cleanup;
                            }
                            if (nvlink_init_context(reassembly_nvlink_ctxs[ctx_idx], qp_gpu, slice_gpu) != 0) {
                                fprintf(stderr, "Failed to init NVLink reassembly GPU%d -> GPU%d\n", qp_gpu, slice_gpu);
                                ret = 1;
                                goto cleanup;
                            }
                            printf("Reassembly NVLink QP%d->GPU%d: GPU%d -> GPU%d\n", i, j, qp_gpu, slice_gpu);
                        } else {
                            reassembly_nvlink_ctxs[ctx_idx] = NULL;
                        }
                    }
                }
            } else {
                printf("\n=== Allocating reassembly buffers on receiver ===\n");
                reassembly_buffers = calloc(num_source_gpus, sizeof(void*));
                if (!reassembly_buffers) {
                    fprintf(stderr, "Failed to allocate reassembly buffers array\n");
                    ret = 1;
                    goto cleanup;
                }
                
                size_t reassembly_buffer_size = DEFAULT_BUFFER_SIZE;  /* 1GB per source GPU */
                for (i = 0; i < num_source_gpus; i++) {
                    int reassembly_gpu = source_gpu_ids_list[i];
                    
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
                
                printf("Creating NVLink contexts for reassembly (QP -> source GPU)\n");
                reassembly_nvlink_ctxs = calloc(num_qps * num_source_gpus, sizeof(struct nvlink_context*));
                if (!reassembly_nvlink_ctxs) {
                    fprintf(stderr, "Failed to allocate reassembly NVLink contexts array\n");
                    ret = 1;
                    goto cleanup;
                }
                for (int qp_idx = 0; qp_idx < num_qps; qp_idx++) {
                    int qp_gpu = target_gpu_ids[qp_idx];
                    for (int src_idx = 0; src_idx < num_source_gpus; src_idx++) {
                        int reassembly_gpu = source_gpu_ids_list[src_idx];
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
                            reassembly_nvlink_ctxs[ctx_idx] = NULL;
                        }
                    }
                }
            }
            
            /* Initialize reassembly barrier (M QP threads + main thread on receiver) */
            if (pthread_barrier_init(&reassembly_barrier, NULL, num_qps + 1) != 0) {
                fprintf(stderr, "Failed to initialize reassembly barrier\n");
                ret = 1;
                goto cleanup;
            }
            reassembly_barrier_initialized = 1;
        }
    }
    
    /* Initialize per-QP mutexes for multi-source or all-to-all reassembly (multiple threads per QP) */
    if (!config.is_server && !config.direct_mode && !config.nics_only &&
        (num_source_gpus > 1 || (config.all_to_all && config.reassembly))) {
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
    if (config.is_server && config.all_to_all) {
        printf("\n=== Receiver: all-to-all QPs connected (%d QPs), ready for incoming writes ===\n\n", num_qps);
    }
    
    /* Optional: create and connect util QP on utilize_nic (both sides) for background NIC load */
    if (utilize_nic) {
        const char *util_nic_names[1] = { utilize_nic };
        int util_gpu_id[1] = { -1 };
        struct rdma_multi_qp_config util_config = {0};
        util_config.nic_names = util_nic_names;
        util_config.num_qps = 1;
        util_config.base_port = config.base_port + num_qps;  /* Next port so handshake doesn't clash */
        util_config.server_addr = config.server_addr;
        util_config.is_server = config.is_server;
        util_config.buffer_size = (size_t)1024 * 1024 * 1024;  /* 1 GB per write to create heavy NIC load */
        util_config.gpu_id = util_gpu_id;
        util_config.nics_only = 1;
        if (rdma_multi_qp_init(&util_config, &util_ctx, 1) != 0) {
            fprintf(stderr, "Failed to init util QP on NIC %s\n", utilize_nic);
            ret = 1;
            goto cleanup;
        }
        /* Client: give server time to finish main connect and start listening for util QP */
        if (!config.is_server) {
            sleep(2);
        }
        if (rdma_multi_qp_connect(util_ctx) != 0) {
            fprintf(stderr, "Failed to connect util QP (is the receiver also running with --utilize-nic %s?)\n", utilize_nic);
            rdma_multi_qp_cleanup(util_ctx);
            util_ctx = NULL;
            /* Continue without util QP so the benchmark can still run */
        } else {
            printf("Util QP connected on NIC %s (buffer %zu bytes)\n", utilize_nic, (size_t)util_config.buffer_size);
        }
    }
    
    /* Initialize buffers with test data (client/sender only) */
    if (!config.is_server) {
        printf("\n=== Initializing test data in buffers ===\n");
        if (config.all_to_all) {
            printf("  (all-to-all direct: %d QP buffers + %d source buffers)\n", num_qps, num_source_gpus);
        }
        
        if (config.direct_mode) {
            /* Direct mode: write test string at start of each QP buffer */
            int *buf_gpu_ids = expanded_gpu_ids ? expanded_gpu_ids : gpu_ids;
            for (i = 0; i < num_qps; i++) {
                cudaError_t err = cudaSetDevice(buf_gpu_ids[i]);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", buf_gpu_ids[i], cudaGetErrorString(err));
                    continue;
                }
                
                void *qp_buf = rdma_get_local_buffer(ctx, i);
                size_t buf_size = rdma_get_buffer_size(ctx);
                char test_str[128];
                snprintf(test_str, sizeof(test_str), "one buffer of size %zu on GPU_%d\n", buf_size, buf_gpu_ids[i]);
                
                err = cudaMemcpy(qp_buf, test_str, strlen(test_str) + 1, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to write test data to GPU %d: %s\n", buf_gpu_ids[i], cudaGetErrorString(err));
                }
                if (!config.all_to_all || i < 4 || i >= num_qps - 2) {  /* Limit print for all-to-all */
                    printf("Initialized QP %d buffer on GPU %d with test string\n", i, buf_gpu_ids[i]);
                } else if (i == 4) {
                    printf("  ... (%d more QP buffers)\n", num_qps - 6);
                }
            }
            if (config.all_to_all && source_buffers) {
                printf("\n=== Initializing source buffers for all-to-all direct ===\n");
                for (i = 0; i < num_source_gpus; i++) {
                    cudaError_t err = cudaSetDevice(source_gpu_ids_list[i]);
                    if (err != cudaSuccess) continue;
                    size_t src_size = config.buffer_size;
                    size_t slice_size = src_size / num_nics;
                    for (int m = 0; m < num_nics; m++) {
                        char test_str[128];
                        snprintf(test_str, sizeof(test_str), "source_gpu_%d slice_%d/%d\n", i, m, num_nics);
                        void *dst = (char *)source_buffers[i] + m * slice_size;
                        err = cudaMemcpy(dst, test_str, strlen(test_str) + 1, cudaMemcpyHostToDevice);
                        if (err != cudaSuccess) break;
                    }
                    printf("  Source buffer on GPU %d: %d slices\n", source_gpu_ids_list[i], num_nics);
                }
                printf("=== End source buffer init (all-to-all) ===\n");
            }
        } else if (config.all_to_all && config.reassembly && source_buffers && num_source_gpus > 0) {
            /* All-to-all NVLink reassembly: write test string into each sub-slice (i,j) */
            size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;       /* 1GB/M */
            size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);  /* 1GB/M² */
            for (int n = 0; n < num_source_gpus; n++) {
                int src_gpu = source_gpu_ids_list[n];
                cudaError_t err = cudaSetDevice(src_gpu);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to set device %d: %s\n", src_gpu, cudaGetErrorString(err));
                    continue;
                }
                for (int j = 0; j < num_qps; j++) {
                    for (int i = 0; i < num_qps; i++) {
                        char test_str[128];
                        snprintf(test_str, sizeof(test_str), "subslice %d,%d source %d gpu_%d\n", i, j, n, src_gpu);
                        size_t offset = (size_t)j * slice_size + (size_t)i * sub_slice_sz;
                        void *dst = (char *)source_buffers[n] + offset;
                        err = cudaMemcpy(dst, test_str, strlen(test_str) + 1, cudaMemcpyHostToDevice);
                        if (err != cudaSuccess) {
                            fprintf(stderr, "Failed to write sub-slice (%d,%d) on GPU %d: %s\n", i, j, src_gpu, cudaGetErrorString(err));
                        }
                    }
                }
                printf("Initialized source buffer on GPU %d with %d×%d test sub-slices\n", src_gpu, num_qps, num_qps);
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
    if (!config.is_server && !config.direct_mode && num_source_gpus > 0) {
        if (config.all_to_all && config.reassembly) {
            /* All-to-all reassembly: one NVLink context per thread (N×M×M) so no sharing / no mutex */
            printf("Creating NVLink contexts for all-to-all reassembly: %d threads (one per thread)\n", num_threads);
            for (int n = 0; n < num_source_gpus; n++) {
                int src_gpu = source_gpu_ids_list[n];
                for (int i = 0; i < num_qps; i++) {
                    int tgt_gpu = target_gpu_ids[i];
                    for (int j = 0; j < num_qps; j++) {
                        int thread_idx = n * num_qps * num_qps + i * num_qps + j;
                        if (src_gpu >= 0 && tgt_gpu >= 0 && src_gpu != tgt_gpu) {
                            if (nvlink_init_context(&nvlink_ctxs[thread_idx], src_gpu, tgt_gpu) != 0) {
                                fprintf(stderr, "Failed to init NVLink all-to-all thread %d (GPU%d->GPU%d)\n",
                                        thread_idx, src_gpu, tgt_gpu);
                                ret = 1;
                                goto cleanup;
                            }
                        }
                    }
                }
            }
            printf("NVLink all-to-all: %d contexts created (one per thread, no sharing)\n", num_threads);
        } else {
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
            int *path_gpu_ids = expanded_gpu_ids ? expanded_gpu_ids : gpu_ids;
            for (i = 0; i < num_qps; i++) {
                int src_gpu = path_gpu_ids[i];
                const char *local_nic = nic_names[expanded_nic_names ? (i / num_nics) : i];
                const char *target_nic = expanded_nic_names ? nic_names[i % num_nics] : nic_names[i];
                
                if (config.direct_mode) {
                    if (src_gpu >= 0) {
                        if (config.all_to_all && expanded_nic_names) {
                            /* All-to-all: same source GPU/NIC for QPs from same source; show target NIC */
                            printf("  Thread %d: GPU%d -> %s -> RDMA (direct) -> %s\n", i, src_gpu, local_nic, target_nic);
                        } else {
                            printf("  Thread %d: GPU%d -> %s -> RDMA (direct)\n", i, src_gpu, local_nic);
                        }
                    } else {
                        printf("  Thread %d: Host -> %s -> RDMA (direct)\n", i, local_nic);
                    }
                } else if (config.nics_only) {
                    if (src_gpu >= 0) {
                        printf("  Thread %d: GPU%d -> %s -> RDMA\n", i, src_gpu, local_nic);
                    } else {
                        printf("  Thread %d: Host -> %s -> RDMA\n", i, local_nic);
                    }
                }
            }
        }
        printf("\n");
    }
    
    /* Initialize barriers for synchronizing thread start/end */
    pthread_barrier_t start_barrier, end_barrier, completion_barrier, iteration_barrier;
    int barriers_initialized = 0;
    int completion_barrier_initialized = 0;
    int iteration_barrier_initialized = 0;
    /* Barriers needed for client, and also for server in allow-nvlink mode and reassembly mode (for timing) */
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
        /* Completion barrier: workers signal they finished work for this iteration */
        if (pthread_barrier_init(&completion_barrier, NULL, num_threads + 1) != 0) {
            fprintf(stderr, "Failed to initialize completion barrier\n");
            pthread_barrier_destroy(&start_barrier);
            pthread_barrier_destroy(&end_barrier);
            ret = 1;
            goto cleanup;
        }
        completion_barrier_initialized = 1;
        /* Iteration barrier: main signals signal-back received, workers can start next iteration */
        if (pthread_barrier_init(&iteration_barrier, NULL, num_threads + 1) != 0) {
            fprintf(stderr, "Failed to initialize iteration barrier\n");
            pthread_barrier_destroy(&start_barrier);
            pthread_barrier_destroy(&end_barrier);
            pthread_barrier_destroy(&completion_barrier);
            ret = 1;
            goto cleanup;
        }
        barriers_initialized = 1;
        iteration_barrier_initialized = 1;
    }
    
    /* Setup thread arguments */
    {
        struct thread_setup_env env = {
            .ctx = ctx,
            .config = &config,
            .num_threads = num_threads,
            .num_qps = num_qps,
            .num_source_gpus = num_source_gpus,
            .num_nics = num_nics,
            .source_gpu_ids_list = source_gpu_ids_list,
            .target_gpu_ids = target_gpu_ids,
            .gpu_ids = gpu_ids,
            .expanded_gpu_ids = expanded_gpu_ids,
            .source_buffers = source_buffers,
            .nvlink_ctxs = nvlink_ctxs,
            .start_barrier = &start_barrier,
            .end_barrier = &end_barrier,
            .completion_barrier = &completion_barrier,
            .iteration_barrier = &iteration_barrier,
            .completion_barrier_initialized = completion_barrier_initialized,
            .iteration_barrier_initialized = iteration_barrier_initialized,
            .reassembly_buffers = reassembly_buffers,
            .reassembly_nvlink_ctxs = reassembly_nvlink_ctxs,
            .reassembly_barrier = &reassembly_barrier,
            .reassembly_barrier_initialized = reassembly_barrier_initialized,
            .reassembly_buffers_2d = reassembly_buffers_2d,
            .qp_mutexes = qp_mutexes,
            .qp_mutexes_initialized = qp_mutexes_initialized,
            .full_source_buffer_size_all_to_all = full_source_buffer_size_all_to_all,
            .bandwidths = bandwidths,
            .statuses = statuses,
        };
        setup_thread_args(args, statuses, bandwidths, &env);
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
    void *(*thread_func)(void *) = write_thread;
    if (config.is_server && !config.direct_mode && !config.nics_only && config.reassembly) {
        if (config.all_to_all)
            thread_func = receive_thread_all_to_all_reassembly;
        else
            thread_func = receive_thread_with_reassembly;
    }
    
    for (i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, thread_func, &args[i]) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            ret = 1;
            goto cleanup;
        }
    }
    
    /* Start util NIC thread (client only): write+poll loop on util QP to simulate other traffic on that NIC */
    if (util_ctx && !config.is_server) {
        util_args.ctx = util_ctx;
        util_args.stop = &util_stop;
        util_args.write_size = rdma_get_buffer_size(util_ctx);
        if (util_args.write_size == 0) util_args.write_size = (size_t)1024 * 1024 * 1024;  /* 1 GB */
        util_stop = 0;
        if (pthread_create(&util_thread, NULL, util_nic_thread, &util_args) == 0) {
            util_thread_started = 1;
            printf("Util NIC thread started on %s\n", utilize_nic);
        }
    }
    
    /* Post receive WQEs for signal-back on sender (one per iteration) for allow-nvlink mode and reassembly mode */
    if (!config.is_server && !config.direct_mode && !config.nics_only && config.reassembly && num_source_gpus > 0) {
        for (i = 0; i < DEFAULT_ITERATIONS; i++) {
            if (rdma_post_receive(ctx, 0, 0, 1, 0) != 0) {
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
                /* Receiver: wait for all reassembly threads, then send signal back */
                pthread_barrier_wait(&reassembly_barrier);
                
                /* Wait for workers to finish (completion_barrier) */
                if (needs_barriers) {
                    pthread_barrier_wait(&completion_barrier);
                }
                
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
                
                /* Release workers to start next iteration */
                if (needs_barriers) {
                    pthread_barrier_wait(&iteration_barrier);
                }
            } else {
                /* Sender: wait for workers to complete, poll for signal-back, then release workers */
                if (needs_barriers) {
                    pthread_barrier_wait(&completion_barrier);
                }
                
                /* Poll for signal-back while workers wait at iteration_barrier */
                uint32_t dummy_imm = 0;
                if (rdma_poll_completion_with_imm(ctx, 0, -1, &dummy_imm, NULL) != 0) {
                    fprintf(stderr, "Failed to receive signal-back for iteration %d\n", i);
                    ret = 1;
                    goto cleanup;
                }
                
                /* Release workers to start next iteration */
                if (needs_barriers) {
                    pthread_barrier_wait(&iteration_barrier);
                }
            }
        } else {
            /* Non-reassembly modes: sync at completion barrier, then release via iteration barrier */
            if (needs_barriers) {
                pthread_barrier_wait(&completion_barrier);
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
    
    /* Stop and join util NIC thread */
    if (util_thread_started) {
        util_stop = 1;
        pthread_join(util_thread, NULL);
        util_thread_started = 0;
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
        /* Calculate total data transferred per iteration */
        size_t total_data;
        if (config.direct_mode) {
            printf("total_buffer_size: %zu and num_threads: %d\n", total_buffer_size, num_threads);
            if (config.all_to_all && num_source_gpus > 0) {
                /* All-to-all: per-QP size × M × N */
                total_data = total_buffer_size * num_nics * num_source_gpus;
            } else {
                /* Plain direct: each QP sends full buffer */
                total_data = total_buffer_size * num_qps;
            }
        } else if (config.all_to_all && config.reassembly && num_source_gpus > 0) {
            /* All-to-all NVLink reassembly: N×M×M sub-slices × sub_slice_size = N×1GB */
            total_data = (size_t)num_source_gpus * DEFAULT_BUFFER_SIZE;
        } else if (!config.nics_only && num_source_gpus > 0) {
            printf("total_buffer_size: %zu and num_qps: %d and num_source_gpus: %d\n", total_buffer_size, num_qps, num_source_gpus);
            total_data = total_buffer_size * num_qps;
        } else {
            total_data = total_buffer_size;
        }
        printf("total_data: %zu\n", total_data);
        double aggregate_bw = (double)(total_data * DEFAULT_ITERATIONS) / (total_time * 1e9);
        total_bw = aggregate_bw;
        double avg_iter_time = total_time / DEFAULT_ITERATIONS;
        double avg_iter_bw = total_data / (avg_iter_time * 1e9);
        double latency_sec = total_time / DEFAULT_ITERATIONS;  /* latency = total time / iterations */
        printf("Average per iteration: %.2f GB/s (%.6f seconds)\n", avg_iter_bw, avg_iter_time);
        printf("\033[0;32mTotal bandwidth: %.2f GB/s\033[0m\n", total_bw);
        printf("Latency (per iteration): %.6f s (%.3f ms)\n", latency_sec, latency_sec * 1000.0);
    }
    /*verify data -print QP and data buffer */
    print_results(config, ctx, expanded_gpu_ids, target_gpu_ids, num_source_gpus, gpu_ids, num_qps, source_buffers, source_gpu_ids_list, reassembly_buffers, reassembly_buffers_2d);
    
    /* Skip cleanup to avoid segfault - just return */
    // return ret;
    
cleanup:
    /* Stop util NIC thread and cleanup util context */
    if (util_thread_started) {
        util_stop = 1;
        pthread_join(util_thread, NULL);
        util_thread_started = 0;
    }
    if (util_ctx) {
        rdma_multi_qp_cleanup(util_ctx);
        util_ctx = NULL;
    }
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
    /* Free all-to-all reassembly 2D buffers (on target GPUs) */
    if (reassembly_buffers_2d) {
        for (int j = 0; j < num_qps; j++) {
            int gpu_j = target_gpu_ids ? target_gpu_ids[j] : 0;
            for (int n = 0; n < num_source_gpus; n++) {
                int idx = n * num_qps + j;
                if (reassembly_buffers_2d[idx]) {
                    cudaSetDevice(gpu_j);
                    cudaFree(reassembly_buffers_2d[idx]);
                }
            }
        }
        free(reassembly_buffers_2d);
    }
    
    /* Free reassembly NVLink contexts if allocated (count before freeing reassembly_buffers_2d) */
    if (reassembly_nvlink_ctxs) {
        int reassembly_ctx_count = reassembly_buffers_2d ? (num_qps * num_qps) : (num_qps * num_source_gpus);
        for (i = 0; i < reassembly_ctx_count; i++) {
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
    if (completion_barrier_initialized) {
        pthread_barrier_destroy(&completion_barrier);
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
    {
        int nic_count = expanded_nic_names ? num_nics : num_qps;
        if (expanded_nic_names) free(expanded_nic_names);
        if (expanded_gpu_ids) free(expanded_gpu_ids);
        if (external_buffers) free(external_buffers);
        if (nic_names) {
            for (i = 0; i < nic_count; i++) {
            if (nic_names[i]) free(nic_names[i]);
        }
            free(nic_names);
        }
    }
    if (gpu_ids) free(gpu_ids);
    if (target_gpu_ids) free(target_gpu_ids);
    if (source_gpu_ids_list) free(source_gpu_ids_list);
    
    return ret;
}

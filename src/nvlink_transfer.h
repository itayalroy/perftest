/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * NVLink Transfer Module
 * Provides functions for GPU-to-GPU transfers via NVLink using CUDA Runtime API
 */

#ifndef NVLINK_TRANSFER_H
#define NVLINK_TRANSFER_H

#include "config.h"
#include <stddef.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
#error "CUDA support is required for NVLink transfers"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* NVLink context structure */
struct nvlink_context {
	int gpu0_id;
	int gpu1_id;
	cudaStream_t stream;
	int peer_access_enabled;
};

/**
 * Check if peer access (NVLink/P2P) is available between two GPUs
 *
 * @param gpu0_id: Source GPU device ID
 * @param gpu1_id: Destination GPU device ID
 * @return: 1 if peer access is available, 0 otherwise
 */
int nvlink_check_peer_access(int gpu0_id, int gpu1_id);

/**
 * Initialize NVLink context for two GPUs
 * Enables peer access and creates CUDA stream for asynchronous transfers
 *
 * @param ctx: Pointer to nvlink_context structure (will be initialized)
 * @param gpu0_id: First GPU device ID
 * @param gpu1_id: Second GPU device ID
 * @return: 0 on success, non-zero on error
 */
int nvlink_init_context(struct nvlink_context *ctx, int gpu0_id, int gpu1_id);

/**
 * Cleanup NVLink context
 * Destroys CUDA stream and disables peer access
 *
 * @param ctx: Pointer to nvlink_context structure
 * @return: 0 on success, non-zero on error
 */
int nvlink_cleanup_context(struct nvlink_context *ctx);

/**
 * Copy data from one GPU to another via NVLink (synchronous)
 * Uses the nvlink_context which has peer access already enabled
 *
 * @param ctx: NVLink context (must be initialized)
 * @param dst: Destination buffer pointer (on ctx->gpu1_id)
 * @param src: Source buffer pointer (on ctx->gpu0_id)
 * @param size: Size in bytes to copy
 * @return: 0 on success, non-zero on error
 */
int nvlink_copy_gpu_to_gpu_sync(struct nvlink_context *ctx, void *dst, void *src, size_t size);

/**
 * Copy data from one GPU to another via NVLink (asynchronous)
 * Uses the transfer stream from the nvlink_context
 *
 * @param ctx: NVLink context (must be initialized)
 * @param dst: Destination buffer pointer (on ctx->gpu1_id)
 * @param src: Source buffer pointer (on ctx->gpu0_id)
 * @param size: Size in bytes to copy
 * @return: 0 on success, non-zero on error
 */
int nvlink_copy_gpu_to_gpu_async(struct nvlink_context *ctx, void *dst, void *src, size_t size);

/**
 * Synchronize NVLink transfer stream
 * Waits for all pending transfers to complete
 *
 * @param ctx: NVLink context
 * @return: 0 on success, non-zero on error
 */
int nvlink_synchronize(struct nvlink_context *ctx);

/**
 * Get NVLink topology information
 * Prints information about P2P access capabilities
 *
 * @param gpu0_id: First GPU device ID
 * @param gpu1_id: Second GPU device ID
 */
void nvlink_print_topology(int gpu0_id, int gpu1_id);

#ifdef __cplusplus
}
#endif

#endif /* NVLINK_TRANSFER_H */


/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * NVLink Transfer Module Implementation using CUDA Runtime API
 */

#include "config.h"
#include "nvlink_transfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
#error "CUDA support is required for NVLink transfers"
#endif

/******************************************************************************
 * Check if peer access (NVLink/P2P) is available between two GPUs
 ******************************************************************************/
int nvlink_check_peer_access(int gpu0_id, int gpu1_id)
{
	int can_access_peer = 0;
	cudaError_t err;

	if (gpu0_id == gpu1_id) {
		return 1;  /* Same GPU */
	}

	err = cudaDeviceCanAccessPeer(&can_access_peer, gpu0_id, gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceCanAccessPeer failed: %s\n", cudaGetErrorString(err));
		return 0;
	}

	return can_access_peer;
}

/******************************************************************************
 * Initialize NVLink context for two GPUs
 ******************************************************************************/
int nvlink_init_context(struct nvlink_context *ctx, int gpu0_id, int gpu1_id)
{
	cudaError_t err;
	struct cudaDeviceProp prop0, prop1;
	char name0[256], name1[256];

	if (!ctx) {
		fprintf(stderr, "nvlink_init_context: NULL context pointer\n");
		return 1;
	}

	memset(ctx, 0, sizeof(struct nvlink_context));
	ctx->gpu0_id = gpu0_id;
	ctx->gpu1_id = gpu1_id;

	/* Get device properties */
	err = cudaGetDeviceProperties(&prop0, gpu0_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_init_context: Failed to get properties for GPU %d: %s\n",
			gpu0_id, cudaGetErrorString(err));
		return 1;
	}

	err = cudaGetDeviceProperties(&prop1, gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_init_context: Failed to get properties for GPU %d: %s\n",
			gpu1_id, cudaGetErrorString(err));
		return 1;
	}

	strncpy(name0, prop0.name, sizeof(name0) - 1);
	name0[sizeof(name0) - 1] = '\0';
	strncpy(name1, prop1.name, sizeof(name1) - 1);
	name1[sizeof(name1) - 1] = '\0';

	printf("NVLink: Initializing context for GPU %d (%s) and GPU %d (%s)\n",
		gpu0_id, name0, gpu1_id, name1);

	/* Check peer access */
	if (gpu0_id != gpu1_id) {
		if (!nvlink_check_peer_access(gpu0_id, gpu1_id)) {
			fprintf(stderr, "nvlink_init_context: GPU %d cannot access GPU %d (no NVLink/P2P)\n",
				gpu0_id, gpu1_id);
			return 1;
		}

		/* Enable peer access */
		err = cudaSetDevice(gpu0_id);
		if (err != cudaSuccess) {
			fprintf(stderr, "nvlink_init_context: Failed to set device %d: %s\n",
				gpu0_id, cudaGetErrorString(err));
			return 1;
		}

		err = cudaDeviceEnablePeerAccess(gpu1_id, 0);
		if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
			fprintf(stderr, "nvlink_init_context: Failed to enable peer access: %s\n",
				cudaGetErrorString(err));
			return 1;
		}

		if (err == cudaErrorPeerAccessAlreadyEnabled) {
			printf("NVLink: Peer access already enabled\n");
		} else {
			printf("NVLink: Peer access enabled from GPU %d to GPU %d\n", gpu0_id, gpu1_id);
		}

		ctx->peer_access_enabled = 1;
	} else {
		/* Same GPU, no peer access needed */
		ctx->peer_access_enabled = 0;
	}

	/* Create CUDA stream for asynchronous transfers on destination GPU */
	err = cudaSetDevice(gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_init_context: Failed to set device %d for stream: %s\n",
			gpu1_id, cudaGetErrorString(err));
		return 1;
	}

	err = cudaStreamCreate(&ctx->stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_init_context: Failed to create stream: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	printf("NVLink: Context initialized successfully\n");
	return 0;
}

/******************************************************************************
 * Cleanup NVLink context
 ******************************************************************************/
int nvlink_cleanup_context(struct nvlink_context *ctx)
{
	cudaError_t err;

	if (!ctx) {
		return 0;
	}

	/* Synchronize devices before cleanup */
	if (ctx->stream) {
		cudaSetDevice(ctx->gpu1_id);
		cudaStreamSynchronize(ctx->stream);
	}

	cudaSetDevice(ctx->gpu0_id);
	cudaDeviceSynchronize();

	if (ctx->gpu0_id != ctx->gpu1_id) {
		cudaSetDevice(ctx->gpu1_id);
		cudaDeviceSynchronize();
	}

	/* Destroy stream */
	if (ctx->stream) {
		cudaSetDevice(ctx->gpu1_id);
		err = cudaStreamDestroy(ctx->stream);
		if (err != cudaSuccess) {
			fprintf(stderr, "nvlink_cleanup_context: Failed to destroy stream: %s\n",
				cudaGetErrorString(err));
		}
		ctx->stream = NULL;
	}

	printf("NVLink: Context cleaned up\n");
	return 0;
}

/******************************************************************************
 * Copy data from one GPU to another via NVLink (synchronous)
 ******************************************************************************/
int nvlink_copy_gpu_to_gpu_sync(struct nvlink_context *ctx, void *dst, void *src, size_t size)
{
	cudaError_t err;

	if (!ctx) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_sync: NULL context\n");
		return 1;
	}

	if (!dst || !src) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_sync: NULL pointer\n");
		return 1;
	}

	if (size == 0) {
		return 0;
	}

	/* Set device to destination GPU to enable parallel transfers */
	err = cudaSetDevice(ctx->gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_sync: Failed to set device %d: %s\n",
			ctx->gpu1_id, cudaGetErrorString(err));
		return 1;
	}

	/* Perform peer-to-peer copy */
	err = cudaMemcpyPeer(dst, ctx->gpu1_id, src, ctx->gpu0_id, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_sync: cudaMemcpyPeer failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	/* cudaMemcpyPeer can be asynchronous for peer-to-peer transfers */
	/* Explicitly synchronize to ensure transfer completes */
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_sync: cudaDeviceSynchronize failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	return 0;
}

/******************************************************************************
 * Copy data from one GPU to another via NVLink (asynchronous)
 ******************************************************************************/
int nvlink_copy_gpu_to_gpu_async(struct nvlink_context *ctx, void *dst, void *src, size_t size)
{
	cudaError_t err;

	if (!ctx) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_async: NULL context\n");
		return 1;
	}

	if (!dst || !src) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_async: NULL pointer\n");
		return 1;
	}

	if (size == 0) {
		return 0;
	}

	if (!ctx->stream) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_async: Transfer stream not initialized\n");
		return 1;
	}

	/* Set device to destination GPU to enable parallel transfers */
	err = cudaSetDevice(ctx->gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_async: Failed to set device %d: %s\n",
			ctx->gpu1_id, cudaGetErrorString(err));
		return 1;
	}

	/* Perform asynchronous peer-to-peer copy */
	err = cudaMemcpyPeerAsync(dst, ctx->gpu1_id, src, ctx->gpu0_id, size, ctx->stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_copy_gpu_to_gpu_async: cudaMemcpyPeerAsync failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	return 0;
}

/******************************************************************************
 * Synchronize NVLink transfer stream
 ******************************************************************************/
int nvlink_synchronize(struct nvlink_context *ctx)
{
	cudaError_t err;

	if (!ctx) {
		fprintf(stderr, "nvlink_synchronize: NULL context\n");
		return 1;
	}

	if (!ctx->stream) {
		fprintf(stderr, "nvlink_synchronize: Transfer stream not initialized\n");
		return 1;
	}

	err = cudaSetDevice(ctx->gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_synchronize: Failed to set device %d: %s\n",
			ctx->gpu1_id, cudaGetErrorString(err));
		return 1;
	}

	err = cudaStreamSynchronize(ctx->stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "nvlink_synchronize: cudaStreamSynchronize failed: %s\n",
			cudaGetErrorString(err));
		return 1;
	}

	return 0;
}

/******************************************************************************
 * Get NVLink topology information
 ******************************************************************************/
void nvlink_print_topology(int gpu0_id, int gpu1_id)
{
	int device_count;
	struct cudaDeviceProp prop0, prop1;
	int can_access_peer = 0;

	printf("\n========================================\n");
	printf("NVLink Topology Information:\n");
	printf("========================================\n");

	cudaGetDeviceCount(&device_count);
	if (device_count == 0) {
		fprintf(stderr, "No CUDA devices found\n");
		return;
	}

	if (gpu0_id >= device_count || gpu1_id >= device_count) {
		fprintf(stderr, "Invalid GPU IDs\n");
		return;
	}

	cudaGetDeviceProperties(&prop0, gpu0_id);
	cudaGetDeviceProperties(&prop1, gpu1_id);

	printf("GPU %d: %s\n", gpu0_id, prop0.name);
	printf("GPU %d: %s\n", gpu1_id, prop1.name);

	if (gpu0_id == gpu1_id) {
		printf("Same GPU - no peer access needed\n");
		printf("========================================\n\n");
		return;
	}

	if (nvlink_check_peer_access(gpu0_id, gpu1_id)) {
		printf("Peer access: GPU %d CAN access GPU %d\n", gpu0_id, gpu1_id);
		printf("(NVLink or P2P connection available)\n");
	} else {
		printf("Peer access: GPU %d CANNOT access GPU %d\n", gpu0_id, gpu1_id);
		printf("(No NVLink or P2P connection available)\n");
	}

	printf("========================================\n\n");
}


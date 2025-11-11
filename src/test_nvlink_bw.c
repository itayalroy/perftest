/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * Simple NVLink Bandwidth Test using CUDA Runtime API
 * Tests GPU-to-GPU transfer bandwidth via NVLink
 */

#include "config.h"
#include "nvlink_transfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>
#include <stdint.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
#error "CUDA support is required for this test"
#endif

#define DEFAULT_BUFFER_SIZE (64 * 1024 * 1024)  /* 64 MB */
#define DEFAULT_ITERATIONS 100
#define WARMUP_ITERATIONS 10

static void print_usage(const char *prog_name)
{
	printf("Usage: %s [OPTIONS]\n", prog_name);
	printf("Options:\n");
	printf("  -g, --gpu0 ID          Source GPU ID (default: 0)\n");
	printf("  -G, --gpu1 ID          Destination GPU ID (default: 1)\n");
	printf("  -s, --size SIZE        Buffer size in bytes (default: 64MB)\n");
	printf("  -i, --iterations NUM   Number of iterations (default: 100)\n");
	printf("  -a, --async            Use asynchronous transfers\n");
	printf("  -h, --help             Show this help message\n");
	printf("\n");
	printf("Examples:\n");
	printf("  %s -g 0 -G 1 -s 1048576 -i 50\n", prog_name);
	printf("  %s --gpu0 0 --gpu1 1 --size 67108864 --iterations 200 --async\n", prog_name);
}

static double get_time_usec(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000.0 + tv.tv_usec;
}


int main(int argc, char *argv[])
{
	int gpu0_id = 0;
	int gpu1_id = 1;
	size_t buffer_size = DEFAULT_BUFFER_SIZE;
	int iterations = DEFAULT_ITERATIONS;
	int use_async = 0;
	int opt;
	void *src_buffer = NULL;
	void *dst_buffer = NULL;
	struct nvlink_context nvlink_ctx;
	cudaError_t err;
	double start_time, end_time;
	double total_time = 0.0;
	double bandwidth_gbps;
	int i;
	int ret = 0;

	/* Command line parsing */
	static struct option long_options[] = {
		{"gpu0", required_argument, 0, 'g'},
		{"gpu1", required_argument, 0, 'G'},
		{"size", required_argument, 0, 's'},
		{"iterations", required_argument, 0, 'i'},
		{"async", no_argument, 0, 'a'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	while ((opt = getopt_long(argc, argv, "g:G:s:i:ah", long_options, NULL)) != -1) {
		switch (opt) {
		case 'g':
			gpu0_id = atoi(optarg);
			break;
		case 'G':
			gpu1_id = atoi(optarg);
			break;
		case 's':
			buffer_size = strtoull(optarg, NULL, 0);
			break;
		case 'i':
			iterations = atoi(optarg);
			break;
		case 'a':
			use_async = 1;
			break;
		case 'h':
			print_usage(argv[0]);
			return 0;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	printf("\n========================================\n");
	printf("NVLink Bandwidth Test\n");
	printf("========================================\n");
	printf("Source GPU: %d\n", gpu0_id);
	printf("Destination GPU: %d\n", gpu1_id);
	printf("Buffer size: %zu bytes (%.2f MB)\n", buffer_size, buffer_size / (1024.0 * 1024.0));
	printf("Iterations: %d\n", iterations);
	printf("Transfer mode: %s\n", use_async ? "Asynchronous" : "Synchronous");
	printf("========================================\n\n");

	/* Check peer access */
	printf("Checking NVLink/P2P connectivity...\n");
	if (!nvlink_check_peer_access(gpu0_id, gpu1_id)) {
		fprintf(stderr, "ERROR: GPU %d cannot access GPU %d (no NVLink/P2P)\n",
			gpu0_id, gpu1_id);
		return 1;
	}
	printf("NVLink/P2P connectivity confirmed\n\n");

	/* Print topology */
	nvlink_print_topology(gpu0_id, gpu1_id);

	/* Initialize NVLink context (enables peer access and creates stream if async) */
	if (nvlink_init_context(&nvlink_ctx, gpu0_id, gpu1_id) != 0) {
		fprintf(stderr, "Failed to initialize NVLink context\n");
		return 1;
	}

	/* Allocate source buffer on GPU 0 */
	printf("Allocating source buffer on GPU %d (%zu bytes)...\n", gpu0_id, buffer_size);
	err = cudaSetDevice(gpu0_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device %d: %s\n", gpu0_id, cudaGetErrorString(err));
		nvlink_cleanup_context(&nvlink_ctx);
		return 1;
	}

	err = cudaMalloc(&src_buffer, buffer_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate source buffer: %s\n", cudaGetErrorString(err));
		nvlink_cleanup_context(&nvlink_ctx);
		return 1;
	}
	printf("Source buffer allocated at %p (0x%016llx)\n", src_buffer, (unsigned long long)(uintptr_t)src_buffer);

	/* Allocate destination buffer on GPU 1 */
	printf("Allocating destination buffer on GPU %d (%zu bytes)...\n", gpu1_id, buffer_size);
	err = cudaSetDevice(gpu1_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device %d: %s\n", gpu1_id, cudaGetErrorString(err));
		cudaSetDevice(gpu0_id);
		cudaFree(src_buffer);
		nvlink_cleanup_context(&nvlink_ctx);
		return 1;
	}

	err = cudaMalloc(&dst_buffer, buffer_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate destination buffer: %s\n", cudaGetErrorString(err));
		cudaSetDevice(gpu0_id);
		cudaFree(src_buffer);
		nvlink_cleanup_context(&nvlink_ctx);
		return 1;
	}
	printf("Destination buffer allocated at %p (0x%016llx)\n\n", dst_buffer, (unsigned long long)(uintptr_t)dst_buffer);


	/* Warmup transfers */
	printf("Performing %d warmup iterations...\n", WARMUP_ITERATIONS);
	for (i = 0; i < WARMUP_ITERATIONS; i++) {
		if (use_async) {
			if (nvlink_copy_gpu_to_gpu_async(&nvlink_ctx, dst_buffer, src_buffer, buffer_size) != 0) {
				fprintf(stderr, "Warmup transfer %d failed\n", i);
				ret = 1;
				goto cleanup;
			}
			if (nvlink_synchronize(&nvlink_ctx) != 0) {
				fprintf(stderr, "Warmup synchronization %d failed\n", i);
				ret = 1;
				goto cleanup;
			}
		} else {
			if (nvlink_copy_gpu_to_gpu_sync(&nvlink_ctx, dst_buffer, src_buffer, buffer_size) != 0) {
				fprintf(stderr, "Warmup transfer %d failed\n", i);
				ret = 1;
				goto cleanup;
			}
		}
	}
	printf("Warmup completed\n\n");

	/* Perform timed transfers */
	printf("Starting bandwidth measurement...\n");
	printf("Running %d iterations...\n", iterations);

	start_time = get_time_usec();

	for (i = 0; i < iterations; i++) {
		if (use_async) {
			if (nvlink_copy_gpu_to_gpu_async(&nvlink_ctx, dst_buffer, src_buffer, buffer_size) != 0) {
				fprintf(stderr, "Transfer %d failed\n", i);
				ret = 1;
				goto cleanup;
			}
		} else {
			if (nvlink_copy_gpu_to_gpu_sync(&nvlink_ctx, dst_buffer, src_buffer, buffer_size) != 0) {
				fprintf(stderr, "Transfer %d failed\n", i);
				ret = 1;
				goto cleanup;
			}
		}
	}

	/* Synchronize if using async */
	if (use_async) {
		if (nvlink_synchronize(&nvlink_ctx) != 0) {
			fprintf(stderr, "Final synchronization failed\n");
			ret = 1;
			goto cleanup;
		}
	}

	end_time = get_time_usec();
	total_time = (end_time - start_time) / 1000000.0;  /* Convert to seconds */

	/* Calculate bandwidth */
	bandwidth_gbps = (buffer_size * iterations * 8.0) / (total_time * 1e9);  /* bits per second to Gbps */

	printf("\n========================================\n");
	printf("Results:\n");
	printf("========================================\n");
	printf("Total time: %.6f seconds\n", total_time);
	printf("Average time per transfer: %.6f seconds\n", total_time / iterations);
	printf("Total data transferred: %.2f MB\n", (buffer_size * iterations) / (1024.0 * 1024.0));
	printf("Bandwidth: %.2f Gbps\n", bandwidth_gbps);
	printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps / 8.0);
	printf("========================================\n\n");

cleanup:
	/* Cleanup NVLink context (synchronizes and destroys stream) */
	nvlink_cleanup_context(&nvlink_ctx);

	/* Free buffers */
	if (dst_buffer) {
		cudaSetDevice(gpu1_id);
		err = cudaFree(dst_buffer);
		if (err != cudaSuccess) {
			fprintf(stderr, "Warning: Failed to free destination buffer: %s\n", cudaGetErrorString(err));
		}
		dst_buffer = NULL;
	}

	if (src_buffer) {
		cudaSetDevice(gpu0_id);
		err = cudaFree(src_buffer);
		if (err != cudaSuccess) {
			fprintf(stderr, "Warning: Failed to free source buffer: %s\n", cudaGetErrorString(err));
		}
		src_buffer = NULL;
	}

	/* Reset device to default */
	cudaSetDevice(0);

	return ret;
}

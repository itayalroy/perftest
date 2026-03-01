/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * Minimal RDMA Multi-QP API
 * Supports multiple NICs with one QP per NIC
 */

#ifndef RDMA_MULTI_QP_H
#define RDMA_MULTI_QP_H

#include "config.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for multi-QP context */
typedef struct rdma_multi_qp_context rdma_multi_qp_context_t;

/**
 * Configuration for multi-QP setup
 */
struct rdma_multi_qp_config {
    const char **nic_names;    /* Array of NIC device names (e.g., "mlx5_0", "mlx5_1") */
    int num_qps;               /* Number of QPs (must match length of nic_names and gpu_id arrays) */
    uint16_t base_port;        /* Base port (port i = base_port + i) */
    const char *server_addr;   /* Server IP (NULL for server mode) */
    int is_server;             /* 1 for server, 0 for client */
    size_t buffer_size;        /* Buffer size per QP */
    int *gpu_id;               /* Array of GPU IDs for buffer allocation per QP (-1 for host) */
    bool nics_only;            /* Whether to only use NICs for buffer allocation */
    bool direct_mode;          /* Each GPU sends directly via its own NIC (no NVLink) */
    bool reassembly;           /* Enable reassembly on receiver side (multi-source NVLink only) */
    bool all_to_all;           /* Spread from each source to every target (N*M QPs); with --direct or --allow-nvlink --reassembly */
    void **external_buffers;   /* Optional: per-QP buffer pointers (e.g. slices of source buffer); if set, no allocation, caller owns memory */
    size_t transport_buffer_size; /* 0 = use full slice size (no piping); otherwise QP buffer size per GPU */
};

/**
 * Initialize multi-QP context
 * Creates num_qps QPs (one per NIC)
 *
 * @param config: Configuration structure
 * @param ctx: Output parameter - pointer to context handle
 * @return: 0 on success, non-zero on error
 */
int rdma_multi_qp_init(const struct rdma_multi_qp_config *config,
                       rdma_multi_qp_context_t **ctx, bool nics_only);

/**
 * Connect QPs to remote peer
 * QP 0 connects to remote QP 0, QP 1 connects to remote QP 1
 *
 * @param ctx: Context handle
 * @return: 0 on success, non-zero on error
 */
int rdma_multi_qp_connect(rdma_multi_qp_context_t *ctx);

/**
 * Perform RDMA Write on a specific QP
 *
 * @param ctx: Context handle
 * @param qp_index: QP index (0 or 1)
 * @param local_offset: Offset in local buffer
 * @param size: Size to write
 * @param remote_offset: Offset in remote buffer
 * @return: 0 on success, non-zero on error
 */
int rdma_write(rdma_multi_qp_context_t *ctx,
               int qp_index,
               uint64_t local_offset,
               uint64_t size,
               uint64_t remote_offset);

/**
 * Poll for completion on a specific QP
 *
 * @param ctx: Context handle
 * @param qp_index: QP index (0 or 1)
 * @param timeout_ms: Timeout in milliseconds (-1 for infinite)
 * @return: 0 on success (completion found), 1 on timeout, negative on error
 */
int rdma_poll_completion(rdma_multi_qp_context_t *ctx,
                         int qp_index,
                         int timeout_ms);

/**
 * Get local buffer pointer for a QP
 *
 * @param ctx: Context handle
 * @param qp_index: QP index (0 or 1)
 * @return: Buffer pointer, NULL on error
 */
void *rdma_get_local_buffer(rdma_multi_qp_context_t *ctx, int qp_index);

/**
 * Get buffer size
 *
 * @param ctx: Context handle
 * @return: Buffer size in bytes, 0 on error
 */
size_t rdma_get_buffer_size(rdma_multi_qp_context_t *ctx);

/**
 * Get number of QPs in context
 *
 * @param ctx: Context handle
 * @return: Number of QPs, 0 on error
 */
int rdma_multi_qp_get_num_qps(rdma_multi_qp_context_t *ctx);

/**
 * RDMA write with immediate data
 *
 * @param ctx: Context handle
 * @param qp_index: QP index
 * @param local_offset: Offset in local buffer
 * @param size: Number of bytes to write
 * @param remote_offset: Offset in remote buffer
 * @param imm_data: 32-bit immediate data to send with completion
 * @return: 0 on success, non-zero on error
 */
int rdma_write_with_imm(rdma_multi_qp_context_t *ctx,
                        int qp_index,
                        uint64_t local_offset,
                        uint64_t size,
                        uint64_t remote_offset,
                        uint32_t imm_data);

/**
 * Post receive work request
 *
 * @param ctx: Context handle
 * @param qp_index: QP index
 * @param local_offset: Offset in local buffer for receive
 * @param size: Maximum size to receive
 * @param wr_id: Work request ID (returned in completion; use 0 if not needed)
 * @return: 0 on success, non-zero on error
 */
int rdma_post_receive(rdma_multi_qp_context_t *ctx,
                      int qp_index,
                      uint64_t local_offset,
                      uint64_t size,
                      uint64_t wr_id);

/**
 * Poll for completion with immediate data
 *
 * @param ctx: Context handle
 * @param qp_index: QP index
 * @param timeout_ms: Timeout in milliseconds (-1 for infinite)
 * @param imm_data: Output parameter for immediate data (can be NULL)
 * @param wr_id_out: Output parameter for work request ID from recv (can be NULL)
 * @return: 0 on success, 1 on timeout, negative on error
 */
int rdma_poll_completion_with_imm(rdma_multi_qp_context_t *ctx,
                                   int qp_index,
                                   int timeout_ms,
                                   uint32_t *imm_data,
                                   uint64_t *wr_id_out);

/**
 * Cleanup and destroy multi-QP context
 *
 * @param ctx: Context handle to destroy
 * @return: 0 on success, non-zero on error
 */
int rdma_multi_qp_cleanup(rdma_multi_qp_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* RDMA_MULTI_QP_H */


/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2024
 *
 * Minimal RDMA Multi-QP Implementation
 * Opens 2 QPs on 2 IB devices and connects them to remote QPs
 */

#include "config.h"
#include "rdma_multi_qp.h"
#include "perftest_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#define NUM_QPS 2
#define QUEUE_DEPTH 256
#define HANDSHAKE_PORT_OFFSET 30000

struct qp_context {
    struct ibv_device *dev;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    void *buffer;
    size_t buffer_size;
    
    /* Connection info */
    uint32_t local_qpn;
    uint32_t remote_qpn;
    uint16_t local_lid;
    uint16_t remote_lid;
    uint32_t local_psn;
    uint32_t remote_psn;
    uint32_t rkey;
    uint64_t vaddr;
    uint32_t remote_rkey;
    uint64_t remote_vaddr;
    uint8_t port_num;
    uint16_t pkey_index;
    enum ibv_mtu mtu;
    union ibv_gid local_gid;
    union ibv_gid remote_gid;
    int gid_index;
    int connected;
};

struct rdma_multi_qp_context {
    struct qp_context qps[NUM_QPS];
    int is_server;
    uint16_t base_port;
    const char *server_addr;
    int gpu_id;
    pthread_mutex_t mutex[NUM_QPS];
};

static int allocate_buffer(struct qp_context *qp, size_t size, int gpu_id)
{
#ifdef HAVE_CUDA
    if (gpu_id >= 0) {
        cudaError_t err = cudaSetDevice(gpu_id);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&qp->buffer, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
    } else
#endif
    {
        qp->buffer = malloc(size);
        if (!qp->buffer) {
            fprintf(stderr, "malloc failed\n");
            return -1;
        }
        memset(qp->buffer, 0, size);
    }
    qp->buffer_size = size;
    return 0;
}

static void free_buffer(struct qp_context *qp, int gpu_id)
{
    if (!qp->buffer) return;
#ifdef HAVE_CUDA
    if (gpu_id >= 0) {
        cudaFree(qp->buffer);
    } else
#endif
    {
        free(qp->buffer);
    }
    qp->buffer = NULL;
}

static int setup_device(struct qp_context *qp, const char *dev_name)
{
    struct ibv_device **dev_list;
    struct ibv_port_attr port_attr;
    int num_devices;
    int i;
    uint16_t pkey;
    
    /* Find device */
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        fprintf(stderr, "ibv_get_device_list failed\n");
        return -1;
    }
    
    qp->dev = NULL;
    for (i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
            qp->dev = dev_list[i];
            break;
        }
    }
    
    if (!qp->dev) {
        fprintf(stderr, "Device %s not found\n", dev_name);
        ibv_free_device_list(dev_list);
        return -1;
    }
    
    /* Open device */
    qp->ctx = ibv_open_device(qp->dev);
    ibv_free_device_list(dev_list);
    if (!qp->ctx) {
        fprintf(stderr, "ibv_open_device failed\n");
        return -1;
    }
    
    /* Query port (try 1, then 2) */
    qp->port_num = 1;
    if (ibv_query_port(qp->ctx, qp->port_num, &port_attr) != 0) {
        qp->port_num = 2;
        if (ibv_query_port(qp->ctx, qp->port_num, &port_attr) != 0) {
            fprintf(stderr, "ibv_query_port failed\n");
            ibv_close_device(qp->ctx);
            return -1;
        }
    }
    
    qp->local_lid = port_attr.lid;
    qp->mtu = port_attr.active_mtu;
    
    /* Query GID (use index 0, or find best GID) */
    qp->gid_index = 0;
    if (port_attr.gid_tbl_len > 0) {
        if (ibv_query_gid(qp->ctx, qp->port_num, qp->gid_index, &qp->local_gid) != 0) {
            fprintf(stderr, "ibv_query_gid failed\n");
            ibv_close_device(qp->ctx);
            return -1;
        }
    }
    
    fprintf(stderr, "DEBUG [%s]: Port %d - LID=%u, State=%d, MTU=%d, GID index=%d\n",
            dev_name, qp->port_num, qp->local_lid, port_attr.state, qp->mtu, qp->gid_index);
    
    /* Find valid pkey */
    qp->pkey_index = 0;
    for (i = 0; i < port_attr.pkey_tbl_len; i++) {
        if (ibv_query_pkey(qp->ctx, qp->port_num, i, &pkey) == 0) {
            pkey = ntohs(pkey);
            if (pkey & 0x8000) {
                qp->pkey_index = i;
                break;
            }
        }
    }
    
    /* Allocate PD */
    qp->pd = ibv_alloc_pd(qp->ctx);
    if (!qp->pd) {
        fprintf(stderr, "ibv_alloc_pd failed\n");
        ibv_close_device(qp->ctx);
        return -1;
    }
    
    /* Create CQ */
    qp->cq = ibv_create_cq(qp->ctx, QUEUE_DEPTH, NULL, NULL, 0);
    if (!qp->cq) {
        fprintf(stderr, "ibv_create_cq failed\n");
        ibv_dealloc_pd(qp->pd);
        ibv_close_device(qp->ctx);
        return -1;
    }
    
    /* Create QP */
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.send_cq = qp->cq;
    qp_attr.recv_cq = qp->cq;
    qp_attr.cap.max_send_wr = QUEUE_DEPTH;
    qp_attr.cap.max_recv_wr = QUEUE_DEPTH;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp_attr.cap.max_inline_data = 0;
    qp_attr.sq_sig_all = 0;
    
    qp->qp = ibv_create_qp(qp->pd, &qp_attr);
    if (!qp->qp) {
        fprintf(stderr, "ibv_create_qp failed\n");
        ibv_destroy_cq(qp->cq);
        ibv_dealloc_pd(qp->pd);
        ibv_close_device(qp->ctx);
        return -1;
    }
    
    qp->local_qpn = qp->qp->qp_num;
    qp->local_psn = rand() & 0xffffff;
    qp->connected = 0;
    
    return 0;
}

static int register_mr(struct qp_context *qp, int gpu_id)
{
    int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    
    qp->mr = ibv_reg_mr(qp->pd, qp->buffer, qp->buffer_size, flags);
    if (!qp->mr) {
        fprintf(stderr, "ibv_reg_mr failed\n");
        return -1;
    }
    
    qp->rkey = qp->mr->rkey;
    qp->vaddr = (uint64_t)(uintptr_t)qp->buffer;
    
    return 0;
}

static int exchange_qp_info(rdma_multi_qp_context_t *ctx, int qp_idx, int sockfd)
{
    struct qp_context *qp = &ctx->qps[qp_idx];
    char local_msg[512], remote_msg[512];
    int i;
    
    /* Format message: QPN VADDR LID PSN RKEY GID[16 bytes as hex] */
    char gid_str[33];
    for (i = 0; i < 16; i++) {
        sprintf(&gid_str[i*2], "%02x", qp->local_gid.raw[i]);
    }
    gid_str[32] = '\0';
    
    snprintf(local_msg, sizeof(local_msg), "%u %lu %hu %u %u %s",
             qp->local_qpn, qp->vaddr, qp->local_lid, qp->local_psn, qp->rkey, gid_str);
    
    fprintf(stderr, "DEBUG [QP %d %s]: Sending - QPN=%u, VADDR=%lu, LID=%u, PSN=%u, RKEY=%u, GID=%s\n",
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT",
            qp->local_qpn, qp->vaddr, qp->local_lid, qp->local_psn, qp->rkey, gid_str);
    
    if (ctx->is_server) {
        send(sockfd, local_msg, sizeof(local_msg), 0);
        recv(sockfd, remote_msg, sizeof(remote_msg), 0);
    } else {
        recv(sockfd, remote_msg, sizeof(remote_msg), 0);
        send(sockfd, local_msg, sizeof(local_msg), 0);
    }
    
    fprintf(stderr, "DEBUG [QP %d %s]: Received raw: %s\n",
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT", remote_msg);
    
    char remote_gid_str[33];
    int parsed = sscanf(remote_msg, "%u %lu %hu %u %u %32s",
                        &qp->remote_qpn, &qp->remote_vaddr, &qp->remote_lid,
                        &qp->remote_psn, &qp->remote_rkey, remote_gid_str);
    
    /* Parse GID from hex string */
    if (parsed >= 6) {
        for (i = 0; i < 16; i++) {
            unsigned int val;
            sscanf(&remote_gid_str[i*2], "%02x", &val);
            qp->remote_gid.raw[i] = (uint8_t)val;
        }
    }
    
    fprintf(stderr, "DEBUG [QP %d %s]: Parsed %d values - Remote QPN=%u, VADDR=%lu, LID=%u, PSN=%u, RKEY=%u, GID=%s\n",
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT", parsed,
            qp->remote_qpn, qp->remote_vaddr, qp->remote_lid,
            qp->remote_psn, qp->remote_rkey, remote_gid_str);
    
    return 0;
}

static int handshake(rdma_multi_qp_context_t *ctx, int qp_idx)
{
    struct qp_context *qp = &ctx->qps[qp_idx];
    int sockfd;
    struct sockaddr_in addr;
    char local_msg[512], remote_msg[512];
    int port = ctx->base_port + HANDSHAKE_PORT_OFFSET + qp_idx;
    int i;
    
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        fprintf(stderr, "socket failed\n");
        return -1;
    }
    
    /* Enable SO_REUSEADDR to allow binding to address already in use */
    int reuse = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        fprintf(stderr, "setsockopt SO_REUSEADDR failed: %s\n", strerror(errno));
        close(sockfd);
        return -1;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    fprintf(stderr, "DEBUG [QP %d %s]: Using handshake port %d\n", 
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT", port);
    
    if (ctx->is_server) {
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            fprintf(stderr, "bind failed: %s\n", strerror(errno));
            close(sockfd);
            return -1;
        }
        fprintf(stderr, "DEBUG [QP %d SERVER]: Bound to port %d, listening...\n", qp_idx, port);
        if (listen(sockfd, 1) < 0) {
            fprintf(stderr, "listen failed\n");
            close(sockfd);
            return -1;
        }
        int client_fd = accept(sockfd, NULL, NULL);
        close(sockfd);
        sockfd = client_fd;
        fprintf(stderr, "DEBUG [QP %d SERVER]: Accepted connection\n", qp_idx);
    } else {
        addr.sin_addr.s_addr = ctx->server_addr ? 
            inet_addr(ctx->server_addr) : inet_addr("127.0.0.1");
        fprintf(stderr, "DEBUG [QP %d CLIENT]: Connecting to %s:%d...\n", 
                qp_idx, ctx->server_addr ? ctx->server_addr : "127.0.0.1", port);
        if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            fprintf(stderr, "connect failed: %s\n", strerror(errno));
            close(sockfd);
            return -1;
        }
        fprintf(stderr, "DEBUG [QP %d CLIENT]: Connected\n", qp_idx);
    }
    
    /* Format message: QPN VADDR LID PSN RKEY GID[16 bytes as hex] */
    char gid_str[33];
    for (i = 0; i < 16; i++) {
        sprintf(&gid_str[i*2], "%02x", qp->local_gid.raw[i]);
    }
    gid_str[32] = '\0';
    
    snprintf(local_msg, sizeof(local_msg), "%u %lu %hu %u %u %s",
             qp->local_qpn, qp->vaddr, qp->local_lid, qp->local_psn, qp->rkey, gid_str);
    
    fprintf(stderr, "DEBUG [QP %d %s]: Sending - QPN=%u, VADDR=%lu, LID=%u, PSN=%u, RKEY=%u, GID=%s\n",
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT",
            qp->local_qpn, qp->vaddr, qp->local_lid, qp->local_psn, qp->rkey, gid_str);
    
    if (ctx->is_server) {
        send(sockfd, local_msg, sizeof(local_msg), 0);
        recv(sockfd, remote_msg, sizeof(remote_msg), 0);
    } else {
        recv(sockfd, remote_msg, sizeof(remote_msg), 0);
        send(sockfd, local_msg, sizeof(local_msg), 0);
    }
    
    fprintf(stderr, "DEBUG [QP %d %s]: Received raw: %s\n",
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT", remote_msg);
    
    char remote_gid_str[33];
    int parsed = sscanf(remote_msg, "%u %lu %hu %u %u %32s",
                        &qp->remote_qpn, &qp->remote_vaddr, &qp->remote_lid,
                        &qp->remote_psn, &qp->remote_rkey, remote_gid_str);
    
    /* Parse GID from hex string */
    if (parsed >= 6) {
        for (i = 0; i < 16; i++) {
            unsigned int val;
            sscanf(&remote_gid_str[i*2], "%02x", &val);
            qp->remote_gid.raw[i] = (uint8_t)val;
        }
    }
    
    fprintf(stderr, "DEBUG [QP %d %s]: Parsed %d values - Remote QPN=%u, VADDR=%lu, LID=%u, PSN=%u, RKEY=%u, GID=%s\n",
            qp_idx, ctx->is_server ? "SERVER" : "CLIENT", parsed,
            qp->remote_qpn, qp->remote_vaddr, qp->remote_lid,
            qp->remote_psn, qp->remote_rkey, remote_gid_str);
    
    close(sockfd);
    return 0;
}

static int modify_qp_to_rts(struct qp_context *qp)
{
    struct ibv_qp_attr attr;
    struct ibv_ah_attr ah_attr;
    int flags;
    
    /* INIT */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = qp->port_num;
    attr.pkey_index = qp->pkey_index;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp->qp, &attr, flags)) {
        fprintf(stderr, "Failed to modify QP to INIT\n");
        return -1;
    }
    
    /* RTR */
    memset(&ah_attr, 0, sizeof(ah_attr));
    ah_attr.port_num = qp->port_num;
    ah_attr.sl = 0;
    ah_attr.src_path_bits = 0;
    
    /* Use GID-based addressing if LID is 0 */
    if (qp->local_lid == 0 || qp->remote_lid == 0) {
        ah_attr.is_global = 1;
        ah_attr.grh.dgid = qp->remote_gid;
        ah_attr.grh.sgid_index = qp->gid_index;
        ah_attr.grh.hop_limit = 0xFF;
        ah_attr.grh.traffic_class = 0;
        fprintf(stderr, "DEBUG: Using GID-based addressing - Local GID index=%d\n", qp->gid_index);
    } else {
        ah_attr.is_global = 0;
        ah_attr.dlid = qp->remote_lid;
        fprintf(stderr, "DEBUG: Using LID-based addressing - Remote LID=%u\n", qp->remote_lid);
    }
    
    fprintf(stderr, "DEBUG: Modifying QP to RTR - Port=%d, Remote QPN=%u, Remote PSN=%u, MTU=%d\n",
            qp->port_num, qp->remote_qpn, qp->remote_psn, qp->mtu);
    
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = qp->mtu;
    attr.dest_qp_num = qp->remote_qpn;
    attr.rq_psn = qp->remote_psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr = ah_attr;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (ibv_modify_qp(qp->qp, &attr, flags)) {
        fprintf(stderr, "Failed to modify QP to RTR: %s\n", strerror(errno));
        return -1;
    }
    
    /* RTS */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 0x12;
    attr.retry_cnt = 14;  /* Increased from 7 to handle packet loss better */
    attr.rnr_retry = 7;
    attr.sq_psn = qp->local_psn;
    attr.max_rd_atomic = 16;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    if (ibv_modify_qp(qp->qp, &attr, flags)) {
        fprintf(stderr, "Failed to modify QP to RTS\n");
        return -1;
    }
    
    return 0;
}

int rdma_multi_qp_init(const struct rdma_multi_qp_config *config,
                       rdma_multi_qp_context_t **ctx_out)
{
    rdma_multi_qp_context_t *ctx;
    int i;
    
    if (!config || !ctx_out) return -1;
    
    srand(time(NULL));
    
    ctx = calloc(1, sizeof(rdma_multi_qp_context_t));
    if (!ctx) {
        fprintf(stderr, "calloc failed\n");
        return -1;
    }
    
    ctx->is_server = config->is_server;
    ctx->base_port = config->base_port;
    ctx->server_addr = config->server_addr;
    ctx->gpu_id = config->gpu_id[0];  /* Keep for backward compatibility */
    
    for (i = 0; i < NUM_QPS; i++) {
        pthread_mutex_init(&ctx->mutex[i], NULL);
        
        if (setup_device(&ctx->qps[i], config->nic_names[i]) != 0) {
            fprintf(stderr, "Failed to setup device %d\n", i);
            goto error;
        }
        
        if (allocate_buffer(&ctx->qps[i], config->buffer_size, config->gpu_id[i]) != 0) {
            fprintf(stderr, "Failed to allocate buffer %d\n", i);
            goto error;
        }
        
        if (register_mr(&ctx->qps[i], config->gpu_id[i]) != 0) {
            fprintf(stderr, "Failed to register MR %d\n", i);
            goto error;
        }
    }
    
    *ctx_out = ctx;
    return 0;
    
error:
    for (i = 0; i < NUM_QPS; i++) {
        if (ctx->qps[i].buffer) free_buffer(&ctx->qps[i], config->gpu_id[i]);
        if (ctx->qps[i].mr) ibv_dereg_mr(ctx->qps[i].mr);
        if (ctx->qps[i].qp) ibv_destroy_qp(ctx->qps[i].qp);
        if (ctx->qps[i].cq) ibv_destroy_cq(ctx->qps[i].cq);
        if (ctx->qps[i].pd) ibv_dealloc_pd(ctx->qps[i].pd);
        if (ctx->qps[i].ctx) ibv_close_device(ctx->qps[i].ctx);
        pthread_mutex_destroy(&ctx->mutex[i]);
    }
    free(ctx);
    return -1;
}

int rdma_multi_qp_connect(rdma_multi_qp_context_t *ctx)
{
    int i;
    int listen_socks[NUM_QPS] = {-1, -1};
    int client_socks[NUM_QPS] = {-1, -1};
    struct sockaddr_in addr;
    int reuse = 1;
    
    /* Phase 1: Set up all listening sockets (server) or connect to all (client) */
    for (i = 0; i < NUM_QPS; i++) {
        int port = ctx->base_port + HANDSHAKE_PORT_OFFSET + i;
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            fprintf(stderr, "socket failed for QP %d\n", i);
            goto error;
        }
        
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
            fprintf(stderr, "setsockopt SO_REUSEADDR failed for QP %d: %s\n", i, strerror(errno));
            close(sockfd);
            goto error;
        }
        
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        
        if (ctx->is_server) {
            addr.sin_addr.s_addr = INADDR_ANY;
            if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
                fprintf(stderr, "bind failed for QP %d: %s\n", i, strerror(errno));
                close(sockfd);
                goto error;
            }
            if (listen(sockfd, 1) < 0) {
                fprintf(stderr, "listen failed for QP %d\n", i);
                close(sockfd);
                goto error;
            }
            listen_socks[i] = sockfd;
            fprintf(stderr, "DEBUG [QP %d SERVER]: Listening on port %d\n", i, port);
        } else {
            addr.sin_addr.s_addr = ctx->server_addr ? 
                inet_addr(ctx->server_addr) : inet_addr("127.0.0.1");
            fprintf(stderr, "DEBUG [QP %d CLIENT]: Connecting to %s:%d...\n", 
                    i, ctx->server_addr ? ctx->server_addr : "127.0.0.1", port);
            if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
                fprintf(stderr, "connect failed for QP %d: %s\n", i, strerror(errno));
                close(sockfd);
                goto error;
            }
            client_socks[i] = sockfd;
            fprintf(stderr, "DEBUG [QP %d CLIENT]: Connected\n", i);
        }
    }
    
    /* Phase 2: Accept connections (server) or use existing connections (client) */
    for (i = 0; i < NUM_QPS; i++) {
        int sockfd;
        if (ctx->is_server) {
            fprintf(stderr, "DEBUG [QP %d SERVER]: Accepting connection...\n", i);
            sockfd = accept(listen_socks[i], NULL, NULL);
            close(listen_socks[i]);
            listen_socks[i] = -1;
            if (sockfd < 0) {
                fprintf(stderr, "accept failed for QP %d\n", i);
                goto error;
            }
            fprintf(stderr, "DEBUG [QP %d SERVER]: Accepted connection\n", i);
        } else {
            sockfd = client_socks[i];
        }
        
        /* Exchange QP information */
        if (exchange_qp_info(ctx, i, sockfd) != 0) {
            fprintf(stderr, "Failed to exchange QP info for QP %d\n", i);
            close(sockfd);
            goto error;
        }
        
        close(sockfd);
        
        /* Modify QP to RTS */
        if (modify_qp_to_rts(&ctx->qps[i]) != 0) {
            fprintf(stderr, "Failed to modify QP %d to RTS\n", i);
            goto error;
        }
        
        ctx->qps[i].connected = 1;
    }
    
    return 0;
    
error:
    /* Cleanup */
    for (i = 0; i < NUM_QPS; i++) {
        if (listen_socks[i] >= 0) close(listen_socks[i]);
        if (client_socks[i] >= 0) close(client_socks[i]);
    }
    return -1;
}

int rdma_write(rdma_multi_qp_context_t *ctx, int qp_index,
               uint64_t local_offset, uint64_t size, uint64_t remote_offset)
{
    struct qp_context *qp;
    struct ibv_sge sge;
    struct ibv_send_wr wr, *bad_wr;
    
    if (!ctx || qp_index < 0 || qp_index >= NUM_QPS || !ctx->qps[qp_index].connected) {
        return -1;
    }
    
    qp = &ctx->qps[qp_index];
    pthread_mutex_lock(&ctx->mutex[qp_index]);
    
    sge.addr = (uint64_t)(uintptr_t)qp->buffer + local_offset;
    sge.length = size;
    sge.lkey = qp->mr->lkey;
    
    memset(&wr, 0, sizeof(wr));
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = qp->remote_vaddr + remote_offset;
    wr.wr.rdma.rkey = qp->remote_rkey;
    
    if (ibv_post_send(qp->qp, &wr, &bad_wr)) {
        pthread_mutex_unlock(&ctx->mutex[qp_index]);
        return -1;
    }
    
    pthread_mutex_unlock(&ctx->mutex[qp_index]);
    return 0;
}

int rdma_poll_completion(rdma_multi_qp_context_t *ctx, int qp_index, int timeout_ms)
{
    struct qp_context *qp;
    struct ibv_wc wc;
    int ne;
    int polled = 0;
    
    if (!ctx || qp_index < 0 || qp_index >= NUM_QPS) return -1;
    
    qp = &ctx->qps[qp_index];
    
    while (polled < timeout_ms || timeout_ms < 0) {
        ne = ibv_poll_cq(qp->cq, 1, &wc);
        if (ne > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "WC error: %s\n", ibv_wc_status_str(wc.status));
                return -1;
            }
            return 0;
        }
    }
    
    return 1; /* Timeout */
}

void *rdma_get_local_buffer(rdma_multi_qp_context_t *ctx, int qp_index)
{
    if (!ctx || qp_index < 0 || qp_index >= NUM_QPS) return NULL;
    return ctx->qps[qp_index].buffer;
}

size_t rdma_get_buffer_size(rdma_multi_qp_context_t *ctx)
{
    if (!ctx) return 0;
    return ctx->qps[0].buffer_size;
}

int rdma_multi_qp_cleanup(rdma_multi_qp_context_t *ctx)
{
    int i;
    
    if (!ctx) return -1;
    
    for (i = 0; i < NUM_QPS; i++) {
        if (ctx->qps[i].qp) ibv_destroy_qp(ctx->qps[i].qp);
        if (ctx->qps[i].cq) ibv_destroy_cq(ctx->qps[i].cq);
        if (ctx->qps[i].mr) ibv_dereg_mr(ctx->qps[i].mr);
        if (ctx->qps[i].buffer) free_buffer(&ctx->qps[i], ctx->gpu_id);
        if (ctx->qps[i].pd) ibv_dealloc_pd(ctx->qps[i].pd);
        if (ctx->qps[i].ctx) ibv_close_device(ctx->qps[i].ctx);
        pthread_mutex_destroy(&ctx->mutex[i]);
    }
    
    free(ctx);
    return 0;
}


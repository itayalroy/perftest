# All-to-All Configurable Target Count: Design Document

## Overview

This document describes the desired behavior and implementation plan for decoupling the **number of logical targets** (M) from the **number of transport NICs/QPs** (K) in supported modes. Currently both are tied to `-n`/`-g`. The original goal was to allow M ≠ K in NVLink all-to-all reassembly mode:

- **2 sources, 4 targets, 8 transport NICs** (N=2, M=4, K=8)
- **4 sources, 2 targets, 8 transport NICs** (N=4, M=2, K=8)

This has since been extended (see §Extension below) to two additional modes where M < K means **"use only the first M NICs/GPUs"** (i.e. K is effectively capped to M, not kept independent).

---

## Terminology

| Symbol | Meaning | Current Source | Proposed |
|--------|---------|----------------|----------|
| **N** | Number of source GPUs | `--source-gpus` | `--source-gpus` (unchanged) |
| **M** | Number of logical targets | `-n` (same as NIC count) | New `--target-count` option |
| **K** | Number of transport NICs/QPs | `-n`, `-g` | `-n`, `-g` (unchanged for reassembly; capped to M for other modes) |

When `--target-count` is not specified: **M = K** (current behavior, fully backward compatible).

---

## Scope

Three modes support `--target-count`:

| Mode flags | M ≠ K semantics |
|------------|-----------------|
| `--allow-nvlink --reassembly --all-to-all` | M and K are truly independent (original design) |
| `--direct --all-to-all` | K is capped to M; only first M NICs/GPUs used |
| `--allow-nvlink` (no `--reassembly`, no `--all-to-all`) | K is capped to M; only first M NICs/GPUs used |

`--allow-nvlink --all-to-all` without `--reassembly` is **not** affected.

---

## Desired Behavior

### Example 1: N=2 sources, M=4 targets, K=8 NICs

#### Data flow (per iteration)

```
Source GPU n (n=0,1): 1 GB → split into M=4 logical slices of 256 MB each

For each target m (0..3), for each transport NIC k (0..7):
  sub_slice = 256 MB / 8 = 32 MB
  Thread (n, k, m): NVLink-copy sub_slice from source[n][m*256MB + k*32MB]
                    → QP buffer k, slot (n*M+m), size 32 MB
                    → RDMA write on NIC k

Receiver QP k: receives N×M = 2×4 = 8 sub-slices
  Slot (n,m): reassemble → reassembly_buffers_2d[n*M+m] at offset k*32MB
```

**Result:** M=4 target buffers, each of size N×(1GB/M) = 2×256MB = 512 MB.
Target buffer m lives on GPU m (= target_gpu_ids[m], requires m < K).

#### Thread count
- Sender: N×K×M = 2×8×4 = 64 threads
- Receiver: K = 8 threads (unchanged)

---

### Example 2: N=4 sources, M=2 targets, K=8 NICs

```
Source GPU n (n=0..3): 1 GB → 2 slices of 512 MB each

sub_slice = 512 MB / 8 = 64 MB
Thread (n, k, m): copies sub_slice → QP buffer k, slot (n*M+m)

Receiver QP k: receives 4×2 = 8 sub-slices
```

**Result:** M=2 target buffers, each 4×512MB = 2 GB.
Sender threads: 4×8×2 = 64.

---

## Key Formulas (M ≠ K)

| Quantity | Formula | Note |
|----------|---------|------|
| sub_slice_size | `1GB / (M × K)` | Was `1GB / K²` (= `1GB/(M×K)` when M=K) |
| slice_size (per target, per source) | `1GB / M` | Each source GPU's contribution to one target |
| QP buffer size per QP | `N × (1GB / K)` | **Unchanged!** N×M slots × sub_slice = N/K GB |
| Sender threads | `N × K × M` | Was `N × K × K` |
| Receiver threads | `K` | **Unchanged** |
| reassembly_buffers_2d | `N × M` buffers, size `1GB/M` each | Was N×K buffers of size 1GB/K |
| Reassembly NVLink contexts | `K × M` | Was K×K; index `k*M+m` |
| Sender NVLink contexts | `N × K × M` | Was N×K²; one per thread |

> **Why QP buffer size is unchanged:** Each QP now holds N×M sub-slices (was N×K). Each sub-slice is smaller: `1GB/(M×K)` instead of `1GB/K²`. Total = N×M × 1GB/(M×K) = N × 1GB/K. The existing formula `actual_buffer_size = (1GB/K) * N` remains correct.

---

## Open Questions — Resolved

### Q1: Target GPU assignment when M < K

**Answer:** Target buffer m lives on `target_gpu_ids[m]` — the m-th entry of the GPU list passed via `-g`.
- This requires M ≤ K.
- If M > K: error (cannot place M distinct target buffers with only K GPUs/NICs).
- User must ensure first M GPUs in `-g` are the desired target GPUs.

### Q2 & Q3: Direct mode with M ≠ K

**Answer:** `--target-count` does **not** affect direct mode. In direct all-to-all, M is always equal to the number of NICs specified via `-n`. Document validation should reject `--target-count` combined with `--direct`.

---

## Implementation Changes

### 1. New Command-Line Option

Add `--target-count M` (short `-C M`) to avoid collision with existing short options:

```c
{"target-count", required_argument, 0, 'C'},
```

Parse it into a new config field (see §4). Add to `getopt_long` string: `"C:"`.

**Validation** (after parsing):
```c
if (num_targets != 0) {
    if (config.direct_mode) {
        fprintf(stderr, "--target-count is not supported in direct mode\n");
        return 1;
    }
    if (!config.all_to_all || !config.reassembly) {
        fprintf(stderr, "--target-count requires --all-to-all --allow-nvlink --reassembly\n");
        return 1;
    }
    if (num_targets > num_nics) {
        fprintf(stderr, "--target-count (%d) cannot exceed number of NICs (%d)\n", num_targets, num_nics);
        return 1;
    }
}
int M = (num_targets > 0) ? num_targets : num_nics;  /* default: M = K */
```

### 2. Config Struct (`src/rdma_multi_qp.h`)

```c
struct rdma_multi_qp_config {
    /* ... existing fields ... */
    int num_targets;  /* M, logical target count; 0 = use num_nics (M=K) */
};
```

### 3. Thread Args Struct (`struct thread_args` in `test_rdma_multi_qp.c`)

Add one field:
```c
int num_targets;   /* M: logical target count (num_qps when M=K, otherwise < num_qps) */
```

The existing `num_qps` remains **K** (transport NIC/QP count). In current all-to-all code, references to `num_qps` in sender/receiver threads that mean "M" (e.g., `M = args->num_qps` in `receive_thread_all_to_all_reassembly`) must be updated to use `args->num_targets`.

### 4. Thread Setup Env (`struct thread_setup_env`)

Add `int num_targets;` parallel to `num_nics`.

### 5. Thread Count and Setup — Sender

**Location:** `setup_thread_args`, all-to-all reassembly sender block (around line 197).

**Current** (N×K×K, innermost j loops 0..K-1):
```c
for (int j = 0; j < num_qps; j++) {
    int thread_idx = n * num_qps * num_qps + i * num_qps + j;
```

**New** (N×K×M, innermost j loops 0..M-1):
```c
int M = env->num_targets;  /* logical target count */
int K = env->num_qps;      /* transport QP count */
/* ... */
for (int j = 0; j < M; j++) {
    int thread_idx = n * K * M + i * M + j;
    /* ... */
    a->num_targets = M;
    /* a->num_qps remains K */
```

**Thread count** update in main (around line 1763):
```c
/* Old: */
num_threads = num_source_gpus * num_qps * num_qps;  /* N×K×K */
/* New: */
num_threads = num_source_gpus * num_qps * M;         /* N×K×M */
```

### 6. Sub-Slice Size

**Location:** two places — sender thread setup (~line 172) and receiver thread setup (~line 172):

```c
/* Old: */
size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);    /* 1GB/K² */
/* New: */
size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / ((size_t)M * num_qps);  /* 1GB/(M×K) */
```

### 7. QP Buffer Size

**Location:** buffer sizing block (around line 1810).

**No formula change needed.** The existing formula:
```c
size_t slice_size_at = config.buffer_size / num_nics;   /* 1GB / K */
actual_buffer_size   = slice_size_at * num_source_gpus; /* N × 1GB/K */
```
produces the correct answer regardless of M, because N×M slots each of 1GB/(M×K) = N × 1GB/K total.

### 8. Sender `write_thread` — NVLink Copy Offsets

**Location:** `write_thread`, `all_to_all_reassembly` branch (around line 843).

```c
/* Old: */
size_t slice_size = DEFAULT_BUFFER_SIZE / args->num_qps;  /* 1GB / K (wrong when M≠K!) */

/* New: */
int M = args->num_targets;  /* logical target count */
int K = args->num_qps;      /* transport QP count */
size_t slice_size = DEFAULT_BUFFER_SIZE / (size_t)M;  /* 1GB / M: per-target slice */
```

Source and destination offsets remain structurally the same — just M now comes from `num_targets` instead of `num_qps`:
```c
/* j = args->slice_index_j (now 0..M-1), qp_index = transport index k (0..K-1) */
src_offset = (size_t)j * slice_size + (size_t)qp_index * args->sub_slice_size;
dst_offset = (size_t)(n * M + j) * slot_size;
```

Also update the `imm_val` encoding (around line 1013) which encodes the slot index `n*M+j`:
```c
/* Old (WRONG when M≠K — encodes n*K+j instead of n*M+j): */
imm_val = (uint32_t)(args->source_gpu_index * args->num_qps + args->slice_index_j);

/* New: */
imm_val = (uint32_t)(args->source_gpu_index * args->num_targets + args->slice_index_j);
```
The receiver decodes `n = imm / M; j = imm % M`, so using K instead of M here would corrupt slot assignment when M ≠ K.

### 9. Receiver `receive_thread_all_to_all_reassembly`

**Location:** function starting around line 563.

**Current** `M = args->num_qps`. Replace with:
```c
int M = args->num_targets;  /* logical target count */
int K = args->num_qps;      /* transport QP count (= number of receiver threads) */
```

**Slots to poll per iteration:** `N × M` (was `N × K`). All references to `N * M` in the loop bounds stay correct once `M` is redefined.

**Reassembly context index:** `ctx_idx = qp_index * M + j` — unchanged in form, but now `M = num_targets`, `j` runs 0..M-1.

**No-piping reassembly loop** (around line 692):
```c
for (int n = 0; n < N; n++) {
    for (int j = 0; j < M; j++) {                     /* j = target index, 0..M-1 */
        void *dst_buffer = args->reassembly_buffers_2d[n * M + j];
        size_t dst_offset = (size_t)qp_index * sub_slice;
        /* slot = n*M+j; same formula */
```

**Piping reassembly loop:** same change — `M = num_targets`, loop bounds `N * M`.

### 10. Receiver Reassembly Buffer Allocation

**Location:** main, around line 1984.

```c
/* Old: */
size_t slice_size = DEFAULT_BUFFER_SIZE / num_qps;              /* 1GB/K */
reassembly_buffers_2d = calloc(num_source_gpus * num_qps, ...); /* N×K */
for (int j = 0; j < num_qps; j++) {
    int gpu_j = target_gpu_ids[j];
    /* allocate num_source_gpus buffers of slice_size on GPU j */
}

/* New: */
size_t slice_size = DEFAULT_BUFFER_SIZE / (size_t)M;            /* 1GB/M */
reassembly_buffers_2d = calloc(num_source_gpus * M, ...);       /* N×M */
for (int m = 0; m < M; m++) {
    int gpu_m = target_gpu_ids[m];   /* m-th GPU (requires m < K) */
    /* allocate num_source_gpus buffers of slice_size on GPU m */
    for (int n = 0; n < num_source_gpus; n++) {
        int idx = n * M + m;
        cudaMalloc(&reassembly_buffers_2d[idx], slice_size);
    }
}
```

### 11. Reassembly NVLink Context Allocation

**Location:** main, around line 2015.

```c
/* Old: */
reassembly_nvlink_ctxs = calloc(num_qps * num_qps, ...);  /* K×K */
for (int i = 0; i < num_qps; i++) {
    int qp_gpu = target_gpu_ids[i];
    for (int j = 0; j < num_qps; j++) {                  /* j = target GPU index */
        int slice_gpu = target_gpu_ids[j];
        int ctx_idx = i * num_qps + j;

/* New: */
reassembly_nvlink_ctxs = calloc(num_qps * M, ...);        /* K×M */
for (int k = 0; k < num_qps; k++) {
    int qp_gpu = target_gpu_ids[k];
    for (int m = 0; m < M; m++) {                         /* m = logical target index */
        int slice_gpu = target_gpu_ids[m];
        int ctx_idx = k * M + m;
        /* init NVLink qp_gpu → slice_gpu (or NULL if same) */
```

### 12. Sender NVLink Context Initialization

**Location:** main, around line 2329.

```c
/* Old: N×K×K contexts, thread_idx = n*K*K + i*K + j */
for (int n = 0; n < num_source_gpus; n++) {
    for (int i = 0; i < num_qps; i++) {
        for (int j = 0; j < num_qps; j++) {               /* j = target index (0..K-1) */
            int thread_idx = n * num_qps * num_qps + i * num_qps + j;
            /* NVLink: source_gpu_ids_list[n] → target_gpu_ids[i] */

/* New: N×K×M contexts, thread_idx = n*K*M + i*M + j */
for (int n = 0; n < num_source_gpus; n++) {
    for (int i = 0; i < num_qps; i++) {
        for (int j = 0; j < M; j++) {                     /* j = target index (0..M-1) */
            int thread_idx = n * num_qps * M + i * M + j;
            /* NVLink: source_gpu_ids_list[n] → target_gpu_ids[i] (same GPU pair for all j) */
```

Note: for a given (n, i), all M contexts share the same GPU pair `src[n] → tgt[i]`. This is fine — each thread gets its own `nvlink_context` object even if they point to the same GPU pair (no sharing/mutex needed).

### 13. Piping Parameters

**Location:** around line 1829.

```c
/* Old: */
pipe_N_per_qp  = num_source_gpus * num_qps;                /* N×K */
pipe_slice_size = DEFAULT_BUFFER_SIZE / ((size_t)num_qps * num_qps);  /* 1GB/K² */

/* New: */
pipe_N_per_qp  = num_source_gpus * M;                      /* N×M */
pipe_slice_size = DEFAULT_BUFFER_SIZE / ((size_t)M * num_qps);        /* 1GB/(M×K) */
```

### 14. `setup_thread_args` — Receiver

**Location:** around line 172 (receiver path for `config->is_server && config->reassembly`).

Add `a->num_targets = env->num_targets;` alongside existing field assignments.
The sub_slice_sz passed is `1GB/(M×K)` (see §6 above).

### 15. print_results / Verification

**Location:** `print_results` function and the reassembly verification block (~line 1367).

Update all `num_qps` references that mean "number of targets" to use `M`:
```c
/* Old: */
size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / (num_qps * num_qps);
for (int j = 0; j < num_qps; j++) { ... }

/* New: */
size_t sub_slice_sz = DEFAULT_BUFFER_SIZE / ((size_t)M * num_qps);
for (int m = 0; m < M; m++) { ... }
```

---

---

## Extension: `--target-count` for Direct All-to-All and NVLink No-Reassembly

### Motivation

The original design kept K fixed and made M a separate logical concept. For the simpler modes (direct all-to-all, nvlink no-reassembly) there is no reassembly step and no independent K vs M distinction needed. Instead, `--target-count M` simply **caps the number of NICs (and paired GPUs) used to M**, transporting all data through M NICs instead of K. Buffers scale accordingly so no data is lost.

### Direct All-to-All (`--direct --all-to-all --target-count M`)

**Current behavior (M = K):** N source GPUs × K target NICs → N×K QPs. Each source GPU's 1 GB is split into K slices of `1GB/K`; QP buffer = `1GB/K`.

**With `--target-count M` (M < K):**
- Only the first M NIC/GPU entries from `-n`/`-g` are used.
- Expansion creates N×M QPs (instead of N×K).
- Each source GPU's 1 GB is split into M slices: `slice = 1GB/M` (larger).
- QP buffer = `1GB/M` (larger).
- Sender threads = N×M. Receiver QPs = M (recycled across N sources as before).

```
Source GPU n (n=0..N-1): 1 GB → M slices of 1GB/M each
QP (n*M + m): sends slice m of source n over NIC m
QP buffer on NIC m: receives slices from all N sources
```

**Key formula change in `compute_buffer_sizes`:**
```c
/* Old: */
actual_buffer_size = config.buffer_size / num_nics;   /* 1GB / K */

/* New: */
actual_buffer_size = config.buffer_size / num_targets; /* 1GB / M */
```

**Expansion change (main, direct all-to-all block):**
```c
/* Old: */
int M = num_nics;   /* always K */

/* New: */
int M = num_targets; /* M ≤ K; equals num_nics when --target-count not given */
```

### NVLink No-Reassembly (`--allow-nvlink --target-count M`, no `--reassembly`, no `--all-to-all`)

**Current behavior (M = K):** N source GPUs × K QPs → N×K sender threads. Each source's 1 GB is NVLink-copied as K slices of `1GB/K` to K QP buffers. QP buffer = `N × 1GB/K`.

**With `--target-count M` (M < K):**
- `num_qps` is capped to M before thread/buffer setup (only first M NICs and M GPUs used).
- Sender threads = N×M.
- Each source's 1 GB is split into M slices: `1GB/M` per slice (larger).
- QP buffer = `N × 1GB/M` (larger; formula `N × buffer_size / num_qps` is unchanged, just `num_qps` is now M).
- Receiver QPs = M.

**All data is still transported**: the same total N×1 GB per iteration passes through M NICs instead of K, so each NIC carries more per iteration but no data is dropped.

**Implementation:** cap `num_qps = num_targets` in the NVLink no-reassembly thread-count branch before `num_threads = num_source_gpus * num_qps` is computed.

```c
/* NVLink no-reassembly: cap num_qps to M when --target-count is set */
if (!config.all_to_all && !config.reassembly) {
    num_qps = num_targets;   /* M ≤ K; no-op when --target-count not given */
}
num_threads = num_source_gpus * num_qps;
```

### Validation Changes

Old: `--target-count` with `--direct` was always rejected.

New: accepted for these three combinations:
1. `--direct --all-to-all`
2. `--allow-nvlink` (no `--reassembly`, no `--all-to-all`)
3. `--allow-nvlink --reassembly --all-to-all` (original)

Rejected for everything else (e.g. `--direct` without `--all-to-all`, `--nics-only`).

---

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `src/rdma_multi_qp.h` | Add `num_targets` to `struct rdma_multi_qp_config` |
| `src/test_rdma_multi_qp.c` | Add `--target-count` CLI option; `num_targets` in `thread_args` and `thread_setup_env`; update thread count, sub_slice formula, sender loop bounds, receiver loop bounds, reassembly buffer alloc, reassembly NVLink contexts, sender NVLink contexts, piping params, print_results |

---

## Updated Pipeline: NVLink All-to-All with M ≠ K

### Sender (thread n, k, m) — unchanged structure, just M ≠ K

```
1. start_barrier
2. For each iteration:
   a. NVLink copy: source[n][m*(1GB/M) + k*(1GB/(M×K))]
                 → QP buffer k, slot (n*M+m)×sub_slice
   b. NVLink sync
   c. qp_mutex lock (QP k)
   d. RDMA write with imm (imm = n*M+m)
   e. RDMA poll
   f. qp_mutex unlock
   g. completion_barrier
   h. iteration_barrier
3. end_barrier
```

### Receiver QP thread k — polls N×M instead of N×K

```
1. start_barrier
2. For each iteration:
   a. Post N×M receives (slot nj = n*M+m occupies offset nj*sub_slice in QP buffer)
   b. Poll N×M completions (wr_id = slot index nj; imm = nj)
   c. For each polled slot nj:
      n = nj / M, m = nj % M
      NVLink async: QP buf k[nj*sub_slice] → reassembly_buffers_2d[n*M+m] at k*sub_slice
   d. NVLink sync (once)
   e. reassembly_barrier
   f. completion_barrier
   g. (Main) signal-back
   h. iteration_barrier
3. end_barrier
```

---

## Backward Compatibility

When `--target-count` is absent: `M = K = num_nics`. All formulas reduce to the current behavior:
- sub_slice = 1GB/(M×K) = 1GB/K²  ✓
- N×K×M threads = N×K×K threads  ✓
- N×M reassembly buffers = N×K reassembly buffers  ✓
- K×M NVLink contexts = K×K contexts  ✓

No existing tests are affected.

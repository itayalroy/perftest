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

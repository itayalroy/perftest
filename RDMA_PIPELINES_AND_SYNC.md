# RDMA Test Pipelines and Synchronization

This document describes the data pipelines for sender and receiver in all test modes, together with synchronization points between threads.

---

## Synchronization Primitives

| Primitive | Participants | Purpose |
|-----------|--------------|---------|
| **start_barrier** | Main + all worker threads | All threads ready before timed work begins |
| **end_barrier** | Main + all worker threads | All threads finished; main takes end time |
| **completion_barrier** | Main + all worker threads | Workers signal "iteration work done" |
| **iteration_barrier** | Main + all worker threads | Main releases workers for next iteration (after signal-back in reassembly) |
| **reassembly_barrier** | Main + M receiver QP threads | All receiver threads finished reassembly for this iteration |
| **qp_mutex** | Per QP, shared by threads using that QP | Serialize RDMA operations when multiple threads share a QP |

**When barriers are used:** `needs_barriers = !is_server || (allow-nvlink && reassembly)`

---

## Mode 1: Direct Mode (`--direct`, no all-to-all)

**Threads:** N threads (1 per GPU/NIC pair). Each thread owns one QP.

### Sender Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. RDMA write (from QP buffer, full size)
   b. RDMA poll
   c. completion_barrier
   d. iteration_barrier
3. end_barrier
```

### Receiver Pipeline

**Passive.** Receiver threads run `write_thread` which immediately `sleep(300)`. Data arrives via RDMA one-sided write into pre-posted receive buffers. No active pipeline.

### Synchronization

| Point | Sender | Receiver |
|-------|--------|----------|
| start_barrier | Main + N workers | — |
| completion_barrier | Main + N workers | — |
| iteration_barrier | Main + N workers | — |
| end_barrier | Main + N workers | — |
| qp_mutex | No (1 thread per QP) | No |

---

## Mode 2: Direct All-to-All (`--direct --all-to-all`)

**Threads:** N×M threads (N sources × M targets). Each thread owns one QP. Data: slice of source buffer (no copy; external buffer registration).

### Sender Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. RDMA write (slice from source buffer, part_size)
   b. RDMA poll
   c. completion_barrier
   d. iteration_barrier
3. end_barrier
```

### Receiver Pipeline

**Passive.** Same as Mode 1.

### Synchronization

Same as Mode 1. No qp_mutex (1 thread per QP).

---

## Mode 3: NICs-Only (`--nics-only`)

**Threads:** M threads (1 per NIC). All threads use GPU 0's buffer; each reads a different slice.

### Sender Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. RDMA write (slice from shared buffer, part_size = buffer_size/num_qps)
   b. RDMA poll
   c. completion_barrier
   d. iteration_barrier
3. end_barrier
```

### Receiver Pipeline

**Passive.** Same as Mode 1.

### Synchronization

Same as Mode 1. No qp_mutex.

---

## Mode 4: Allow-NVLink, No Reassembly (`--allow-nvlink --source-gpus ...`)

**Threads:** N×M threads (N sources × M QPs). Single-source: N=1, so M threads (1 per QP). Multi-source: N>1, so N threads share each QP.

**Pipeline is the same** for single- and multi-source. The only difference: when N>1, multiple threads share a QP and must serialize RDMA access via `qp_mutex`.

### Sender Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. NVLink copy (or cudaMemcpy if same GPU): source slice → QP buffer [at offset if multi-source]
   b. NVLink sync (or cudaStreamSynchronize)
   c. [If N>1: qp_mutex lock]
   d. RDMA write
   e. RDMA poll
   f. [If N>1: qp_mutex unlock]
   g. completion_barrier
   h. iteration_barrier
3. end_barrier
```

### Receiver Pipeline

**Passive.** Data lands in QP buffers via RDMA. No reassembly.

### Synchronization

| Point | Sender | Receiver |
|-------|--------|----------|
| start_barrier | Main + N×M workers | — |
| completion_barrier | Main + N×M workers | — |
| iteration_barrier | Main + N×M workers | — |
| end_barrier | Main + N×M workers | — |
| qp_mutex | Only when N>1 (N threads per QP serialize) | No |

---

## Mode 5: Allow-NVLink With Reassembly (`--allow-nvlink --reassembly --source-gpus 0,1,...,7`)

**Threads:** Sender N×M, Receiver M (one per QP). Signal-back: receiver main → sender main (RDMA write with imm).

**Ordering:** Receiver must post receives before sender writes. Sender must post receives for signal-back before main loop.

### Sender Pipeline (per iteration)

```
0. (Main, once before loop) Post receive WQEs for signal-back on QP 0 (one per iteration)
1. start_barrier
2. For each iteration:
   a. NVLink copy: source slice → QP buffer
   b. NVLink sync
   c. qp_mutex lock
   d. RDMA write with immediate (data + source index in one operation)
   e. RDMA poll (wait for send completion)
   f. qp_mutex unlock
   h. completion_barrier
   i. iteration_barrier  ← blocks until main receives signal-back
3. end_barrier
```

### Receiver Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. Post N receives (one per source)  ← must complete before sender writes
   b. Poll all N completions, recording source_gpu_idx from imm for each
   c. For each polled completion (any order):
      - NVLink reassembly async: QP buffer section → reassembly_buffer[source_gpu_idx]
   d. NVLink sync (once, after all N async copies launched)
   e. reassembly_barrier  ← all M receiver threads done
   f. completion_barrier
   g. (Main) rdma_write_with_imm(signal-back) to sender
   h. (Main) rdma_poll_completion (wait for signal-back send completion)
   i. iteration_barrier  ← release workers for next iteration
3. end_barrier
```

**Note:** All N NVLink copies are launched asynchronously and synced once at the end (step d). This allows all N copies to run concurrently on the NVLink fabric instead of serialising them. The sync separates polling all completions (step b) from launching the copies (step c), so all N source sections are guaranteed to be in the QP buffer before any copy starts.

### Main Thread (both sides)

**Sender main (before iteration loop):**
```
Post num_iterations receive WQEs on QP 0 for signal-back
```

**Receiver main (per iteration):**
```
  a. reassembly_barrier  ← wait for M receiver workers
  b. completion_barrier  ← wait for all workers (sender + receiver)
  c. rdma_write_with_imm(signal-back)
  d. rdma_poll_completion (wait for send completion)
  e. iteration_barrier   ← release everyone
```

**Sender main (per iteration):**
```
  a. completion_barrier  ← wait for all workers
  b. rdma_poll_completion_with_imm(signal-back)  ← consume pre-posted receive
  c. iteration_barrier   ← release workers for next iteration
```

### Synchronization

| Point | Sender Workers | Receiver Workers | Sender Main | Receiver Main |
|-------|----------------|------------------|-------------|---------------|
| start_barrier | ✓ | ✓ | ✓ | ✓ |
| reassembly_barrier | — | ✓ | — | ✓ |
| completion_barrier | ✓ | ✓ | ✓ | ✓ |
| iteration_barrier | ✓ | ✓ | ✓ | ✓ |
| end_barrier | ✓ | ✓ | ✓ | ✓ |
| qp_mutex | ✓ (per QP) | No | — | — |

---

## Mode 6: All-to-All NVLink Reassembly (`--allow-nvlink --reassembly --all-to-all`)

**Threads:** Sender N×M×M (one per sub-slice), Receiver M (one per QP). Each sender thread: (source n, QP i, slice j).

### Sender Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. NVLink copy: sub-slice (n,i,j) → QP buffer at slot (n*M+j)
   b. NVLink sync
   c. qp_mutex lock (QP i)
   d. RDMA write with imm (imm = n*M+j)
   e. RDMA poll
   f. qp_mutex unlock
   g. completion_barrier
   h. iteration_barrier
3. end_barrier
```

### Receiver Pipeline (per iteration)

```
1. start_barrier
2. For each iteration:
   a. Post N×M receives per QP (one per (source, slice))
   b. Poll all N×M completions (order non-deterministic), recording (n,j) from imm for each
   c. For each polled completion (any order):
      - NVLink reassembly async: QP buffer slot (n*M+j) → reassembly_buffers_2d[n*M+j] at offset
   d. NVLink sync (once, after all N×M async copies launched)
   e. reassembly_barrier
   f. completion_barrier
   g. (Main) signal-back
   h. iteration_barrier
3. end_barrier
```

**Note:** All N×M NVLink copies are launched asynchronously and synced once at the end (step d), mirroring Mode 5. This allows all copies to run concurrently on the NVLink fabric.

### Synchronization

Same as Mode 5. qp_mutex: one per QP; N×M sender threads per QP serialize.

---

## Summary: Barrier Usage by Mode

| Mode | start | completion | iteration | end | reassembly | qp_mutex |
|------|-------|------------|-----------|-----|------------|----------|
| Direct | Sender | Sender | Sender | Sender | — | No |
| Direct all-to-all | Sender | Sender | Sender | Sender | — | No |
| NICs-only | Sender | Sender | Sender | Sender | — | No |
| NVLink (no reassembly) | Sender | Sender | Sender | Sender | — | Only when N>1 |
| NVLink reassembly | Both | Both | Both | Both | Receiver | Yes |
| All-to-all reassembly | Both | Both | Both | Both | Receiver | Yes |

---

## Piping (`--transport-buffer`)

When `transport_buffer_size / N < slice_size` (single-source: `N=1`; all-to-all: `N×M`), the transfer is split into `num_pipes = ceil(slice_size / section_size)` pipe chunks. See `TRANSPORT_BUFFER_PIPING_DESIGN.md` for full pseudocode.

**Buffer sizing is asymmetric:**
- **Sender:** always uses small transport buffer (`transport_buffer_size`).
- **Receiver without reassembly:** keeps full-size QP buffer so all pipe chunks accumulate at the correct positions. RDMA remote offset advances per pipe: `remote_off = source_gpu_index * pipe_slice_size + pipe * section_size`.
- **Receiver with reassembly:** uses small transport buffer (same as sender); each pipe chunk is immediately NVLink-copied to the reassembly buffer at `pipe * section_size` before the buffer is reused.

**Per-mode behaviour:**
- **Allow-NVLink (no reassembly):** Pipe loop on sender; receiver is fully passive. No per-pipe synchronization. Each pipe writes to a distinct remote offset.
- **Allow-NVLink with reassembly:** Receiver signals sender per pipe via RDMA write-with-imm on each QP (per-QP back-channel). Sender leader polls signal before starting next pipe; non-leaders wait on `pthread_cond` with generation counter predicate (`qp_pipe_ready`). No global barriers per pipe; main thread only participates at start/end barriers.

  **CQ isolation:** Each QP uses separate send and receive CQs (`send_cq` / `recv_cq`). This prevents a race where the leader's `rdma_poll_completion_with_imm` (outside `qp_mutex`) would steal a send completion from another sender thread's concurrent `rdma_poll_completion` (inside `qp_mutex`) on the shared CQ. `rdma_poll_completion` polls `send_cq`; `rdma_poll_completion_with_imm` polls `recv_cq`.
- **All-to-all piping:** Same pattern as multi-source reassembly with `N×M` total sources per QP. Leader designation: `source_gpu_index == 0 && slice_index == 0`.

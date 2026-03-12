# Transport Buffer Piping: Design Document

## Overview

In non-direct modes (NICs-only, single-source NVLink, multi-source NVLink, all-to-all NVLink reassembly), we use two buffer types:

1. **Data buffer** – Source data to be transferred (e.g., 1GB per source)
2. **Transport/QP buffer** – Staging buffer used for NVLink copy and RDMA (currently sized to hold all sources' data for one full timing iteration)

This feature allows configuring a **smaller total transport buffer** per GPU. When the transport buffer is smaller than the total data it must carry per timing iteration, we **pipe** the transfer: send in multiple chunks until the full data is transferred.

**Goal:** Optimize memory usage by reducing transport buffer size while still transferring the full data.

---

## Terminology

| Term | Meaning |
|------|---------|
| **Data buffer** | Source buffer containing the data to transfer (e.g., 1GB) |
| **Slice** | Portion of data for one (source, target) path (e.g., 128MB) |
| **Transport buffer** | Total QP buffer allocated per GPU (RDMA-registered staging area). Shared among all N sources targeting that GPU. (e.g., 32MB total) |
| **Section** | Each source's share of the transport buffer: `section_size = transport_buffer_size / N` |
| **Pipe iteration** | One chunk transfer: copy slice_chunk to transport buffer → NVLink (if needed) → RDMA |
| **Timing iteration** | Our existing iteration (e.g., 10); each timing iteration = full data transfer, possibly via multiple pipe iterations |

---

## Current Behavior (No Piping)

### Single-Source NVLink (1 source, 8 targets)

- Data buffer: 1GB on source GPU
- Slice per path: 1GB/8 = 128MB
- Transport buffer per path: 128MB (same as slice)
- Flow: Copy 128MB source→QP buffer, RDMA 128MB. One transfer per path per timing iteration.

### Multi-Source NVLink (8 sources, 8 targets)

- Data buffer: 1GB per source
- Slice per source per QP: 128MB (1GB / 8 QPs)
- Transport buffer per target QP: 1GB (8 sources × 128MB section each). Each source writes its 128MB into its dedicated section in one go.
- Flow: 8 sources per QP each do one 128MB NVLink copy + RDMA write into their section.

### All-to-All NVLink Reassembly

- Sub-slice per path: 1GB/(M²) (e.g., 1GB/64 ≈ 16MB for M=8)
- Transport = sub-slice size (no piping needed for typical M)

---

## Desired Behavior (With Piping)

### Option: `--transport-buffer SIZE`

- **When not specified:** Current behavior. Transport buffer = full QP buffer required (no piping).
- **When specified:** `SIZE` is the **total QP buffer allocation per GPU**. Each of the N sources sharing that QP gets a section of `section_size = SIZE / N`. If `section_size < slice_size`, piping is active and `num_pipes = ceil(slice_size / section_size)`.
- For single-source (N=1): `section_size = SIZE`, so piping activates when `SIZE < slice_size`.
- For multi-source (N>1): piping activates when `SIZE < N × slice_size` (i.e., when the buffer can't hold all N sources' full slices at once).

### Single-Source Example: `--transport-buffer 32M`

- Slice size: 128MB
- Transport buffer: 32MB
- Pipe iterations per path: ceil(128MB / 32MB) = 4

**Flow per path per timing iteration (with reassembly):**
```
Pipe 0: Copy data[0:32MB] → transport → NVLink → RDMA → receiver NVLink → reassembly[0:32MB]
Pipe 1: Copy data[32MB:64MB] → transport → NVLink → RDMA → receiver NVLink → reassembly[32MB:64MB]
Pipe 2: Copy data[64MB:96MB] → transport → NVLink → RDMA → receiver NVLink → reassembly[64MB:96MB]
Pipe 3: Copy data[96MB:128MB] → transport → NVLink → RDMA → receiver NVLink → reassembly[96MB:128MB]
```

**Important:** The RDMA remote write offset is **always 0** each pipe — the receiver's QP buffer (32MB) is reused per pipe. The offsets 0/32MB/64MB/96MB above represent where each chunk lands in the receiver's **reassembly buffer** after its NVLink copy step, not the RDMA wire offset. For modes without reassembly (passive receiver), the QP buffer is simply overwritten each pipe.

**Memory:** 8 QPs × 32MB = 256MB transport (vs. 8×128MB = 1GB today).

### Multi-Source Example: `--transport-buffer 32M` (N=8 sources, M=8 QPs)

- Total QP buffer per GPU: 32MB
- section_size per source: 32MB / 8 = 4MB
- Slice per source per QP: 128MB (1GB / 8 QPs)
- Pipe iterations per source: ceil(128MB / 4MB) = **32 pipes**

**Flow:** Each source n writes its 4MB section to the QP buffer each pipe, NVLink then RDMA. N sources write to non-overlapping slots (`n * 4MB`) within the 32MB buffer. After all N sections arrive, receiver reassembles and signals. Repeat for 32 pipes to transfer the full 128MB slice per source.

**Memory:** Sender: 8 QPs × 32MB = **256MB** total transport (vs. 8 × 1GB = 8GB today). Receiver: 8 QPs × 32MB = **256MB** (same). Reassembly buffers (full size) are separate and unchanged.

---

## Buffer Size Semantics

### Per GPU (Total QP Buffer)

`--transport-buffer SIZE` sets the **total QP buffer per GPU**. It is the same size on both sender and receiver. Key derived values:

| Derived value | Formula |
|---------------|---------|
| `section_size` | `transport_buffer_size / N` (N = num_sources sharing that QP; 1 for single-source/NICs-only) |
| `num_pipes` | `ceil(slice_size / section_size)` |
| No-piping condition | `section_size >= slice_size` (i.e., `transport_buffer_size >= N × slice_size`) |

Slice size per source per QP varies by mode:

| Mode | N (sources per QP) | slice_size | No-piping threshold |
|------|---------------------|------------|---------------------|
| NICs-only | 1 | buffer_size / num_qps | transport_buffer ≥ slice_size |
| Single-source NVLink | 1 | buffer_size / num_qps | transport_buffer ≥ slice_size |
| Multi-source NVLink | num_source_gpus | buffer_size / num_qps | transport_buffer ≥ N × slice_size |
| All-to-all NVLink reassembly | num_source_gpus × num_qps | 1GB / M² | transport_buffer ≥ N×M × sub_slice |

**Validation:** `transport_buffer_size` must be divisible by N (so `section_size` is an integer). If not, error out. If no-piping condition is met, cap `transport_buffer_size` at `N × slice_size` (no piping, same as today).

---

## Implementation Changes

### 1. Config and CLI

**Add to `struct rdma_multi_qp_config`:**
```c
size_t transport_buffer_size;  /* 0 = use slice size (no piping) */
```

**Add option:** `--transport-buffer SIZE` (bytes; supports 32M, 1G, etc.)

**Validation:**
- Only valid in non-direct modes (NICs-only, allow-nvlink)
- If specified in direct mode: error or ignore
- If larger than slice size: cap at slice size (no piping)

### 2. RDMA Buffer Sizing (Asymmetric)

**With piping, buffer sizes differ between sender and receiver depending on mode:**

| Side | Mode | QP buffer size |
|------|------|----------------|
| Sender | Any NVLink mode | `transport_buffer_size` (small staging buffer) |
| Receiver | **Without reassembly** | **Full size** (unchanged: `N × slice_size`) |
| Receiver | **With reassembly** | `transport_buffer_size` (reused per pipe; data moved to reassembly buffer) |

**Rationale:**
- **Sender (staging):** The sender copies one `section_size` chunk per pipe into the transport buffer, then RDMA-writes it. The small transport buffer is sufficient.
- **Receiver without reassembly:** The receiver is passive; data accumulates across pipes. The receiver must hold the full slice so that `remote_offset = source_n * slice_size + pipe * section_size` places each chunk at the correct position. If the receiver used only `transport_buffer_size`, every pipe would overwrite the same area and only the last chunk would survive.
- **Receiver with reassembly:** The receiver processes each pipe immediately (NVLink-copies to reassembly buffer, then signals). The QP buffer only needs to hold one pipe's worth of data at a time (`transport_buffer_size`), since it is freed before the next pipe arrives.

**Remote write offset (no-reassembly):**
```
remote_off = source_gpu_index * pipe_slice_size + pipe * section_size
```
- `pipe_slice_size` = full per-source-per-QP data size (e.g., 128MB for single-source)
- `pipe * section_size` advances within that source's region per pipe
- For single-source (source_gpu_index=0): `remote_off = 0, section_size, 2*section_size, ...`

**Remote write offset (with reassembly):**
```
remote_off = dst_offset  (same slot reused each pipe; QP buffer is the same small size)
```

**Impact on `rdma_multi_qp`:** Sender passes `buffer_size = transport_buffer_size`; receiver without reassembly passes the full `actual_buffer_size`. Asymmetric sizes are fine for RDMA — the remote write offset must not exceed the remote buffer size, which is guaranteed by the formula above.

### 3. Piping Implementation: Two Modes

Piping differs between **allow-nvlink (no reassembly)** and **allow-nvlink with reassembly**.

#### 3a. Allow-NVLink Without Reassembly (Simple)

**Current sender per iteration:**
1. NVLink transfer + Sync
2. Lock QP
3. RDMA write
4. RDMA poll
5. Unlock QP

**With piping:** After RDMA poll, the transport buffer is free. Start the next NVLink transfer immediately. A single loop works:

```c
for (i = 0; i < args->iterations; i++) {
    for (pipe = 0; pipe < num_pipes; pipe++) {
        NVLink copy chunk[pipe] → transport buffer (at dst_offset, same slot reused)
        Sync
        lock QP
        /* remote_off advances per pipe so each chunk lands at the correct position
         * in the receiver's full-size QP buffer */
        remote_off = source_gpu_index * pipe_slice_size + pipe * section_size;
        RDMA write (local=dst_offset, size=part_size, remote=remote_off)
        RDMA poll
        unlock QP
    }
    pthread_barrier_wait(iteration_barrier);  /* Sync with main for next timing iter */
}
```

**Receiver buffer (no reassembly):** Receiver keeps full-size QP buffer (unchanged from no-piping case). Sender uses small transport buffer. Each pipe lands at a distinct offset in the receiver's buffer, accumulating the full slice across pipes. No per-pipe synchronization needed — receiver is passive.

Single-source example (pipe_slice_size=128MB, section_size=32MB, source_gpu_index=0):
- pipe 0 → remote_off = 0
- pipe 1 → remote_off = 32MB
- pipe 2 → remote_off = 64MB
- pipe 3 → remote_off = 96MB

#### 3b. Allow-NVLink With Reassembly (Complex)

**Current sender per iteration:**
1. NVLink transfer + Sync
2. Lock QP
3. RDMA write
4. RDMA poll
5. RDMA write with immediate (signals receiver: can start reassembly)
6. Unlock QP
7. Main thread receives signal-back (reassembly finished)

**Current receiver:**
1. RDMA poll (write finished)
2. NVLink reassembly + Sync
3. (Main thread) RDMA write with immediate (signal-back to sender)

**Overlap with piping:**
- **After step 5 (RDMA imm):** Next pipe iteration can do NVLink transfer + Sync to transport buffer (overlaps with receiver reassembly).
- **After step 7 (main receives reassembly finished):** Next pipe iteration can do RDMA write (receiver buffer is free).

**Why one for loop is not enough:** The next pipe's NVLink can start after step 5, but the next pipe's RDMA write must wait until step 7. So we have two different "release points" within a pipe iteration.

#### Per-QP Signaling (Design)

**Scope:** Per-QP signaling is used only when piping is enabled (`--transport-buffer` < slice size). Non-piping reassembly keeps the current global barriers and main-thread signal-back.

Each receiver QP signals when it finishes reassembly via RDMA write-with-imm. Only senders targeting that QP wait. No global barriers per pipe; the main thread does not participate in per-pipe sync.

**Benefits:**
- **Per-QP independence:** Sender (n,j) waits only for receiver QP j. Fast QPs (e.g., QP 0) do not block slow QPs (e.g., QP 7).
- **No global barriers per pipe:** No completion_barrier per pipe. Workers self-sync via per-QP RDMA signals. `iteration_barrier` only at the end of each timing iteration (sync with main for next iter).
- **Simple main thread:** Main only does start_barrier and end_barrier for timing; no per-pipe loop.
- **Natural overlap:** NVLink for pipe N+1 can overlap with receiver reassembly for pipe N, because each QP proceeds independently.

**Signal flow:**
- **Receiver QP j:** After reassembly for pipe N → `rdma_write_with_imm` (1 byte, imm=0xDEADBEEF) to sender QP j → `rdma_poll_completion` (wait for send completion).
- **Sender (n,j):** Before RDMA for pipe N+1 → `rdma_post_receive` (pre-post at start of pipe) → `rdma_poll_completion_with_imm` (wait for signal). Pre-posting avoids the race where the receiver sends before the sender has posted.

**Back-channel connection requirement:** This per-QP signal is a receiver→sender RDMA write. In Mode 5 (non-piping), only QP 0 carries signal-back traffic via the main thread. With piping, all M receiver QPs need a back-channel to their corresponding sender QP. This requires the handshake to exchange remote MR addresses for all M QPs in **both** directions (sender→receiver and receiver→sender). See "Files to Modify" for the affected code.

**Pre-posting:** The sender must post the receive for the signal *before* the receiver can send. Therefore: at the start of each pipe iteration, *before* RDMA, the sender posts receive for *this* pipe's signal. When the receiver finishes reassembly and sends, the receive is already posted. We consume it at the start of the next iteration (or after the loop for the last pipe).

---

**Sender pipeline (per timing iteration):**

```c
/* Pre-fetch first chunk before pipe loop */
NVLink copy chunk[0] → transport buffer
Sync

for (pipe = 0; pipe < num_pipes; pipe++) {
    /* Wait for receiver signal: "reassembly of pipe-1 done" (except pipe 0) */
    if (pipe > 0) {
        rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);  /* Consume pre-posted */
    }

    rdma_post_receive(ctx, qp_index, 0, 1, 0);  /* Pre-post for this pipe's signal (before RDMA) */
    lock QP
    RDMA write chunk[pipe] (remote offset = 0)  /* Receiver reuses its chunk_size QP buffer each pipe */
    RDMA poll
    RDMA write with imm  /* Data + source index */
    unlock QP

    /* NVLink for next chunk (overlaps with receiver reassembly of this pipe) */
    if (pipe + 1 < num_pipes) {
        NVLink copy chunk[pipe+1] → transport buffer
        Sync
    }
}

/* Last pipe: wait for receiver signal (receive was pre-posted at start of last iteration) */
rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);

pthread_barrier_wait(iteration_barrier);  /* Sync with main + all workers for next timing iter */
```

**N=1 (single sender per QP):** Use the simple pipeline above. No designated leader or cond vars needed.

**Multi-source (N > 1):** N sender threads share QP j. Only one receive can be posted per QP per pipe (one signal per QP). Per-QP coordination is required; see below.

**Why coordination is needed:** The receiver sends exactly one signal per QP per pipe. But N sender threads use QP j. If each sender posts receive and polls, the first would consume the signal; the others would block forever waiting for a signal that will never arrive. So only one sender must post receive and poll per QP per pipe. The others must wait until that signal is received, then proceed.

**Designated leader:** Sender with `source_gpu_index == 0` (i.e., the first entry in the source GPU list, not necessarily physical GPU 0) is the leader for its QP. Only the leader pre-posts receives and polls for the per-pipe signal. Non-leaders wait on a per-QP condition variable until the leader broadcasts.

**Per-QP state:** `qp_signal_mutex[j]`, `qp_signal_cond[j]`, `qp_pipe_ready[j]` (int, initialized to 0). Create only when `reassembly && piping && num_source_gpus > 1`.

**Why a generation counter is required:** `pthread_cond_broadcast` does not "save" the signal for threads that have not yet called `pthread_cond_wait`. If the leader receives the RDMA signal and broadcasts before a non-leader has reached `cond_wait`, that non-leader will wait forever. The `qp_pipe_ready[j]` counter acts as a predicate: the leader increments it before broadcasting; non-leaders check it in a `while` loop so they never block if the signal already arrived.

**Full sender loop (multi-source, designated leader):**

```c
int is_leader = (args->source_gpu_index == 0);
uint32_t dummy_imm = 0;
/* qp_pipe_ready[qp_index]: shared int per QP, zero-initialized before start_barrier */
/* section_size = transport_buffer_size / num_sources  (each source occupies one slot) */

for (iter = 0; iter < timing_iterations; iter++) {

/* Pre-fetch first chunk before pipe loop */
nvlink_copy(chunk[0] → transport_buffer);
nvlink_sync();

for (pipe = 0; pipe < num_pipes; pipe++) {
    /* Wait for receiver signal: "reassembly of pipe-1 done" (except pipe 0) */
    if (pipe > 0) {
        if (is_leader) {
            rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);  /* Consume pre-posted */
            pthread_mutex_lock(&qp_signal_mutex[qp_index]);
            qp_pipe_ready[qp_index]++;  /* Predicate: increment before broadcast */
            pthread_cond_broadcast(&qp_signal_cond[qp_index]);
            pthread_mutex_unlock(&qp_signal_mutex[qp_index]);
        } else {
            pthread_mutex_lock(&qp_signal_mutex[qp_index]);
            while (qp_pipe_ready[qp_index] < pipe)  /* Check predicate to avoid lost wakeup */
                pthread_cond_wait(&qp_signal_cond[qp_index], &qp_signal_mutex[qp_index]);
            pthread_mutex_unlock(&qp_signal_mutex[qp_index]);
        }
    }

    if (is_leader) {
        rdma_post_receive(ctx, qp_index, 0, 1, 0);  /* Pre-post for this pipe's signal (before RDMA) */
    }
    pthread_mutex_lock(&qp_mutex[qp_index]);
    rdma_write_with_imm(ctx, qp_index, ...);  /* Data + source index; remote offset = source_gpu_index * section_size */
    rdma_poll_completion(ctx, qp_index, -1);
    pthread_mutex_unlock(&qp_mutex[qp_index]);

    /* NVLink for next chunk (overlaps with receiver reassembly of this pipe) */
    if (pipe + 1 < num_pipes) {
        nvlink_copy(chunk[pipe + 1] → transport_buffer);
        nvlink_sync();
    }
}

/* Last pipe: wait for receiver signal (receive was pre-posted at start of last pipe) */
if (is_leader) {
    rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);
    pthread_mutex_lock(&qp_signal_mutex[qp_index]);
    qp_pipe_ready[qp_index]++;
    pthread_cond_broadcast(&qp_signal_cond[qp_index]);
    pthread_mutex_unlock(&qp_signal_mutex[qp_index]);
} else {
    pthread_mutex_lock(&qp_signal_mutex[qp_index]);
    while (qp_pipe_ready[qp_index] < num_pipes)
        pthread_cond_wait(&qp_signal_cond[qp_index], &qp_signal_mutex[qp_index]);
    pthread_mutex_unlock(&qp_signal_mutex[qp_index]);
}

pthread_barrier_wait(iteration_barrier);  /* Sync with main + all workers; once per timing iter */

/* Reset predicate for next iteration (leader only). Happens after the barrier that already
   fenced all threads, so all writes from this iteration are visible. Non-leaders don't
   check qp_pipe_ready until pipe > 0 of the next iteration, which requires NVLink + RDMA
   to complete first — plenty of time for the leader's reset to propagate via the mutex. */
if (is_leader) {
    pthread_mutex_lock(&qp_signal_mutex[qp_index]);
    qp_pipe_ready[qp_index] = 0;
    pthread_mutex_unlock(&qp_signal_mutex[qp_index]);
}

} /* end timing iteration loop */
```

**Ordering:** RDMA writes within a pipe are serialized through `qp_mutex`, so the receiver will not send the pipe-N signal until all N senders have completed their RDMA writes. The leader then polls and gets the signal. Non-leaders use `while (qp_pipe_ready < pipe)` as the predicate so they never block if the leader already incremented and broadcast before they reached `cond_wait`.

**`qp_pipe_ready` reset:** The leader resets to 0 immediately **after** the end-of-iteration `iteration_barrier`. The barrier already fences all prior writes. Non-leaders don't check `qp_pipe_ready` until `pipe > 0` of the next iteration, which requires completing at least one NVLink copy + one RDMA write first — sufficient time for the leader's reset (a single mutex lock/set/unlock) to complete and become visible via the mutex acquire in the non-leader's predicate check. No extra barrier is needed and adding one would deadlock (main only hits `iteration_barrier` once per timing iteration).

**Note on predicate value:** `qp_pipe_ready` counts completed pipes (1 after pipe 0's signal, 2 after pipe 1's signal, etc.). Non-leaders check `< pipe` (for pipes 1..num_pipes-1) and `< num_pipes` for the final wait. This correctly handles the case where the leader is faster than all non-leaders.

**Remote offset layout (multi-source):** Each sender n writes to remote offset `n * section_size` in the receiver's QP buffer. The receiver's QP buffer has N non-overlapping slots (total size = `chunk_size = N * section_size`). The same slots are reused each pipe iteration (buffer reuse). The receiver reads from `source_gpu_idx * section_size` (obtained from the imm value) and NVLink-copies to the correct offset in the reassembly buffer.

---

**Receiver pipeline (per timing iteration):**

```c
for (pipe = 0; pipe < num_pipes; pipe++) {
    /* Post receives for this pipe (one per source). QP buffer reused per pipe. */
    section_size = chunk_size / num_sources;  /* Each source sends section_size per pipe */
    for (src = 0; src < num_sources; src++) {
        offset = src * section_size;  /* Reuse same offsets each pipe; buffer size = chunk_size */
        rdma_post_receive(ctx, qp_index, offset, section_size, wr_id_with_src);
    }

    /* Poll and reassemble */
    for (src = 0; src < num_sources; src++) {
        rdma_poll_completion_with_imm(..., &source_gpu_idx, ...);
        src_offset = source_gpu_idx * section_size;
        nvlink_copy(QP_buffer[src_offset] → reassembly_buffer[source_gpu_idx] + pipe * section_size);
        nvlink_sync();
    }

    /* Signal sender: "reassembly for this pipe done" */
    rdma_write_with_imm(ctx, qp_index, 0, 1, 0, 0xDEADBEEF);
    rdma_poll_completion(ctx, qp_index, -1);  /* Wait for send completion */
}

pthread_barrier_wait(iteration_barrier);  /* Sync with main + all workers for next timing iter */
```

**Buffer layout (receiver with reassembly):** QP buffer size = `transport_buffer_size` (small, same as sender). Reused per pipe. For pipe P, source S: receive offset = `S * section_size` (same offsets each pipe). Reassembly buffer for source S: chunks accumulated at offsets `0, section_size, 2*section_size, ...` across pipes.

---

**Main thread:**

```c
pthread_barrier_wait(start_barrier);
for (iter = 0; iter < timing_iterations; iter++) {
    pthread_barrier_wait(iteration_barrier);  /* Wait for all workers to finish all pipes */
    /* Optional: take timing, print bandwidth */
}
pthread_barrier_wait(end_barrier);
```

Main participates in `start_barrier`, `iteration_barrier` (once per timing iter), and `end_barrier`. No per-pipe coordination; workers self-sync via per-QP RDMA signals within a timing iteration.

**iteration_barrier count (non-all-to-all):** Sender: `1 (main) + N×M (workers)`. Receiver: `1 (main) + M (workers)`.

---

**Ordering and overlap:**
- Pipe 0: Sender does NVLink(chunk 0) → RDMA(chunk 0). Receiver polls, reassembles, signals. Sender can start NVLink(chunk 1) as soon as RDMA imm is done (overlaps with receiver reassembly).
- Pipe 1: Sender waits for signal (receiver done with pipe 0) → RDMA(chunk 1). Receiver polls, reassembles, signals. Sender starts NVLink(chunk 2).
- This continues; each QP proceeds at its own pace.

### 4. All-to-All Reassembly With Piping

**When:** `--allow-nvlink --reassembly --all-to-all --transport-buffer SIZE` and `transport_buffer < sub_slice` (sub_slice = 1GB/M²).

**Structure:** N×M×M sender threads (source n, QP i, slice j). M receiver threads (one per QP). Same per-QP signaling pattern: receiver QP i signals after reassembling each pipe; senders for QP i wait. Designated leader per QP (N×M senders share each QP).

- QP buffer size: `transport_buffer_size` (total per GPU)
- `section_size = transport_buffer_size / (N×M)` — one slot per (n,j) pair
- `num_pipes = ceil(sub_slice / section_size)` where `sub_slice = 1GB / M²`
- Offset for slot (n,j): `(n*M+j) * section_size` within the QP buffer

Receiver posts N×M receives per pipe (one per (n,j)), polls, NVLink-copies each `section_size` chunk to `reassembly_buffers_2d[n*M+j]` at offset `pipe * section_size`, then signals once per pipe.

**Implementation order:** Implement multi-source piping first; add all-to-all piping if needed.

### 5. Receiver Thread Changes

**Current (no piping):** Post one receive per (source, slice); receive full chunk. One reassembly_barrier, completion_barrier, iteration_barrier per iteration.

**With piping (per-QP signaling):** Post multiple receives per pipe, each for `section_size`. The receiver must know `chunk_size`, `num_pipes`, and `num_sources`. For each pipe: post receives → poll and reassemble → `rdma_write_with_imm` (signal sender) → `rdma_poll_completion` (send completion). No barriers per pipe; each receiver QP signals its sender directly via RDMA.

### 6. Thread Args

**Add to `struct thread_args`:**
```c
size_t transport_buffer_size;  /* Total QP buffer per GPU; 0 = no piping */
size_t section_size;           /* transport_buffer_size / num_sources; bytes per source per pipe */
int    num_pipe_iterations;    /* ceil(slice_size / section_size) */
```

`section_size` and `chunk_size` in the pipeline pseudocode refer to the same value. `num_pipe_iterations` = 1 when piping is inactive (section_size >= slice_size).

### 7. Modes Summary

| Mode | N (sources/QP) | slice_size | section_size (32MB QP / N) | num_pipes |
|------|----------------|------------|----------------------------|-----------|
| NICs-only | 1 | 128MB | 32MB | 4 |
| Single-source NVLink | 1 | 128MB | 32MB | 4 |
| Multi-source NVLink (N=8) | 8 | 128MB | 4MB | 32 |
| All-to-all NVLink (N=8, M=8) | 64 | 16MB | 0.5MB | 32 |

For All-to-all: `section_size = 32MB / (N×M) = 32MB / 64 = 0.5MB`; `num_pipes = 16MB / 0.5MB = 32`. Piping is only active if `section_size < slice_size`.

---

## Examples

### Example 1: Single-Source, 32MB Transport

```bash
# Server
./test_rdma_multi_qp -s -n mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 -g 0,1,2,3,4,5,6,7 \
  --allow-nvlink --source-gpus 0 --transport-buffer 32M

# Client
./test_rdma_multi_qp -n mlx5_0,...,mlx5_7 -g 0,...,7 --allow-nvlink --source-gpus 0 -a <server_ip> --transport-buffer 32M
```

**Result:** 1GB data, 8 QPs, 1 source per QP (N=1). section_size = 32MB / 1 = 32MB. 128MB slice / 32MB section = **4 pipe iterations**. Transport: 8×32MB = **256MB** total (vs. 8×128MB = 1GB today).

### Example 2: 8 Sources, 32MB Transport

```bash
# Server
./test_rdma_multi_qp -s -n mlx5_0,...,mlx5_7 -g 0,...,7 --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
  --transport-buffer 32M

# Client
./test_rdma_multi_qp -n mlx5_0,...,mlx5_7 -g 0,...,7 --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
  -a <server_ip> --transport-buffer 32M
```

**Result:** 8GB data total (8 source GPUs × 1GB each). 8 QPs, 8 sources per QP. QP buffer: 32MB total (4MB section per source). Each source: 128MB slice / 4MB section = **32 pipe iterations**. Sender transport: 8×32MB = **256MB** (vs. 8×1GB = 8GB today). Receiver: same. Reassembly buffers (full size, unchanged) hold accumulated data across pipes.

### Example 3: No Piping (Default)

```bash
./test_rdma_multi_qp ... --allow-nvlink --source-gpus 0 -a <server_ip>
# No --transport-buffer: transport = slice size (128MB), no piping
```

---

## Bandwidth Calculation

**Unchanged:** Total data = sum of all slice sizes across all paths. Time = wall-clock for all timing iterations. Bandwidth = total_data / time.

The pipe iterations are internal to each timing iteration; we still measure total data transferred and total time.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/test_rdma_multi_qp.c` | Add `--transport-buffer`; `transport_buffer_size` in config; `section_size`, `num_pipe_iterations`, `pipe_slice_size` in thread args; pipe loop in `write_thread`; asymmetric buffer sizing (sender small, receiver full for no-reassembly); per-QP `qp_pipe_ready`/`qp_signal_mutex`/`qp_signal_cond` for reassembly piping (N>1) |
| `src/rdma_multi_qp.c` | **Split send/receive CQ per QP** to eliminate a CQ race (see "Resolved Design Questions" #2). Handshake already exchanges MR addresses bidirectionally, so per-QP back-channel signals work out of the box. |
| `multi_nic_send_buffer` | Added `--transport-buffer` passthrough (done). |
| `multi_nic_receive_buffer` | Added `--transport-buffer` passthrough (done). Note: receiver without reassembly allocates full-size QP buffer regardless of `--transport-buffer`; the flag only affects sender staging and reassembly receiver sizing. |

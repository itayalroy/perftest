# Double-Buffer Transport Mode: Design Document

## Overview

The **double-buffer** mode is an extension to the transport-buffer piping design
(see `TRANSPORT_BUFFER_PIPING_DESIGN.md`). Instead of a single transport buffer per QP
that is reused across pipe iterations sequentially, this mode allocates **two transport
buffers per QP** and alternates between them. This unlocks two distinct pipeline
overlaps depending on the mode.

**Applies to:** all non-direct modes (NICs-only, allow-nvlink single-source,
allow-nvlink multi-source, allow-nvlink with reassembly, all-to-all with reassembly).

**Does not apply to:** `--direct` and `--direct-all-to-all` (these modes have no
transport buffer; data is sent directly from the data buffer).

---

## Motivation

### Without reassembly

In single-buffer piping, each pipe iteration is serial:

```
NVLink fill buf → sync → RDMA send buf → next pipe
```

NVLink and RDMA never run concurrently. With double-buffer, the NVLink fill of the
idle buffer overlaps with the RDMA send of the active buffer:

```
NVLink fill buf[1]  ──────────────────┐  (async)
RDMA send   buf[0]  ──────────────────┘  (concurrent)
```

Per-pipe latency ≈ max(T_nvlink, T_rdma) instead of T_nvlink + T_rdma.

### With reassembly (larger benefit)

In single-buffer piping with reassembly, the sender is fully stalled after each pipe
waiting for the receiver to finish reassembly before it can reuse the buffer:

```
Sender:   NVLink[0] → RDMA[0] → wait for reassembly signal → NVLink[1] → RDMA[1] → ...
Receiver:                        reassemble[0] → signal
```

The sender is idle for the entire duration of the receiver's NVLink reassembly.

With double-buffer, the receiver has two QP buffers. The sender sends pipe p+1 to
`remote_buf[1]` immediately after sending pipe p to `remote_buf[0]`, without waiting
for any signal. The signal for `remote_buf[0]` only matters at pipe p+2 when the
sender needs to reuse it:

```
Sender:   NVLink[0] → RDMA[0] → NVLink[1] → RDMA[1] → wait → NVLink[2] → RDMA[2] → ...
Receiver:             reassemble[0] ──────────────────────────┘
                                     reassemble[1] ─────────────────────────────────...
```

The sender always has 2 pipes in flight: one being reassembled on the receiver while
the next is already being sent. Receiver reassembly latency is hidden behind the
sender's next RDMA send. This is the larger benefit of double-buffer and applies
even when T_nvlink is negligible.

---

## Terminology

Inherits all terms from `TRANSPORT_BUFFER_PIPING_DESIGN.md`. New terms:

| Term | Meaning |
|------|---------|
| `buf[0]`, `buf[1]` | Two local transport buffers per QP on the sender (each = `transport_buffer_size`) |
| `remote_buf[0]`, `remote_buf[1]` | Two remote QP buffers on the receiver (reassembly mode only; each = `transport_buffer_size`) |
| `cur` | `pipe % 2` — index of the active buffer (being RDMA-sent this pipe) |
| `nxt` | `1 - cur` — index of the idle buffer (being NVLink-filled for next pipe) |

---

## CLI Option

```
--double-buffer
```

Boolean flag. Requires `--transport-buffer SIZE` to be specified (without piping,
there is nothing to double-buffer).

**Validation:** Error if `--double-buffer` is used without `--transport-buffer`, or
if used with `--direct` / `--direct-all-to-all`.

---

## Memory Impact

Each sender QP now holds **two** transport buffers:

| Buffer | Size |
|--------|------|
| `buf[0]` (buf_A) | `transport_buffer_size` |
| `buf[1]` (buf_B) | `transport_buffer_size` |
| Total sender transport per QP | `2 × transport_buffer_size` |

**Receiver without reassembly:** unchanged. Still a single full-size QP buffer.
Each pipe writes to a distinct remote offset in it (pipes are not concurrent
on the receiver side).

**Receiver with reassembly:** also has two QP buffers of size `transport_buffer_size`.
Sender writes to `remote_buf[pipe % 2]` each pipe. See "Receiver Changes" section.

**Reassembly buffers (full size):** unchanged on both sides.

### Memory Comparison (single-source, 8 QPs, 32 MB transport buffer)

| Config | Sender transport | Receiver QP buffer |
|--------|------------------|--------------------|
| No transport buf (current) | 8 × 128 MB = 1 GB | 8 × 128 MB = 1 GB |
| `--transport-buffer 32M` | 8 × 32 MB = 256 MB | no-reassembly: 8 × 128 MB = 1 GB |
| `--transport-buffer 32M --double-buffer` | 8 × 64 MB = 512 MB | no-reassembly: 8 × 128 MB = 1 GB |
| `--transport-buffer 32M --double-buffer` (reassembly) | 8 × 64 MB = 512 MB | 8 × 64 MB = 512 MB |

---

## Pipeline: Allow-NVLink Without Reassembly (N=1, passive receiver)

The simplest case. Receiver is passive with a single full-size QP buffer (unchanged).
Sender alternates between `buf[0]` and `buf[1]`.

```c
/* Pre-load first chunk into buf[0] */
nvlink_copy(chunk[0] → buf[0]);
nvlink_sync();

for (pipe = 0; pipe < num_pipes; pipe++) {
    int cur = pipe % 2;
    int nxt = 1 - cur;

    /* Start async NVLink fill of the idle buffer for the NEXT pipe.
     * Runs concurrently with the RDMA send below. */
    if (pipe + 1 < num_pipes) {
        nvlink_copy(chunk[pipe + 1] → buf[nxt]);   /* async, no sync yet */
    }

    /* RDMA-send the active buffer (overlaps with NVLink above) */
    lock QP;
    remote_off = source_gpu_index * pipe_slice_size + pipe * section_size;
    rdma_write(buf[cur], section_size, remote_off);
    rdma_poll_completion();
    unlock QP;

    /* Ensure idle buffer fill is done before it becomes the active buffer next iteration */
    if (pipe + 1 < num_pipes) {
        nvlink_sync();
    }
}
pthread_barrier_wait(iteration_barrier);
```

**Key change vs. single-buffer:** `nvlink_copy(next)` is launched *before* the RDMA
send (not after), so NVLink and RDMA execute concurrently. `nvlink_sync()` after
`rdma_poll_completion` ensures the idle buffer is ready before the next iteration
flips roles.

**Multi-source (N > 1, no reassembly):** Each source thread gets its own independent
`buf[0]`/`buf[1]` pair, each of size `section_size`. There is no shared layout or
offset arithmetic within the buffer — each thread's `buf[x]` holds only its own chunk.
The `qp_mutex` still serializes RDMA writes from N threads on the same QP. The key
overlap is that while one source holds `qp_mutex` for RDMA, all other sources can run
their NVLink copy into their idle buffer concurrently — NVLink does not require the mutex.

**Buffer allocation:** At setup, allocate one contiguous block of
`N * 2 * section_size` per QP and slice it: thread n gets
`buf[0] = block + n * 2 * section_size` and `buf[1] = buf[0] + section_size`.
Register the whole block as one MR. The remote layout on the receiver is unchanged
(`remote_off = source_gpu_idx * pipe_slice_size + pipe * section_size`).

Per-thread pseudocode (thread for source n, QP j):

```c
/* buf[0] and buf[1] are each section_size, fully private to this thread */

/* Pre-load chunk[0] into buf[0] */
nvlink_copy(data_slice[0] → buf[0], section_size);
nvlink_sync();

for (pipe = 0; pipe < num_pipes; pipe++) {
    int cur = pipe % 2;
    int nxt = 1 - cur;

    /* Start NVLink for next chunk into idle buffer (async, no mutex needed) */
    if (pipe + 1 < num_pipes) {
        nvlink_copy(data_slice[pipe + 1] → buf[nxt], section_size);
    }

    /* Serialize RDMA with the other N-1 sources on this QP */
    pthread_mutex_lock(&qp_mutex[j]);
    remote_off = source_gpu_idx * pipe_slice_size + pipe * section_size;
    rdma_write(buf[cur], section_size, remote_off);
    rdma_poll_completion();
    pthread_mutex_unlock(&qp_mutex[j]);

    /* Ensure NVLink for next chunk is complete before using buf[nxt] next iteration */
    if (pipe + 1 < num_pipes) {
        nvlink_sync();
    }
}
pthread_barrier_wait(iteration_barrier);
```

**Example:** N=2 sources (GPU 0, GPU 1), M=2 QPs, `--transport-buffer 32M`.

- `section_size = 32M / N = 32M / 2 = 16 MB` (each source's share of the 32MB QP budget)
- Each source thread: private `buf[0] = 16 MB`, `buf[1] = 16 MB`
- Total sender transport per QP: `2 × 32 MB = 64 MB` (2 buffers of 32 MB each)
- `slice = 1 GB / M = 512 MB` per source per QP
- `num_pipes = 512 MB / 16 MB = 32`

Approximate timeline for pipe p on QP j (time flows right):

```
Source 0: NVLink[p+1] → buf[nxt] ─────────┐  (async, no mutex)
Source 1: NVLink[p+1] → buf[nxt] ─────────┐  (async, no mutex)
Source 1: qp_mutex ── RDMA buf[cur] ── poll ── unlock
Source 0:             qp_mutex ── RDMA buf[cur] ── poll ── unlock
Source 0: nvlink_sync  (likely already done)
Source 1: nvlink_sync  (likely already done)
```

While source 1 holds the mutex doing RDMA, source 0's NVLink is already in flight.
By the time source 0 acquires the mutex, its NVLink copy for pipe p+1 is done (or
nearly so). No additional synchronization primitives are needed.

---

## Pipeline: Allow-NVLink With Reassembly (N=1, single-source)

The receiver also has two QP buffers (`remote_buf[0]`, `remote_buf[1]`). The sender
writes to `remote_buf[pipe % 2]` each pipe. Before reusing `remote_buf[cur]` (which
happens 2 pipes later), the sender must wait for the receiver's signal confirming that
buffer is processed.

The overlap gained: while the receiver reassembles `remote_buf[cur]` (pipe p), the
sender can already be NVLink-filling its `buf[nxt]` for pipe p+1 AND the receiver
can already be accepting data for pipe p+1 into `remote_buf[nxt]`. The pipeline is
always 2 pipes deep.

### Sender (single-source)

```c
/* Pre-post receive WQEs for the first "buffer done" signals from receiver.
 * Post min(2, num_pipes): if num_pipes == 1, only one signal ever arrives. */
rdma_post_receive(ctx, qp_index, 0, 1, 0);   /* for remote_buf[0] signal (pipe 0) */
if (num_pipes >= 2) {
    rdma_post_receive(ctx, qp_index, 0, 1, 0);   /* for remote_buf[1] signal (pipe 1) */
}

/* Pre-load first chunk */
nvlink_copy(chunk[0] → buf[0]);
nvlink_sync();

for (pipe = 0; pipe < num_pipes; pipe++) {
    int cur = pipe % 2;
    int nxt = 1 - cur;

    /* For pipe >= 2: wait for receiver signal "remote_buf[cur] is processed"
     * (i.e., the signal sent after pipe p-2 was reassembled).
     * This confirms remote_buf[cur] is free for the sender to write again. */
    if (pipe >= 2) {
        rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);

        /* Pre-post a receive for the future signal from pipe+2 (if it will exist) */
        if (pipe + 2 < num_pipes) {
            rdma_post_receive(ctx, qp_index, 0, 1, 0);
        }
    }

    /* Start async NVLink fill of idle buffer for next pipe */
    if (pipe + 1 < num_pipes) {
        nvlink_copy(chunk[pipe + 1] → buf[nxt]);   /* async */
    }

    /* RDMA-send active buffer to remote_buf[cur] on receiver */
    lock QP;
    remote_off = cur * transport_buffer_size;   /* remote_buf[0] or remote_buf[1] */
    rdma_write_with_imm(buf[cur], section_size, remote_off, source_gpu_idx);
    rdma_poll_completion();
    unlock QP;

    /* Wait for NVLink to finish before next iteration uses buf[nxt] */
    if (pipe + 1 < num_pipes) {
        nvlink_sync();
    }
}

/* Drain remaining signals for the last min(2, num_pipes) pipes */
rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);   /* pipe num_pipes-1 */
if (num_pipes >= 2) {
    rdma_poll_completion_with_imm(ctx, qp_index, -1, &dummy_imm, NULL);   /* pipe num_pipes-2 */
}

pthread_barrier_wait(iteration_barrier);
```

**Note on "drain" at end:** The receiver sends one signal per pipe (total = `num_pipes`
signals). The loop consumes `max(0, num_pipes - 2)` of them. The drain calls after the
loop consume the remaining `min(2, num_pipes)` signals. The `num_pipes >= 2` guard
ensures we never poll for a signal that was never sent.

### Receiver (per QP, with double-buffer + reassembly)

The receiver pre-posts receives for two pipes at a time. While reassembling data from
`remote_buf[cur]`, it is already accepting the next pipe into `remote_buf[nxt]`.

```c
/* Pre-post receives for pipe 0 AND pipe 1 before the loop */
for (src = 0; src < num_sources; src++) {
    offset = 0 * transport_buffer_size + src * section_size;   /* remote_buf[0] slots */
    rdma_post_receive(ctx, qp_index, offset, section_size, wr_id_with_src);
}
for (src = 0; src < num_sources; src++) {
    offset = 1 * transport_buffer_size + src * section_size;   /* remote_buf[1] slots */
    rdma_post_receive(ctx, qp_index, offset, section_size, wr_id_with_src);
}

for (pipe = 0; pipe < num_pipes; pipe++) {
    int cur = pipe % 2;

    /* Poll completions for this pipe's data (arrives in remote_buf[cur]) */
    for (src = 0; src < num_sources; src++) {
        rdma_poll_completion_with_imm(ctx, qp_index, -1, &source_gpu_idx, NULL);
        src_offset = cur * transport_buffer_size + source_gpu_idx * section_size;
        reassembly_dst = reassembly_buffer[source_gpu_idx] + pipe * section_size;
        nvlink_copy_async(remote_buf[cur][source_gpu_idx] → reassembly_dst);
    }
    nvlink_sync();   /* Ensures remote_buf[cur] data is in reassembly buffer */

    /* Signal sender: "remote_buf[cur] is processed; you can reuse it for pipe+2" */
    rdma_write_with_imm(ctx, qp_index, 0, 1, 0, 0xDEADBEEF);
    rdma_poll_completion(ctx, qp_index, -1);

    /* Pre-post receives for pipe+2 into the now-free remote_buf[cur] */
    if (pipe + 2 < num_pipes) {
        for (src = 0; src < num_sources; src++) {
            offset = cur * transport_buffer_size + src * section_size;
            rdma_post_receive(ctx, qp_index, offset, section_size, wr_id_with_src);
        }
    }
}

pthread_barrier_wait(iteration_barrier);
```

**Overlap achieved on receiver:** While the receiver is NVLink-copying from
`remote_buf[cur]` (for pipe p), it already has receives posted for `remote_buf[nxt]`
(pipe p+1). Data for pipe p+1 can arrive and land in `remote_buf[nxt]` without
waiting for pipe p's reassembly to complete. Both buffers are independent.

---

## Multi-Source (N > 1) With Reassembly

N sender threads share each QP. Same `qp_pipe_ready` / `qp_signal_mutex` /
`qp_signal_cond` per-QP coordination as in single-buffer piping (see
`TRANSPORT_BUFFER_PIPING_DESIGN.md`), with one change: the leader must consume
the signal before reusing `remote_buf[cur]`, which now happens every 2 pipes
instead of every pipe.

The designated leader:
- Pre-posts 2 receives (for `remote_buf[0]` and `remote_buf[1]` signals) before
  the pipe loop
- At pipe >= 2: polls the signal, increments `qp_pipe_ready[j]`, broadcasts,
  pre-posts for pipe+2 (if applicable)
- At end: drains 2 signals

Non-leaders: `while (qp_pipe_ready[j] < pipe - 1)` before RDMA (same predicate
pattern, threshold adjusted for 2-pipe window).

The receiver is unchanged relative to the single-source description above; it
always signals once per pipe regardless of N.

---

## Receiver Buffer Layout (With Reassembly)

The receiver's QP buffer now holds **two** transport-buffer-sized regions:

```
remote_buf[0]:  [src0_slot | src1_slot | ... | srcN_slot]   (size = transport_buffer_size)
remote_buf[1]:  [src0_slot | src1_slot | ... | srcN_slot]   (size = transport_buffer_size)
Total QP buffer: 2 × transport_buffer_size
```

Remote write offsets from sender:
```
remote_off_for_pipe_p = (pipe % 2) * transport_buffer_size + source_gpu_idx * section_size
```

This is the only offset formula change vs. single-buffer piping (where remote_off = 0
always for the reassembly case).

---

## Overlap Summary

| Phase | Single-buffer piping | Double-buffer piping |
|-------|----------------------|-----------------------|
| NVLink fill + RDMA send | Sequential | **Concurrent** |
| NVLink fill + receiver reassembly | Concurrent (already) | Concurrent (still) |
| Pipeline depth | 1 pipe in-flight | **2 pipes in-flight** |
| Wait before reusing transport buf | Every pipe | Every **2** pipes |

---

## Signal Protocol Changes

| Aspect | Single-buffer | Double-buffer |
|--------|---------------|---------------|
| Receiver signals per timing iteration | `num_pipes` | `num_pipes` (unchanged) |
| Sender pre-posts (total per timing iter) | `num_pipes` | `num_pipes` (unchanged) |
| Pre-posts before pipe loop | 1 | **2** |
| Pre-posts inside loop | `num_pipes - 1` | `num_pipes - 2` |
| Drain after loop | 0 (loop handles last) | **2** |
| First pipe that requires waiting for signal | pipe 1 | **pipe 2** |
| Remote offset formula (reassembly) | `source_gpu_idx * section_size` | `(pipe%2)*transport_buffer_size + source_gpu_idx*section_size` |

---

## Interaction With Existing Piping Infrastructure

- `section_size`, `num_pipes`, `pipe_slice_size`: **unchanged**. Double-buffer
  does not change the chunk size or pipe count; it only changes the number of
  local/remote staging buffers and their access pattern.
- `qp_pipe_ready` / `qp_signal_mutex` / `qp_signal_cond`: still used for N>1;
  the signal semantics are unchanged (one signal per pipe per QP from receiver).
- CQ isolation (`send_cq` / `recv_cq`): unchanged and still required.
- `iteration_barrier`: still called once per timing iteration (not per pipe).
- Receiver without reassembly: **no changes**. The full-size single QP buffer
  is still used; the sender just double-buffers locally.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/test_rdma_multi_qp.c` | Add `--double-buffer` flag; allocate `buf[0]` and `buf[1]` (2× local transport buffer per QP per source); allocate 2 remote QP buffers per QP for reassembly receivers; update pipe loop to interleave NVLink and RDMA (no-reassembly); update pipe loop + pre-post/drain pattern (reassembly); update remote offset formula for reassembly; pre-post 2 receives before loop instead of 1; drain 2 signals after loop. |
| `src/rdma_multi_qp.c` | No structural changes required. Remote buffer registration and MR exchange already handle arbitrary buffer sizes. The 2× QP buffer for reassembly receivers is transparent to the RDMA layer (it is a single larger MR). |
| `multi_nic_send_buffer` | Add `--double-buffer` passthrough. |
| `multi_nic_receive_buffer` | Add `--double-buffer` passthrough. |

---

## Example Configurations

### Example 1: Single-source, double-buffer, no reassembly

```bash
# Receiver (passive)
./test_rdma_multi_qp -s -n mlx5_0,...,mlx5_7 -g 0,...,7 \
  --allow-nvlink --source-gpus 0 --transport-buffer 32M --double-buffer

# Sender
./test_rdma_multi_qp -n mlx5_0,...,mlx5_7 -g 0,...,7 \
  --allow-nvlink --source-gpus 0 -a <server_ip> --transport-buffer 32M --double-buffer
```

- section_size = 32 MB, num_pipes = 4 (128 MB / 32 MB)
- Sender transport: 8 QPs × 2 × 32 MB = **512 MB** (vs. 256 MB single-buffer)
- Receiver QP buffer: 8 × 128 MB = 1 GB (unchanged)
- Each pipe: NVLink(32MB) and RDMA(32MB) overlap

### Example 2: 8 sources, double-buffer, with reassembly

```bash
# Receiver
./test_rdma_multi_qp -s -n mlx5_0,...,mlx5_7 -g 0,...,7 \
  --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
  --transport-buffer 32M --double-buffer

# Sender
./test_rdma_multi_qp -n mlx5_0,...,mlx5_7 -g 0,...,7 \
  --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
  -a <server_ip> --transport-buffer 32M --double-buffer
```

- section_size = 32 MB / 8 = 4 MB, num_pipes = 32 (128 MB / 4 MB)
- Sender transport: 8 QPs × 2 × 32 MB = **512 MB**
- Receiver QP buffer: 8 QPs × 2 × 32 MB = **512 MB** (vs. 256 MB single-buffer)
- Reassembly buffers: 8 × 1 GB = 8 GB (unchanged)

---

## Open Questions

1. **NICs-only mode:** NICs-only has no NVLink step (data stays on GPU 0). With no
   NVLink to overlap, double-buffer provides no benefit. Should `--double-buffer` be
   silently ignored for NICs-only, or return a warning?

2. **All-to-all reassembly:** The all-to-all mode is more complex (N×M sources per QP).
   The double-buffer pattern applies in the same way, but the 2-pipe window interacts
   with the N×M leader election. Implement after multi-source reassembly is validated.

3. **NVLink vs RDMA bandwidth ratio:** The benefit of double-buffer depends on the
   relative speeds. If NVLink is much faster than RDMA (or vice versa), the overlap
   only hides the faster of the two and the slower still dominates. Benchmarking with
   `--transport-buffer` at different sizes will show at what chunk size double-buffer
   improves throughput.

# Multi-NIC RDMA Performance Tests

## Overview

This test suite measures RDMA bandwidth across multiple NICs and GPUs, supporting the following modes:

1. **Direct Mode**: Each GPU sends directly via its paired NIC
2. **NICs-Only Mode**: Single GPU (GPU 0) sends via all NICs
3. **NVLink Mode (no reassembly)**: Multiple GPUs send via all NICs; data lands on the remote side dispersed across target GPUs
4. **NVLink Mode with Reassembly**: Like NVLink mode, but the receiver NVLink-copies each received chunk into a per-source reassembly buffer
5. **All-to-All NVLink Reassembly**: Every source GPU sends a slice to every logical target GPU, using all NICs as transport and reassembling on the receiver into per-target buffers

Transport-buffer piping (`--transport-buffer`) and double-buffering (`--double-buffer`) are pipeline-optimization options that apply to modes 3–5.

---

## Test Modes

### 1. Direct Mode (`--direct`)

**Purpose**: Measure peak bandwidth when each GPU sends independently via its own NIC, without any NVLink transfers.

**Architecture**:
```
GPU 0 → NIC mlx5_0 → RDMA → Remote
GPU 1 → NIC mlx5_1 → RDMA → Remote
GPU 2 → NIC mlx5_2 → RDMA → Remote
...
GPU 7 → NIC mlx5_8 → RDMA → Remote
```

**Buffer Allocation**:
- **Location**: Each QP buffer allocated on its paired GPU
- **Size**: Default 1GB per GPU
- **Total memory**: 8GB (8 GPUs × 1GB)

**Data Flow**:
1. Each GPU's buffer contains test data
2. RDMA writes directly from GPU buffer via paired NIC
3. No NVLink transfers involved

**Threads**: 1 thread per GPU/NIC pair (8 threads for 8 NICs)

**Bandwidth Calculation**:
```
Total data = buffer_size × num_threads
           = 1GB × 8 = 8GB
Bandwidth = Total data / Time
```

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --direct

# Sender
./multi_nic_send_buffer --direct <receiver_ip>
```

**Filtering to a subset of GPUs** (direct mode only):
```bash
# Only GPUs 0, 3, 5 send via their paired NICs
./multi_nic_receive_buffer --direct --source-gpus 0,3,5
./multi_nic_send_buffer    --direct --source-gpus 0,3,5 <receiver_ip>
```

**All-to-all variant** (`--direct --all-to-all --source-gpus ...`): each source GPU sends its full buffer to every target NIC (N×K QPs total). No NVLink involved; each QP sends a slice of the source buffer.

**Use Case**: Baseline measurement showing maximum throughput when GPUs work independently.

---

### 2. NICs-Only Mode (`--nics-only`)

**Purpose**: All NICs send from a single GPU (GPU 0) without NVLink.

**Architecture**:
```
GPU 0 buffer → mlx5_0 → RDMA → Remote
GPU 0 buffer → mlx5_1 → RDMA → Remote
...
GPU 0 buffer → mlx5_8 → RDMA → Remote
```

**Buffer Allocation**:
- **Location**: All 8 QP buffers on GPU 0
- **Size**: 1GB total, partitioned into 8 slices (128MB each)
- **Total memory**: 1GB on GPU 0 only

**Threads**: 8 threads, all using GPU 0's buffer

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --nics-only

# Sender
./multi_nic_send_buffer --nics-only <receiver_ip>
```

---

### 3. NVLink Mode, No Reassembly (`--allow-nvlink --source-gpus ...`)

**Purpose**: Multiple source GPUs send data via all NICs, using NVLink for GPU-to-GPU staging. Data lands dispersed across target QP buffers on the receiver.

#### Single source GPU

```
GPU 0 source buffer (1GB) → sliced into K=8 parts:
  ├─ Slice 0 (128MB) → [local]  GPU 0 QP buffer → mlx5_0 → RDMA
  ├─ Slice 1 (128MB) → NVLink → GPU 1 QP buffer → mlx5_1 → RDMA
  ...
  └─ Slice 7 (128MB) → NVLink → GPU 7 QP buffer → mlx5_8 → RDMA
```

Receiver: one QP buffer per NIC (128MB each), data arrives directly — no reassembly step.

#### Multiple source GPUs (`--source-gpus 0,1,...,7`)

N=8 source GPUs, K=8 QPs. Each QP buffer receives one section from each source GPU.

```
QP k buffer (1GB) = [src0_section | src1_section | ... | src7_section]
Each section = 128MB = (source buffer) / K
```

**Threads**: N × K per side. `qp_mutex` serializes RDMA when N > 1 sources share a QP.

**Bandwidth** (N=8): 8 × 1GB = 8GB per iteration.

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7

# Sender
./multi_nic_send_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7 <receiver_ip>
```

---

### 4. NVLink Mode with Reassembly (`--allow-nvlink --reassembly --source-gpus ...`)

**Purpose**: Like Mode 3, but the receiver NVLink-copies each received section from its QP buffer into a dedicated **reassembly buffer** for each source GPU. The QP buffer can be reused immediately after reassembly (enabling piping with `--transport-buffer`).

**Receiver buffer layout**:
```
reassembly_buffer[src_n]:   [ QP0_section | QP1_section | ... | QP7_section ]
                               (each QP fills its own 128MB slice)
```

**Synchronization**: The receiver sends a per-pipe (or per-iteration) RDMA write-with-imm signal back to the sender after each QP buffer is reassembled. The sender must wait for this signal before reusing the QP transport buffer.

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7

# Sender
./multi_nic_send_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 <receiver_ip>
```

See [`RDMA_PIPELINES_AND_SYNC.md`](RDMA_PIPELINES_AND_SYNC.md) for the full pipeline and signal protocol.

---

### 5. All-to-All NVLink Reassembly (`--allow-nvlink --reassembly --all-to-all --source-gpus ...`)

**Purpose**: Every source GPU (N total) sends a portion of its data to every logical target GPU (M total), using K transport NICs. Each target GPU receives contributions from all N sources. This is the "all-reduce scatter" building block.

**Key parameters**:
- **N** = number of source GPUs (`--source-gpus`)
- **K** = number of transport NICs/QPs (`-n` / `-g` count)
- **M** = number of logical target GPUs (`--target-count`, default M=K)

**Data sizes** (1GB source per GPU):
```
sub_slice_size = 1GB / (M × K)          # per (n,m) slot per QP
slice_size     = 1GB / M                 # one target's contribution from one source
QP buffer size = N × (1GB / K)          # unchanged; N×M slots each of sub_slice
```

**Sender threads**: N × K × M. Thread (n, k, m) copies the sub-slice for target m, transport QP k, from source n, into QP buffer k at slot (n×M+m), then RDMA-writes it.

**Receiver**: K threads (one per QP). Each polls N×M completions, then NVLink-copies each slot into `reassembly_buffers_2d[n×M+m]` at offset `k × sub_slice`.

**Result**: M reassembly buffers, each of size N × (1GB/M), containing contributions from all N sources.

**Example Run** (N=8, K=8, M=8):
```bash
# Receiver
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all --source-gpus 0,1,2,3,4,5,6,7

# Sender
./multi_nic_send_buffer --allow-nvlink --reassembly --all-to-all --source-gpus 0,1,2,3,4,5,6,7 <receiver_ip>
```

**With M ≠ K** (see `--target-count` below):
```bash
# N=2 sources, M=4 targets, K=8 NICs
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all --source-gpus 0,1 --target-count 4
./multi_nic_send_buffer    --allow-nvlink --reassembly --all-to-all --source-gpus 0,1 --target-count 4 <receiver_ip>
```

---

## Pipeline Optimization Options

### Transport Buffer (`--transport-buffer SIZE`)

By default, the QP staging buffer equals the full slice size (data buffer size per source per QP). `--transport-buffer SIZE` reduces the staging buffer to `SIZE` bytes, then transfers the full slice in `num_pipes = ceil(slice_size / section_size)` successive chunks (**piping**).

**When useful**: When NVLink BW or PCIe BW limits total throughput, a smaller staging buffer reduces memory pressure and can improve cache utilization. With `--double-buffer`, it also enables pipelining of NVLink fills with RDMA sends.

**Applies to**: Modes 3, 4, 5. Not valid with `--direct` or `--nics-only`.

**Size format**: bytes, or suffix `K`/`M`/`G` (e.g. `32M`, `128M`, `1G`).

**Example** (32MB transport buffer, 128MB slice → 4 pipes per QP per source):
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M
./multi_nic_send_buffer    --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M <receiver_ip>
```

**Receiver buffer sizing**:

| Mode | Receiver QP buffer |
|------|--------------------|
| No reassembly | Full slice size (unchanged); pipes land at distinct offsets |
| Reassembly (Mode 4 or 5) | = `transport_buffer_size` (small); each pipe is reassembled immediately |
| Reassembly + double-buffer | = `2 × transport_buffer_size` |

See [`TRANSPORT_BUFFER_PIPING_DESIGN.md`](TRANSPORT_BUFFER_PIPING_DESIGN.md) for full design.

---

### Double Buffer (`--double-buffer`)

Allocates **two** transport buffers per QP per source thread, and alternates between them so that the NVLink fill of buffer[next] overlaps with the RDMA send of buffer[current].

**Requires**: `--transport-buffer`. Not valid with `--direct` or `--nics-only`.

**Applies to**: All NVLink modes (3, 4, 5) with piping active.

**Benefits**:
- **Without reassembly**: NVLink fill and RDMA send run concurrently. Per-pipe latency ≈ max(T_nvlink, T_rdma) instead of T_nvlink + T_rdma.
- **With reassembly** (larger benefit): Sender keeps 2 pipes in flight. While the receiver reassembles pipe p, the sender is already sending pipe p+1. Receiver reassembly latency is hidden behind the next RDMA send.

**Memory impact** (8 sources, 8 QPs, 32MB transport buffer):

| Mode | Sender transport | Receiver QP buffer |
|------|------------------|--------------------|
| Single buffer | 8 QPs × 32MB = 256MB | reassembly: 8 × 32MB = 256MB |
| Double buffer | 8 QPs × 64MB = 512MB | reassembly: 8 × 64MB = 512MB |

**Example** (8 sources, reassembly, 32MB transport, double buffer):
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M --double-buffer
./multi_nic_send_buffer    --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M --double-buffer <receiver_ip>
```

**All-to-all with double buffer**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M --double-buffer
./multi_nic_send_buffer    --allow-nvlink --reassembly --all-to-all --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M --double-buffer <receiver_ip>
```

See [`DOUBLE_BUFFER_TRANSPORT_DESIGN.md`](DOUBLE_BUFFER_TRANSPORT_DESIGN.md) for the full design.

---

### Target Count (`--target-count M`)

Decouples the number of **logical target GPUs** (M) from the number of **transport NICs** (K). By default M=K; this option sets M independently.

**Applies to**: `--allow-nvlink --reassembly --all-to-all` only. Not valid with `--direct`.

**Constraint**: M ≤ K. The first M entries of `-g` / `--gpus` are used as the target GPU list.

**Data sizes** with M ≠ K:
```
sub_slice_size = 1GB / (M × K)
slice_size     = 1GB / M          (per-target slice from one source)
QP buffer      = N × (1GB / K)   (unchanged)
Reassembly buf = N × M buffers, each 1GB/M
Sender threads = N × K × M
```

**Example** (N=2, M=4, K=8):
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1 --target-count 4
./multi_nic_send_buffer    --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1 --target-count 4 <receiver_ip>
```

Result: 4 reassembly buffers (on GPUs 0–3), each of size 2 × 256MB = 512MB.

See [`ALL_TO_ALL_TARGET_COUNT_DESIGN.md`](ALL_TO_ALL_TARGET_COUNT_DESIGN.md) for the full design.

---

## Buffer Management

### Direct Mode
```
QP 0: GPU 0 (1GB) ──RDMA──> Remote
QP 1: GPU 1 (1GB) ──RDMA──> Remote
...
QP 7: GPU 7 (1GB) ──RDMA──> Remote
```

### NVLink Mode, No Reassembly (8 sources, 8 QPs)

**Source Buffers** (1GB each on source GPUs):
```
GPU 0: [─────────── 1GB ───────────] (8 slices × 128MB)
...
GPU 7: [─────────── 1GB ───────────] (8 slices × 128MB)
```

**QP Buffers** (1GB each on target GPUs):
```
QP 0 (GPU 0): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_0
...
QP 7 (GPU 7): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_7
```
Where `Gx` = 128MB section from GPU x.

### NVLink Mode with Reassembly (8 sources, 8 QPs)

**QP Buffer** (sender; with `--transport-buffer 32M`):
```
sender QP buffer = 32MB (single) or 64MB (double-buffer)
Each pipe = 4MB per source
```

**Reassembly Buffers** (receiver, one per source GPU):
```
reassembly_buffer[GPU n]:  [ QP0 | QP1 | QP2 | QP3 | QP4 | QP5 | QP6 | QP7 ]
                              128MB per QP slice = 1GB total
```

### All-to-All Reassembly (N=8, M=K=8, 8 QPs)

**QP Buffer** (receiver; full size = N × 1GB/K = 1GB):
```
QP k buffer: [ (n=0,m=0) | (n=0,m=1) | ... | (n=7,m=7) ]   (N×M slots)
Each slot = sub_slice_size = 1GB / (M×K) = 16MB
```

**Reassembly Buffers** (receiver; N×M = 64 buffers, each 1GB/M = 128MB):
```
reassembly_buffers_2d[n*M+m] on GPU m:
  [ QP0_sub_slice | QP1_sub_slice | ... | QP7_sub_slice ]
```

---

## Performance Metrics

### Timing Methodology
1. **Warmup phase**: 5 iterations (not timed)
2. **Barrier synchronization**: All threads reach start barrier
3. **Timed phase**: 10 iterations
4. **Wall-clock measurement**: Total time from start to end barrier

### Bandwidth Formula
```
Bandwidth (GB/s) = Total_Data_Transferred / Time_Elapsed
```

| Mode | Total data per iteration |
|------|--------------------------|
| Direct (N GPUs) | N × 1GB |
| NICs-only | 1GB |
| NVLink (N sources, K QPs) | N × 1GB |
| All-to-all (N sources, M targets) | N × 1GB |

---

## Data Validation

### Initialization (Sender)
**Direct Mode**: Write to start of each QP buffer:
```
"one buffer of size 1073741824 on GPU_X\n"
```

**NVLink Mode**: Write to each slice of source buffers:
```
Source GPU 0, slice 0: "buffer number 0 on source gpu_0\n"
Source GPU 0, slice 1: "buffer number 1 on source gpu_0\n"
...
Source GPU 7, slice 7: "buffer number 7 on source gpu_7\n"
```

### Verification (Both sides)
**Sender**: Verifies NVLink copies succeeded.
**Receiver**: Verifies RDMA transfers succeeded; for reassembly modes, verifies data landed in the correct reassembly buffer slot.

---

## Hardware Requirements

- **GPUs**: NVIDIA GPUs with CUDA support
- **NICs**: Mellanox ConnectX adapters (mlx5_X)
- **NVLink**: Required for Modes 3–5
- **NUMA**: Performance optimal when GPU-NIC pairs are on same NUMA node

---

## Determining GPU-NIC Topology

### Step 1: Run nvidia-smi topo

```bash
nvidia-smi topo -m
```

**Example Output**:
```
        GPU0    GPU1    ...    mlx5_0  mlx5_1  ...
GPU0     X      NV12           PIX     SYS
GPU1    NV12     X             SYS     PIX
...
```

Look for **PIX** connections (same PCIe switch, optimal performance).

### Step 2: Configure Wrapper Scripts

Edit the wrapper scripts to match your topology:

**File: `multi_nic_send_buffer`** and **`multi_nic_receive_buffer`**:
```bash
ALL_NICS="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
ALL_GPUS="0,1,2,3,4,5,6,7"
```

Order matters: GPU index i must map to NIC index i.

---

## Usage Examples

### Basic Tests

**1. Direct mode (8 GPUs, 8 NICs)**:
```bash
./multi_nic_receive_buffer --direct
./multi_nic_send_buffer --direct <server_ip>
```

**2. Single GPU via all NICs**:
```bash
./multi_nic_receive_buffer --nics-only
./multi_nic_send_buffer --nics-only <server_ip>
```

### NVLink Tests

**3. Single source GPU, no reassembly**:
```bash
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0
./multi_nic_send_buffer --allow-nvlink --source-gpus 0 <server_ip>
```

**4. 8 source GPUs, no reassembly**:
```bash
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7
./multi_nic_send_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7 <server_ip>
```

**5. 8 source GPUs with reassembly**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7
./multi_nic_send_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 <server_ip>
```

### Transport Buffer / Piping

**6. Reassembly with 32MB transport buffer (4 pipes)**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
    --transport-buffer 32M
./multi_nic_send_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
    --transport-buffer 32M <server_ip>
```

**7. Piping with double-buffer (overlapped NVLink + RDMA)**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
    --transport-buffer 32M --double-buffer
./multi_nic_send_buffer --allow-nvlink --reassembly --source-gpus 0,1,2,3,4,5,6,7 \
    --transport-buffer 32M --double-buffer <server_ip>
```

### All-to-All

**8. All-to-all reassembly (N=8, M=K=8)**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1,2,3,4,5,6,7
./multi_nic_send_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1,2,3,4,5,6,7 <server_ip>
```

**9. All-to-all with M ≠ K (N=2, M=4, K=8)**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1 --target-count 4
./multi_nic_send_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1 --target-count 4 <server_ip>
```

**10. All-to-all with piping and double-buffer**:
```bash
./multi_nic_receive_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M --double-buffer
./multi_nic_send_buffer --allow-nvlink --reassembly --all-to-all \
    --source-gpus 0,1,2,3,4,5,6,7 --transport-buffer 32M --double-buffer <server_ip>
```

---

## Option Reference

| Option | Short | Description | Constraints |
|--------|-------|-------------|-------------|
| `--direct` | `-D` | Each GPU sends directly via its own NIC | — |
| `--nics-only` | | All NICs send from GPU 0 | — |
| `--allow-nvlink` | | Enable NVLink staging | Requires `--source-gpus` |
| `--reassembly` | `-R` | Receiver reassembles into per-source buffers | Requires `--allow-nvlink` |
| `--all-to-all` | `-A` | Every source sends to every target | Requires `--direct` or `--allow-nvlink --reassembly` + `--source-gpus` |
| `--source-gpus LIST` | `-S` | Source GPU IDs | — |
| `--transport-buffer SIZE` | `-T` | QP staging buffer size; enables piping | Not valid with `--direct` or `--nics-only` |
| `--double-buffer` | `-B` | Allocate 2 transport bufs; overlap NVLink+RDMA | Requires `--transport-buffer`; not valid with `--direct` or `--nics-only` |
| `--target-count M` | `-C` | Logical target count (default M=K) | Only with `--allow-nvlink --reassembly --all-to-all`; M ≤ K |
| `--utilize-nic NIC` | `-U` | Background load thread on specified NIC | NIC must be in `-n` list |
| `--debug` | | Verbose per-pipe/per-iteration output | — |

---

## Code Structure

### Main Components

**`test_rdma_multi_qp.c`**: Main test program
- Command-line parsing
- Buffer allocation (source + QP + reassembly buffers)
- Thread management
- Bandwidth calculation
- Data validation

**`rdma_multi_qp.c`**: RDMA Multi-QP API
- QP creation and connection
- Buffer registration (MR exchange)
- RDMA write operations (plain and write-with-imm)
- Completion polling (send CQ and recv CQ, isolated)

**`nvlink_transfer.c`**: NVLink operations
- Peer access management
- GPU-to-GPU async copies
- Stream synchronization

**Wrapper Scripts**:
- `multi_nic_send_buffer`: Sender wrapper (NIC/GPU lists pre-configured)
- `multi_nic_receive_buffer`: Receiver wrapper (NIC/GPU lists pre-configured)

### Design Documents

| Document | Contents |
|----------|----------|
| `RDMA_PIPELINES_AND_SYNC.md` | Per-mode pipeline pseudocode, barrier tables, piping + double-buffer signal protocol |
| `TRANSPORT_BUFFER_PIPING_DESIGN.md` | Transport buffer sizing, pipe loop design, CQ isolation |
| `DOUBLE_BUFFER_TRANSPORT_DESIGN.md` | Double-buffer memory layout, sender/receiver pseudocode, overlap analysis |
| `ALL_TO_ALL_TARGET_COUNT_DESIGN.md` | M ≠ K formulas, thread counts, buffer allocation changes |

---

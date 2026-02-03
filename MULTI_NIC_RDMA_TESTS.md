# Multi-NIC RDMA Performance Tests

## Overview

This test suite measures RDMA bandwidth across multiple NICs and GPUs, supporting three distinct modes:
1. **Direct Mode**: Each GPU sends directly via its paired NIC
2. **NICs-Only Mode**: Single GPU (GPU 0) sends via all NICs
3. **Multi-Source NVLink Mode**: Multiple GPUs send via all NICs using NVLink for GPU-to-GPU transfers

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

**Use Case**: Baseline measurement showing maximum throughput when GPUs work independently.

---

### 2. NICs-Only Mode (`--nics-only`)

**Purpose**: All NICs send from a single GPU (GPU 0) without NVLink.

**Architecture**:
```
GPU 0 buffer → mlx5_0 → RDMA → Remote
GPU 0 buffer → mlx5_1 → RDMA → Remote
GPU 0 buffer → mlx5_2 → RDMA → Remote
...
GPU 0 buffer → mlx5_8 → RDMA → Remote
```

**Buffer Allocation**:
- **Location**: All 8 QP buffers on GPU 0
- **Size**: 1GB total, partitioned into 8 slices (128MB each)
- **Total memory**: 1GB on GPU 0 only

**Data Flow**:
1. Single buffer on GPU 0 divided into 8 parts
2. Each thread reads its partition (128MB)
3. RDMA write via different NICs in parallel

**Threads**: 8 threads, all using GPU 0's buffer

**Bandwidth Calculation**:
```
Total data = buffer_size = 1GB
Bandwidth = 1GB / Time
```

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --nics-only

# Sender
./multi_nic_send_buffer --nics-only <receiver_ip>
```

---

### 3. Multi-Source NVLink Mode (`--allow-nvlink --source-gpus`)

**Purpose**: Multiple source GPUs send data via all NICs, utilizing NVLink for GPU-to-GPU transfers.

#### 3a. Single Source GPU

**Example**: `--allow-nvlink --source-gpus 0`

**Architecture**:
```
GPU 0 source buffer (1GB) → sliced into 8 parts:
  ├─ Slice 0 (128MB) → GPU 0 QP buffer → mlx5_0 → RDMA
  ├─ Slice 1 (128MB) → NVLink → GPU 1 QP buffer → mlx5_1 → RDMA
  ├─ Slice 2 (128MB) → NVLink → GPU 2 QP buffer → mlx5_2 → RDMA
  ├─ Slice 3 (128MB) → NVLink → GPU 3 QP buffer → mlx5_4 → RDMA
  ├─ Slice 4 (128MB) → NVLink → GPU 4 QP buffer → mlx5_5 → RDMA
  ├─ Slice 5 (128MB) → NVLink → GPU 5 QP buffer → mlx5_6 → RDMA
  ├─ Slice 6 (128MB) → NVLink → GPU 6 QP buffer → mlx5_7 → RDMA
  └─ Slice 7 (128MB) → NVLink → GPU 7 QP buffer → mlx5_8 → RDMA
```
Note that the reciever has also one GPU paired to each QP so the data is also disparsed on dest.

**Buffer Allocation**:
- **Source buffer**: 1GB on GPU 0 (full size)
- **QP buffers**: 128MB on each target GPU (0-7)
- **Total memory**: 1GB (source) + 8×128MB (targets) = 2GB

**Data Flow**:
1. GPU 0 source buffer divided into 8 slices (128MB each)
2. Thread for QP i:
   - Read: source_buffer[i × 128MB]
   - Copy: via NVLink (or local if same GPU) → target GPU i's QP buffer
   - Send: RDMA write from target GPU i's QP buffer
3. All 8 NICs send in parallel

**Threads**: 8 threads (1 per target QP)

**Bandwidth Calculation**:
```
Total data = original_buffer_size = 1GB
Bandwidth = 1GB / Time
```

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0

# Sender
./multi_nic_send_buffer --allow-nvlink --source-gpus 0 <receiver_ip>
```

#### 3b. Multiple Source GPUs

**Example**: `--allow-nvlink --source-gpus 0,1,2,3,4,5,6,7`

**Architecture**:
```
For each target GPU T (0-7):
  QP buffer on GPU T receives from ALL 8 source GPUs:
  ├─ Source GPU 0: slice T → offset 0×128MB in GPU T buffer
  ├─ Source GPU 1: slice T → offset 1×128MB in GPU T buffer
  ├─ Source GPU 2: slice T → offset 2×128MB in GPU T buffer
  ├─ Source GPU 3: slice T → offset 3×128MB in GPU T buffer
  ├─ Source GPU 4: slice T → offset 4×128MB in GPU T buffer
  ├─ Source GPU 5: slice T → offset 5×128MB in GPU T buffer
  ├─ Source GPU 6: slice T → offset 6×128MB in GPU T buffer
  └─ Source GPU 7: slice T → offset 7×128MB in GPU T buffer
  
  Then: GPU T sends its full 1GB buffer via its NIC
```

**Buffer Allocation**:
- **Source buffers**: 1GB on each of 8 source GPUs (8GB total)
- **QP buffers**: 1GB on each of 8 target GPUs (8GB total)
- **Total memory**: 16GB

**Buffer Size Calculation**:
```
QP buffer size = (original_buffer_size × num_source_gpus) / num_qps
               = (1GB × 8) / 8
               = 1GB per QP
```

**Data Flow** (for Target GPU 3 as example):
1. 8 parallel threads write to GPU 3's QP buffer:
   - Thread from Source GPU 0:
     - Read: GPU 0 source_buffer[3 × 128MB]  (slice 3)
     - NVLink to: GPU 3 QP_buffer[0 × 128MB]  (offset for source 0)
   - Thread from Source GPU 1:
     - Read: GPU 1 source_buffer[3 × 128MB]  (slice 3)
     - NVLink to: GPU 3 QP_buffer[1 × 128MB]  (offset for source 1)
   - ... (same pattern for GPUs 2-6)
   - Thread from Source GPU 7:
     - Read: GPU 7 source_buffer[3 × 128MB]  (slice 3)
     - NVLink to: GPU 3 QP_buffer[7 × 128MB]  (offset for source 7)

2. After all NVLink copies complete:
   - GPU 3's QP buffer contains: [GPU0_data | GPU1_data | ... | GPU7_data]
   - Total: 8 × 128MB = 1GB

3. RDMA send from GPU 3 via mlx5_4

**Thread Count**: M source GPUs × N target QPs = 8 × 8 = 64 threads

**Bandwidth Calculation**:
```
Each source GPU contributes: 1GB
Total data = 1GB × 8 = 8GB
Bandwidth = 8GB / Time
```

**Example Run**:
```bash
# Receiver
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7

# Sender
./multi_nic_send_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7 <receiver_ip>
```

**Special Cases**:
- When source GPU == target GPU (diagonal): Uses local `cudaMemcpy` instead of NVLink
- Example: Thread copying GPU 0 → GPU 0 does local copy, not NVLink

---

## Buffer Management

### Direct Mode
```
QP 0: GPU 0 (1GB) ──RDMA──> Remote
QP 1: GPU 1 (1GB) ──RDMA──> Remote
...
QP 7: GPU 7 (1GB) ──RDMA──> Remote
```

### Multi-Source NVLink (8 sources)

**Source Buffers** (1GB each on source GPUs):
```
GPU 0: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 1: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 2: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 3: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 4: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 5: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 6: [─────────── 1GB ───────────] (8 slices × 128MB)
GPU 7: [─────────── 1GB ───────────] (8 slices × 128MB)
```

**QP Buffers** (1GB each on target GPUs):
```
QP 0 (GPU 0): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_0
QP 1 (GPU 1): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_1
QP 2 (GPU 2): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_2
QP 3 (GPU 3): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_3
QP 4 (GPU 4): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_4
QP 5 (GPU 5): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_5
QP 6 (GPU 6): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_6
QP 7 (GPU 7): [G0|G1|G2|G3|G4|G5|G6|G7] ─RDMA→ mlx5_7
```
Where: `Gx` = 128MB section from GPU x

**Data Flow** (via NVLink/local copy):
- Each source GPU sends its slices to ALL target QP buffers
- Each QP buffer receives one slice from EACH source GPU
- Total: 8 sources × 8 slices = 64 NVLink transfers per iteration

---

## Performance Metrics

### Bandwidth Formula
```
Bandwidth (GB/s) = Total_Data_Transferred / Time_Elapsed

Where:
- Total_Data_Transferred depends on mode
- Time_Elapsed measured with CPU cycle counters (get_cycles())
```

### Timing Methodology
1. **Warmup phase**: 5 iterations (not timed)
2. **Barrier synchronization**: All threads reach start barrier
3. **Timed phase**: 10 iterations
   - Start: All threads begin simultaneously
   - Work: NVLink copies + RDMA writes
   - End: All threads complete
4. **Wall-clock measurement**: Total time from start to end barrier

### Per-Thread Bandwidth
```
Thread_BW = (data_size × iterations) / thread_time
```

### Total Bandwidth
```
Total_BW = Total_Data / Wall_Clock_Time
```
This accounts for parallelism and gives aggregate throughput.

---

## Data Validation

The test includes validation to verify correct data transfer:

### Initialization (Client/Sender)
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
**Client**: Verifies NVLink copies succeeded
**Server**: Verifies RDMA transfers succeeded

**Output Example** (Server, NVLink mode with 2 sources):
```
=== Verifying transferred data ===
Target GPU_0 (QP 0) sections:
  Section 0: buffer number 0 on source gpu_0
  Section 1: buffer number 0 on source gpu_1
Target GPU_1 (QP 1) sections:
  Section 0: buffer number 1 on source gpu_0
  Section 1: buffer number 1 on source gpu_1
...
=== Data verification complete ===
```

This confirms:
- Correct source GPU data reached correct target
- Correct slice from each source reached correct offset
- Data integrity through NVLink + RDMA pipeline

---

## Hardware Requirements

- **GPUs**: NVIDIA GPUs with CUDA support
- **NICs**: Mellanox ConnectX adapters (mlx5_X)
- **NVLink**: Required for multi-source NVLink mode
- **NUMA**: Performance optimal when GPU-NIC pairs are on same NUMA node

---

## Determining GPU-NIC Topology

### Step 1: Run nvidia-smi topo

First, check your system's GPU-NIC topology:

```bash
nvidia-smi topo -m
```

**Example Output**:
```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  mlx5_1  mlx5_2  mlx5_3  mlx5_4  mlx5_5  mlx5_6  mlx5_7
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12    PIX     SYS     SYS     SYS     SYS     SYS     SYS     SYS
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12    SYS     PIX     SYS     SYS     SYS     SYS     SYS     SYS
GPU2    NV12    NV12     X      NV12    NV12    NV12    NV12    NV12    SYS     SYS     PIX     SYS     SYS     SYS     SYS     SYS
GPU3    NV12    NV12    NV12     X      NV12    NV12    NV12    NV12    SYS     SYS     SYS     PIX     SYS     SYS     SYS     SYS
GPU4    NV12    NV12    NV12    NV12     X      NV12    NV12    NV12    SYS     SYS     SYS     SYS     PIX     SYS     SYS     SYS
GPU5    NV12    NV12    NV12    NV12    NV12     X      NV12    NV12    SYS     SYS     SYS     SYS     SYS     PIX     SYS     SYS
GPU6    NV12    NV12    NV12    NV12    NV12    NV12     X      NV12    SYS     SYS     SYS     SYS     SYS     SYS     PIX     SYS
GPU7    NV12    NV12    NV12    NV12    NV12    NV12    NV12     X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (slow)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge (BEST for GPU-NIC)
  NV#  = Connection traversing a bonded set of # NVLinks
```

### Step 2: Identify GPU-NIC Pairs

Look for **PIX** connections between GPUs and NICs. PIX means the GPU and NIC are:
- On the same PCIe switch
- Minimal PCIe hops
- **Optimal performance**

From the example above:
```
GPU 0 ↔ mlx5_0 (PIX) ✓ BEST
GPU 1 ↔ mlx5_1 (PIX) ✓ BEST
GPU 2 ↔ mlx5_2 (PIX) ✓ BEST
GPU 3 ↔ mlx5_3 (PIX) ✓ BEST
GPU 4 ↔ mlx5_4 (PIX) ✓ BEST
GPU 5 ↔ mlx5_5 (PIX) ✓ BEST
GPU 6 ↔ mlx5_6 (PIX) ✓ BEST
GPU 7 ↔ mlx5_7 (PIX) ✓ BEST
```

**Avoid SYS connections**: These cross NUMA nodes and have poor performance.

### Step 3: Configure Wrapper Scripts

Edit the wrapper scripts to match your topology:

**File: `multi_nic_send_buffer`**
```bash
# Line ~11-12: Set based on your nvidia-smi topo -m output
ALL_NICS="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
ALL_GPUS="0,1,2,3,4,5,6,7"
```

**File: `multi_nic_receive_buffer`**
```bash
# Line ~11-12: Must match sender
ALL_NICS="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
ALL_GPUS="0,1,2,3,4,5,6,7"
```

**Important**: 
- Order matters! GPU index must match NIC index
- Position 0: GPU 0 paired with first NIC (mlx5_0)
- Position 1: GPU 1 paired with second NIC (mlx5_1)
- etc.

### Step 4: Verify Configuration

After configuring, test with a simple direct mode run:

```bash
# On receiver node
./multi_nic_receive_buffer --direct

# On sender node (should see PIX connections being used)
./multi_nic_send_buffer --direct <receiver_ip>
```

Check output shows correct GPU-NIC pairs are used.

### Example: Non-Standard Topology

If your system has different mappings:

```bash
# nvidia-smi topo -m shows:
# GPU 0 ↔ mlx5_3 (PIX)
# GPU 1 ↔ mlx5_1 (PIX)
# GPU 2 ↔ mlx5_0 (PIX)
# GPU 3 ↔ mlx5_2 (PIX)

# Configure scripts accordingly:
ALL_NICS="mlx5_3,mlx5_1,mlx5_0,mlx5_2"  # Order matches GPU order!
ALL_GPUS="0,1,2,3"
```

## Usage Examples

### Basic Tests

**1. Test single NIC (no GPU)**:
```bash
# Server
./test_rdma_multi_qp -s -n mlx5_0

# Client
./test_rdma_multi_qp -n mlx5_0 -a <server_ip>
```

**2. Direct mode (8 GPUs, 8 NICs)**:
```bash
# Server
./multi_nic_receive_buffer --direct

# Client
./multi_nic_send_buffer --direct <server_ip>
```

**3. Single GPU via all NICs**:
```bash
# Server
./multi_nic_receive_buffer --nics-only

# Client  
./multi_nic_send_buffer --nics-only <server_ip>
```

### Advanced NVLink Tests

**4. One source GPU using all NICs**:
```bash
# Server
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0

# Client
./multi_nic_send_buffer --allow-nvlink --source-gpus 0 <server_ip>
```

**5. Two source GPUs**:
```bash
# Server
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0,1

# Client
./multi_nic_send_buffer --allow-nvlink --source-gpus 0,1 <server_ip>
```

**6. All GPUs as sources** (maximum data, uses NVLink):
```bash
# Server
./multi_nic_receive_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7

# Client
./multi_nic_send_buffer --allow-nvlink --source-gpus 0,1,2,3,4,5,6,7 <server_ip>
```

**7. Direct mode with specific GPUs**:
```bash
# Use only GPUs 0,1,2
# Server
./test_rdma_multi_qp -s -n mlx5_0,mlx5_1,mlx5_2 -g 0,1,2 --direct

# Client
./test_rdma_multi_qp -n mlx5_0,mlx5_1,mlx5_2 -g 0,1,2 --direct -a <server_ip>
```

---

## Code Structure

### Main Components

**`test_rdma_multi_qp.c`**: Main test program
- Command-line parsing
- Buffer allocation (source + QP buffers)
- Thread management
- Bandwidth calculation
- Data validation

**`rdma_multi_qp.c`**: RDMA Multi-QP API
- QP creation and connection
- Buffer registration
- RDMA write operations
- Completion polling

**`nvlink_transfer.c`**: NVLink operations
- Peer access management
- GPU-to-GPU async copies
- Stream synchronization

**Wrapper Scripts**:
- `multi_nic_send_buffer`: Client wrapper
- `multi_nic_receive_buffer`: Server wrapper
- Pre-configured NIC/GPU lists

---
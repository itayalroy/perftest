#!/bin/bash
#SBATCH --job-name=rdma_multi_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=rdma_tests_%j.out
#SBATCH --error=rdma_tests_%j.err
#SBATCH --partition=interactive
#SBATCH --account=network_research_advdev

# Run all RDMA sender/receiver tests: for each source-gpus option, run 5 modes.
# Uses srun: receiver on first node, sender on second node.
#
# Container: When CONTAINER_IMAGE is set, the host directory (BINDIR) is mounted at CONTAINER_WORKDIR
#            inside the container and all test commands run with "cd CONTAINER_WORKDIR && ./multi_nic_*".
#            Set CONTAINER_WORKDIR if your image uses a different path (default /workspace).
#            Optional BIND_MOUNTS adds extra src:dst bind mounts (comma-separated).
# IB IP:    Set RECEIVER_IB_IP to the InfiniBand IP of the first node, or leave unset to auto-detect from the first interface whose name starts with "ib".
#
# Submit:   sbatch run_all_rdma_tests.sbatch
# With container: CONTAINER_IMAGE=/path/img.sif [CONTAINER_WORKDIR=/workspace] sbatch run_all_rdma_tests.sbatch

# It instructs the shell to immediately terminate the script if any command returns a non-zero exit status (which usually indicates an error).
set -e

# Directory containing test_rdma_multi_qp and multi_nic_* scripts (on the host)
BINDIR="${BINDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$BINDIR"

# Container: we mount your host dir (BINDIR) into the container at CONTAINER_WORKDIR and run "cd CONTAINER_WORKDIR && ./multi_nic_*" there.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
# Path *inside the container* where BINDIR is mounted and where we cd before running (e.g. /workspace).
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
# Extra bind mounts only (comma-separated src:dst, e.g. /lustre:/lustre). The mount BINDIR -> CONTAINER_WORKDIR is always added for you.
BIND_MOUNTS="${BIND_MOUNTS:-}"

# When using a container: mount BINDIR at CONTAINER_WORKDIR so the same files are visible inside; every srun does cd CONTAINER_WORKDIR then run.
SRUN_OPTS=()
if [[ -n "$CONTAINER_IMAGE" ]]; then
  SRUN_OPTS+=(--container="$CONTAINER_IMAGE")
  LOCAL_MOUNT="$BINDIR:$CONTAINER_WORKDIR"
  if [[ -n "$BIND_MOUNTS" ]]; then
    SRUN_OPTS+=(--bind="${LOCAL_MOUNT},${BIND_MOUNTS}")
  else
    SRUN_OPTS+=(--bind="$LOCAL_MOUNT")
  fi
fi

# Nodes: first = receiver, second = sender
JOB_ID="${SLURM_JOB_ID:-$$}"
NODELIST="${SLURM_NODELIST:-}"
if [[ -z "$NODELIST" ]]; then
  echo "ERROR: SLURM_NODELIST not set. Run this script under sbatch/srun."
  exit 1
fi
FIRST_NODE=$(scontrol show hostnames "$NODELIST" | head -1)
SECOND_NODE=$(scontrol show hostnames "$NODELIST" | tail -1)

# Log and results in BINDIR: include date, time, and node names so runs don't overwrite and are easy to identify
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
NODE_SUFFIX=$(echo "${FIRST_NODE}_${SECOND_NODE}" | tr -c '[:alnum:]_-' '_')
LOG_FILE="${LOG_FILE:-$BINDIR/rdma_tests_full_${JOB_ID}_${TIMESTAMP}_${NODE_SUFFIX}.log}"
RESULTS_FILE="${RESULTS_FILE:-$BINDIR/rdma_tests_results_${JOB_ID}_${TIMESTAMP}_${NODE_SUFFIX}.txt}"

# IB IP of first node (receiver). Set RECEIVER_IB_IP in the environment to override.
if [[ -n "$RECEIVER_IB_IP" ]]; then
  RECEIVER_IP="$RECEIVER_IB_IP"
else
  RECEIVER_IP=$(srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$FIRST_NODE" --gres=gpu:0 bash -c '
    for iface in /sys/class/net/ib*; do
      [[ -e "$iface" ]] || continue
      iface=$(basename "$iface")
      ip=$(ip -4 addr show "$iface" 2>/dev/null | grep -oP "inet \K[0-9.]+" | head -1)
      if [[ -n "$ip" ]]; then echo "$ip"; exit 0; fi
    done
    hostname -I 2>/dev/null | awk "{print \$1}"
  ' 2>/dev/null | head -1 | tr -d ' \n')
fi
if [[ -z "$RECEIVER_IP" ]]; then
  echo "ERROR: Could not get receiver IB IP. Set RECEIVER_IB_IP or ensure an ib* interface has an address on the first node."
  exit 1
fi

echo "=== RDMA multi-QP test suite ===" | tee -a "$LOG_FILE"
echo "First node (receiver): $FIRST_NODE" | tee -a "$LOG_FILE"
echo "Second node (sender):  $SECOND_NODE" | tee -a "$LOG_FILE"
echo "Receiver IP:           $RECEIVER_IP" | tee -a "$LOG_FILE"
echo "Partition:             ${PARTITION:-<default>}" | tee -a "$LOG_FILE"
echo "Account (group):       ${ACCOUNT:-<default>}" | tee -a "$LOG_FILE"
echo "Container:             ${CONTAINER_IMAGE:-<none>}" | tee -a "$LOG_FILE"
echo "Container workdir:     ${CONTAINER_WORKDIR:-<n/a>}" | tee -a "$LOG_FILE"
echo "Bind mounts:           ${BIND_MOUNTS:-<none>}" | tee -a "$LOG_FILE"
echo "Log file:              $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results file:          $RESULTS_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Source-gpus options: 0 | 0,1 | 0,1,2 | ... | 0,1,2,3,4,5,6,7
SOURCE_GPUS_LIST=("0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7")

# Mode names (for table columns)
MODES=("direct" "direct_alltoall" "allow_nvlink" "allow_nvlink_reassembly" "allow_nvlink_reassembly_alltoall")

# Temp dir for per-test output only (staged here then appended to LOG_FILE; deleted at exit)
RESULTS_DIR=$(mktemp -d)
trap "rm -rf '$RESULTS_DIR'" EXIT

run_one_test() {
  local src_gpus="$1"
  local mode="$2"
  local recv_args="" send_args=""
  case "$mode" in
    direct)
      recv_args="--direct --source-gpus $src_gpus"
      send_args="--direct --source-gpus $src_gpus"
      ;;
    direct_alltoall)
      recv_args="--direct --all-to-all --source-gpus $src_gpus"
      send_args="--direct --all-to-all --source-gpus $src_gpus"
      ;;
    allow_nvlink)
      recv_args="--allow-nvlink --source-gpus $src_gpus"
      send_args="--allow-nvlink --source-gpus $src_gpus"
      ;;
    allow_nvlink_reassembly)
      recv_args="--allow-nvlink --reassembly --source-gpus $src_gpus"
      send_args="--allow-nvlink --reassembly --source-gpus $src_gpus"
      ;;
    allow_nvlink_reassembly_alltoall)
      recv_args="--allow-nvlink --reassembly --all-to-all --source-gpus $src_gpus"
      send_args="--allow-nvlink --reassembly --all-to-all --source-gpus $src_gpus"
      ;;
    *) echo "Unknown mode: $mode"; return 1 ;;
  esac

  local outfile="$RESULTS_DIR/${src_gpus}_${mode}.out"
  local bw=""

  if [[ -n "$CONTAINER_IMAGE" ]]; then
    srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$FIRST_NODE" --gres=gpu:8 bash -c "cd '$CONTAINER_WORKDIR' && ./multi_nic_receive_buffer $recv_args" >> "$LOG_FILE" 2>&1 &
  else
    srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$FIRST_NODE" --gres=gpu:8 ./multi_nic_receive_buffer $recv_args >> "$LOG_FILE" 2>&1 &
  fi
  local recv_pid=$!
  sleep 8

  if [[ -n "$CONTAINER_IMAGE" ]]; then
    srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$SECOND_NODE" --gres=gpu:8 bash -c "cd '$CONTAINER_WORKDIR' && ./multi_nic_send_buffer '$RECEIVER_IP' $send_args" > "$outfile" 2>&1
  else
    srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$SECOND_NODE" --gres=gpu:8 ./multi_nic_send_buffer "$RECEIVER_IP" $send_args > "$outfile" 2>&1
  fi
  if [[ $? -eq 0 ]]; then
    bw=$(grep -oE 'Total bandwidth: [0-9.]+' "$outfile" 2>/dev/null | sed 's/Total bandwidth: //' || true)
    bw=$(echo "$bw" | sed 's/\x1b\[[0-9;]*m//g')
  fi
  kill $recv_pid 2>/dev/null || true
  wait $recv_pid 2>/dev/null || true

  cat "$outfile" >> "$LOG_FILE"
  echo "$bw"
}

# Table header for results
echo "source_gpus	direct	direct_alltoall	allow_nvlink	allow_nvlink_reassembly	allow_nvlink_reassembly_alltoall" > "$RESULTS_FILE"

for src_gpus in "${SOURCE_GPUS_LIST[@]}"; do
  echo "" | tee -a "$LOG_FILE"
  echo "========== source-gpus: $src_gpus ==========" | tee -a "$LOG_FILE"
  row="$src_gpus"
  for mode in "${MODES[@]}"; do
    echo "  --- mode: $mode ---" | tee -a "$LOG_FILE"
    bw=$(run_one_test "$src_gpus" "$mode" || echo "")
    row="${row}	${bw:-}"
  done
  echo "$row" >> "$RESULTS_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== Results table (source_gpus x mode -> GB/s) ===" | tee -a "$LOG_FILE"
if command -v column &>/dev/null; then
  column -t -s $'\t' "$RESULTS_FILE" | tee -a "$LOG_FILE"
else
  cat "$RESULTS_FILE" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"
echo "Done. Full log: $LOG_FILE  Results: $RESULTS_FILE" | tee -a "$LOG_FILE"

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
# For non-direct modes (allow_nvlink*), also sweeps transport buffer: 8M, 16M, 32M, 64M, 128M, and full (default).
# Uses srun: receiver on first node, sender on second node.
#
#
# Container: Your host dir (BINDIR) is not visible inside the container, so we mount it at CONTAINER_WORKDIR
#            (default /workdir). Every srun does: cd CONTAINER_WORKDIR && ./multi_nic_* (run and output there).
#            Set CONTAINER_WORKDIR if your image uses a different path (e.g. /workspace).
#            Optional BIND_MOUNTS adds extra src:dst bind mounts (comma-separated).
# IB IP:    Set RECEIVER_IB_IP to the InfiniBand IP of the first node, or leave unset to auto-detect from the first interface whose name starts with "ib".
# Cooldown: Set POST_TEST_SLEEP (seconds) to wait after each test so ports are released before the next (default 5). Use 0 to disable.
# Receiver startup: RECEIVER_STARTUP_SLEEP = initial sleep after starting receiver (default 8). RECEIVER_PORT_WAIT = max seconds to wait
#   for receiver handshake port to become open before starting sender (default 60); set to 0 to skip wait and rely only on sleep.
#   RECEIVER_PORT = first handshake port (default 48515). If "Connection refused" appears, increase RECEIVER_STARTUP_SLEEP or RECEIVER_PORT_WAIT.
# Test name: Set TEST_NAME to a short label (e.g. "nvlink_fix") to include in output filenames so you can tell runs apart.
#
# Submit:   sbatch run_all_rdma_tests.sbatch
#           Output goes to BW_latency_tests/<jobid>_<date>_<testname>_<nodes>/ with rdma_tests_*.{log,txt,out,err}
# With container: CONTAINER_IMAGE=/absolute/path/to/image.sqsh sbatch run_all_rdma_tests.sbatch
#           We mount your perftest dir at /workdir in the container and run "cd /workdir && ./multi_nic_*" there.
#           If the job script itself runs in a container with host mounted at /workdir, set: BINDIR=/workdir OUTPUT_DIR=/workdir

# It instructs the shell to immediately terminate the script if any command returns a non-zero exit status (which usually indicates an error).
set -e

# Where to find scripts. Base dir for output = submit dir.
# Submit from the perftest directory: cd /path/to/perftest && sbatch run_all_rdma_tests.sbatch
OUTPUT_BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"
# Directory containing test_rdma_multi_qp and multi_nic_* (for container mount and running). Default = same as output base.
BINDIR="${BINDIR:-$OUTPUT_BASE}"
BINDIR="$(cd "$BINDIR" && pwd)"
cd "$BINDIR"

# Container: we mount BINDIR into the container at CONTAINER_WORKDIR and run "cd CONTAINER_WORKDIR && ./multi_nic_*" there.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
# Resolve container image to absolute path so it works when job cwd differs from submit dir
if [[ -n "$CONTAINER_IMAGE" ]] && [[ "$CONTAINER_IMAGE" != /* ]]; then
  CONTAINER_IMAGE="$(cd "$(dirname "$CONTAINER_IMAGE")" 2>/dev/null && pwd)/$(basename "$CONTAINER_IMAGE")" || true
fi
# Path *inside the container* where we mount BINDIR and where we cd to run tests and write output (default /workdir).
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workdir}"
# Extra bind mounts only (comma-separated src:dst, e.g. /lustre:/lustre). BINDIR -> CONTAINER_WORKDIR is always added.
BIND_MOUNTS="${BIND_MOUNTS:-}"

# When using a container: bind-mount BINDIR at CONTAINER_WORKDIR so the same files are visible inside; every srun does cd CONTAINER_WORKDIR then run.
# Bind mount = make a host directory visible at another path inside the container (e.g. /lustre/.../perftest -> /workdir).
# Pyxis (and SLURM) expect --container-image= (not --container=); then --container-mounts is applied.
SRUN_OPTS=()
if [[ -n "$CONTAINER_IMAGE" ]]; then
  SRUN_OPTS+=(--container-image="$CONTAINER_IMAGE")
  MOUNT_SRC="$BINDIR"
  MOUNT_DST="$CONTAINER_WORKDIR"
  if [[ -n "$BIND_MOUNTS" ]]; then
    SRUN_OPTS+=(--container-mounts="${MOUNT_SRC}:${MOUNT_DST},${BIND_MOUNTS}")
  else
    SRUN_OPTS+=(--container-mounts="${MOUNT_SRC}:${MOUNT_DST}")
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

# Log and results: write under BW_latency_tests/<jobid>_<date>_<testname>_<nodes>/
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
NODE_SUFFIX=$(echo "${FIRST_NODE}_${SECOND_NODE}" | tr -c '[:alnum:]_-' '_')
# Optional TEST_NAME is sanitized and added to filenames so you can distinguish runs
TEST_NAME_SANITIZED=""
if [[ -n "${TEST_NAME:-}" ]]; then
  TEST_NAME_SANITIZED=$(echo "${TEST_NAME}" | tr -c '[:alnum:]_-' '_' | sed 's/^_*\|_*$//g')
  [[ -n "$TEST_NAME_SANITIZED" ]] && FILE_SUFFIX="${JOB_ID}_${TIMESTAMP}_${TEST_NAME_SANITIZED}_${NODE_SUFFIX}" || FILE_SUFFIX="${JOB_ID}_${TIMESTAMP}_${NODE_SUFFIX}"
else
  FILE_SUFFIX="${JOB_ID}_${TIMESTAMP}_${NODE_SUFFIX}"
fi
OUTPUT_DIR="$OUTPUT_BASE/BW_latency_tests/$FILE_SUFFIX"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/rdma_tests_full_${FILE_SUFFIX}.log}"
RESULTS_FILE="${RESULTS_FILE:-$OUTPUT_DIR/rdma_tests_results_${FILE_SUFFIX}.txt}"
# Stdout/stderr in same name format (machine names + date) so .out and .err are easy to match to a run
SLURM_OUT="$OUTPUT_DIR/rdma_tests_${FILE_SUFFIX}.out"
SLURM_ERR="$OUTPUT_DIR/rdma_tests_${FILE_SUFFIX}.err"
exec > "$SLURM_OUT" 2> "$SLURM_ERR"

# Print where output goes
echo "OUTPUT_DIR (logs/results): $OUTPUT_DIR"
echo "BINDIR (scripts, container mount): $BINDIR"
[[ -n "${TEST_NAME:-}" ]] && echo "TEST_NAME (in filenames): $TEST_NAME"
echo "LOG_FILE: $LOG_FILE"
echo "RESULTS_FILE: $RESULTS_FILE"
echo "SLURM stdout: $SLURM_OUT"
echo "SLURM stderr: $SLURM_ERR"

# IB IP of first node (receiver). Set RECEIVER_IB_IP in the environment to override.
# Run discovery on the HOST (no container) so we see the node's real ib* interfaces.
if [[ -n "$RECEIVER_IB_IP" ]]; then
  RECEIVER_IP="$RECEIVER_IB_IP"
else
  RECEIVER_IP=$(srun -N 1 -n 1 -w "$FIRST_NODE" --gres=gpu:0 bash -c '
    for iface in /sys/class/net/ib*; do
      [[ -e "$iface" ]] || continue
      iface=$(basename "$iface")
      ip=$(ip -4 addr show "$iface" 2>/dev/null | grep -oP "inet \K[0-9.]+" | head -1)
      if [[ -n "$ip" ]]; then echo "$ip"; exit 0; fi
    done
    hostname -I 2>/dev/null | awk "{print \$1}"
  ' 2>/dev/null | head -1 | tr -d ' \n')
  # Fallback: resolve first node hostname (may give IB or primary IP)
  if [[ -z "$RECEIVER_IP" ]] && command -v getent &>/dev/null; then
    RECEIVER_IP=$(getent hosts "$FIRST_NODE" 2>/dev/null | awk '{print $1}' | head -1)
  fi
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
# Single-step mode: one srun with 2 tasks (avoids receiver step being killed on clusters that allow only one step at a time)
# Default 1: one srun with two tasks avoids receiver step being killed when cluster allows only one step at a time.
USE_SINGLE_SRUN_STEP="${USE_SINGLE_SRUN_STEP:-1}"
# Sender waits this many seconds before connecting (single-step only; default 10). Increase if Connection refused or FPE.
RDMA_SENDER_DELAY="${RDMA_SENDER_DELAY:-10}"
echo "Use single srun step:  $USE_SINGLE_SRUN_STEP" | tee -a "$LOG_FILE"
echo "Sender delay (s):      $RDMA_SENDER_DELAY" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Source-gpus options: 0 | 0,1 | 0,1,2 | ... | 0,1,2,3,4,5,6,7
SOURCE_GPUS_LIST=("0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7")

# Mode names (for table columns)
MODES=("direct" "direct_alltoall" "allow_nvlink" "allow_nvlink_reassembly" "allow_nvlink_reassembly_alltoall")

# Transport buffer sizes for non-direct modes (8, 16, 32, 64, 128 MB and default full buffer)
# Empty string = default (no --transport-buffer, full buffer). Not used for direct modes.
TRANSPORT_BUFFER_SIZES=("8M" "16M" "32M" "64M" "128M" "")

# Temp dir for per-test output only (staged here then appended to LOG_FILE; deleted at exit)
RESULTS_DIR=$(mktemp -d)
RESULTS_TAB=""
trap 'rm -rf "$RESULTS_DIR" ${RESULTS_TAB:+"$RESULTS_TAB"}' EXIT

run_one_test() {
  local src_gpus="$1"
  local mode="$2"
  local transport_buf="${3:-}"
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

  # Add transport-buffer for non-direct modes when specified
  if [[ -n "$transport_buf" ]] && [[ "$mode" != "direct" ]] && [[ "$mode" != "direct_alltoall" ]]; then
    recv_args="${recv_args} --transport-buffer ${transport_buf}"
    send_args="${send_args} --transport-buffer ${transport_buf}"
  fi

  local tb_suffix=""
  [[ -n "$transport_buf" ]] && tb_suffix="_T${transport_buf}" || tb_suffix="_Tfull"
  local outfile="$RESULTS_DIR/${src_gpus}_${mode}${tb_suffix}.out"
  local bw=""
  local lat=""

  if [[ "${USE_SINGLE_SRUN_STEP:-0}" = "1" ]]; then
    # One srun with two tasks so only one job step (avoids receiver step being killed when second srun starts)
    local step_suffix=$(echo "${src_gpus}_${mode}${tb_suffix}" | tr ',' '_' | tr -c '[:alnum:]_.-' '_')
    local step_outfile_host="$BINDIR/.rdma_out_$$_${step_suffix}.out"
    local step_outfile_inside="${CONTAINER_WORKDIR:-$BINDIR}/.rdma_out_$$_${step_suffix}.out"
    if [[ -n "$CONTAINER_IMAGE" ]]; then
      export RDMA_STEP_OUTFILE="$step_outfile_inside"
    else
      export RDMA_STEP_OUTFILE="$step_outfile_host"
    fi
    export RDMA_RECV_ARGS="$recv_args"
    export RDMA_SEND_ARGS="$send_args"
    export RDMA_RECEIVER_IP="$RECEIVER_IP"
    export RDMA_CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-}"
    srun "${SRUN_OPTS[@]}" -N 2 -n 2 --ntasks-per-node=1 -w "$FIRST_NODE","$SECOND_NODE" --gres=gpu:8 bash -c '
      if [ "$SLURM_PROCID" = "0" ]; then
        [ -n "$RDMA_CONTAINER_WORKDIR" ] && cd "$RDMA_CONTAINER_WORKDIR"
        exec ./multi_nic_receive_buffer $RDMA_RECV_ARGS
      else
        [ -n "$RDMA_CONTAINER_WORKDIR" ] && cd "$RDMA_CONTAINER_WORKDIR"
        sleep "${RDMA_SENDER_DELAY:-10}"
        exec ./multi_nic_send_buffer "$RDMA_RECEIVER_IP" $RDMA_SEND_ARGS > "$RDMA_STEP_OUTFILE" 2>&1
      fi
    ' >> "$LOG_FILE" 2>&1
    if [[ -f "$step_outfile_host" ]]; then
      cp "$step_outfile_host" "$outfile"
      rm -f "$step_outfile_host"
    fi
  else
    if [[ -n "$CONTAINER_IMAGE" ]]; then
      srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$FIRST_NODE" --gres=gpu:8 bash -c "cd '$CONTAINER_WORKDIR' && ./multi_nic_receive_buffer $recv_args" >> "$LOG_FILE" 2>&1 &
    else
      srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$FIRST_NODE" --gres=gpu:8 ./multi_nic_receive_buffer $recv_args >> "$LOG_FILE" 2>&1 &
    fi
    local recv_pid=$!
    local startup_sleep="${RECEIVER_STARTUP_SLEEP:-8}"
    sleep "$startup_sleep"

    local port_wait="${RECEIVER_PORT_WAIT:-60}"
    local recv_port="${RECEIVER_PORT:-48515}"
    if [[ -n "$port_wait" ]] && [[ "$port_wait" -gt 0 ]]; then
      local waited=0
      while (( waited < port_wait )); do
        if (echo >/dev/tcp/"$RECEIVER_IP"/"$recv_port") 2>/dev/null; then
          break
        fi
        sleep 2
        waited=$((waited + 2))
      done
      if (( waited >= port_wait )); then
        echo "WARNING: Receiver port $RECEIVER_IP:$recv_port not open after ${port_wait}s; starting sender anyway." | tee -a "$LOG_FILE"
      fi
    fi

    if [[ -n "$CONTAINER_IMAGE" ]]; then
      srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$SECOND_NODE" --gres=gpu:8 bash -c "cd '$CONTAINER_WORKDIR' && ./multi_nic_send_buffer '$RECEIVER_IP' $send_args" > "$outfile" 2>&1
    else
      srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$SECOND_NODE" --gres=gpu:8 ./multi_nic_send_buffer "$RECEIVER_IP" $send_args > "$outfile" 2>&1
    fi
    kill $recv_pid 2>/dev/null || true
    wait $recv_pid 2>/dev/null || true
  fi

  if [[ -f "$outfile" ]]; then
    if [[ -r "$outfile" ]]; then
      bw=$(grep -oE 'Total bandwidth: [0-9.]+' "$outfile" 2>/dev/null | sed 's/Total bandwidth: //' || true)
      bw=$(echo "$bw" | sed 's/\x1b\[[0-9;]*m//g')
      lat=$(grep -oE 'Latency \(per iteration\): [0-9.]+ s \([0-9.]+ ms\)' "$outfile" 2>/dev/null | sed -E -n 's/.*\(([0-9.]+) ms\).*/\1/p' || true)
    fi
  fi

  local post_sleep="${POST_TEST_SLEEP:-5}"
  if [[ -n "$post_sleep" ]] && [[ "$post_sleep" -gt 0 ]]; then
    sleep "$post_sleep"
  fi

  cat "$outfile" >> "$LOG_FILE" 2>/dev/null || true
  echo "$bw	$lat"
}

# Build column headers: direct modes get 1 col each; allow_nvlink modes get 6 cols (8M, 16M, 32M, 64M, 128M, full)
COL_HEADERS="source_gpus"
for mode in "${MODES[@]}"; do
  if [[ "$mode" == "direct" ]] || [[ "$mode" == "direct_alltoall" ]]; then
    COL_HEADERS="${COL_HEADERS}	${mode}"
  else
    for tb in "${TRANSPORT_BUFFER_SIZES[@]}"; do
      [[ -n "$tb" ]] && col="${mode}_${tb}" || col="${mode}_full"
      COL_HEADERS="${COL_HEADERS}	${col}"
    done
  fi
done

# Table header for results (tab-separated; we format to a bordered table below)
RESULTS_TAB=$(mktemp)
RESULTS_TAB_LAT=$(mktemp)
trap 'rm -rf "$RESULTS_DIR" ${RESULTS_TAB:+"$RESULTS_TAB"} ${RESULTS_TAB_LAT:+"$RESULTS_TAB_LAT"}' EXIT
echo "$COL_HEADERS" > "$RESULTS_TAB"
echo "$COL_HEADERS" > "$RESULTS_TAB_LAT"

for src_gpus in "${SOURCE_GPUS_LIST[@]}"; do
  echo "" | tee -a "$LOG_FILE"
  echo "========== source-gpus: $src_gpus ==========" | tee -a "$LOG_FILE"
  row="$src_gpus"
  row_lat="$src_gpus"
  for mode in "${MODES[@]}"; do
    if [[ "$mode" == "direct" ]] || [[ "$mode" == "direct_alltoall" ]]; then
      echo "  --- mode: $mode ---" | tee -a "$LOG_FILE"
      result=$(run_one_test "$src_gpus" "$mode" "" || echo "")
      bw=$(echo "$result" | cut -f1)
      lat=$(echo "$result" | cut -f2)
      row="${row}	${bw:-}"
      row_lat="${row_lat}	${lat:-}"
    else
      for tb in "${TRANSPORT_BUFFER_SIZES[@]}"; do
        [[ -n "$tb" ]] && tb_label="$tb" || tb_label="full"
        echo "  --- mode: $mode transport-buffer: $tb_label ---" | tee -a "$LOG_FILE"
        result=$(run_one_test "$src_gpus" "$mode" "$tb" || echo "")
        bw=$(echo "$result" | cut -f1)
        lat=$(echo "$result" | cut -f2)
        row="${row}	${bw:-}"
        row_lat="${row_lat}	${lat:-}"
      done
    fi
  done
  echo "$row" >> "$RESULTS_TAB"
  echo "$row_lat" >> "$RESULTS_TAB_LAT"
done

# Format tab-separated table as bordered table (columns with | and separator line)
# Handles variable number of columns; col width 10 for data, 16 for source_gpus
format_results_table() {
  local f="$1"
  local unit="$2"
  {
    read -r line
    IFS=$'\t' read -ra COLS <<< "$line"
    local ncol=${#COLS[@]}
    local w1=16
    local w=12
    local fmt=""
    local sep=""
    for ((i=0; i<ncol; i++)); do
      [[ $i -eq 0 ]] && fmt=" %-*s" && sep="-----------------" || { fmt="${fmt} | %*s"; sep="${sep}+----------"; }
    done
    fmt="${fmt}\n"
    # Print header
    printf " %-*s" $((w1-1)) "${COLS[0]}"
    for ((i=1; i<ncol; i++)); do
      printf " | %*s" "$w" "${COLS[$i]}"
    done
    printf "\n"
    echo "$sep"
    # Print data rows
    while IFS=$'\t' read -ra COLS; do
      [[ -z "${COLS[0]}" ]] && continue
      printf " %-*s" $((w1-1)) "${COLS[0]}"
      for ((i=1; i<ncol; i++)); do
        local val="${COLS[$i]:---}"
        printf " | %*s" "$w" "$val"
      done
      printf "\n"
    done
    echo ""
    echo "All values in $unit."
  } < "$f"
}
{
  echo "=== Bandwidth (GB/s) ==="
  format_results_table "$RESULTS_TAB" "GB/s"
  echo ""
  echo "=== Latency (ms per iteration) ==="
  format_results_table "$RESULTS_TAB_LAT" "ms per iteration"
} > "$RESULTS_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Results table (source_gpus x mode -> BW in GB/s, latency in ms per iteration) ===" | tee -a "$LOG_FILE"
cat "$RESULTS_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Done. Full log: $LOG_FILE  Results: $RESULTS_FILE" | tee -a "$LOG_FILE"

#!/bin/bash
#SBATCH --job-name=rdma_subset
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=rdma_tests_%j.out
#SBATCH --error=rdma_tests_%j.err
#SBATCH --partition=interactive
#SBATCH --account=network_research_advdev

# RDMA test suite:
#   1. Direct all-to-all:          5 configs, no transport buffer
#   2. Allow-nvlink:               5 configs × 7 transport buffers × {dbl, nodbl}
#   3. Allow-nvlink reassembly a2a: 5 configs × 7 transport buffers × {dbl, nodbl}
#
# Configs (src_gpus | target_count):
#   0|8,  0,1|8,  0|4,  0,1|4,  0,1,2,3,4,5,6,7|(default 8)
#
# Submit: sbatch run_all_rdma_tests_subset.sbatch
#
# Resume interrupted run (skips completed configs, runs only missing; use same CONTAINER_IMAGE, TEST_NAME):
#   CONTAINER_IMAGE=... TEST_NAME=... RESUME_OUTPUT_DIR=/path/to/output_dir sbatch run_all_rdma_tests_subset.sbatch

set -e

OUTPUT_BASE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"
BINDIR="${BINDIR:-$OUTPUT_BASE}"
BINDIR="$(cd "$BINDIR" && pwd)"
cd "$BINDIR"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
if [[ -n "$CONTAINER_IMAGE" ]] && [[ "$CONTAINER_IMAGE" != /* ]]; then
  CONTAINER_IMAGE="$(cd "$(dirname "$CONTAINER_IMAGE")" 2>/dev/null && pwd)/$(basename "$CONTAINER_IMAGE")" || true
fi
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workdir}"
BIND_MOUNTS="${BIND_MOUNTS:-}"

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

JOB_ID="${SLURM_JOB_ID:-$$}"
NODELIST="${SLURM_NODELIST:-}"
if [[ -z "$NODELIST" ]]; then
  echo "ERROR: SLURM_NODELIST not set. Run this script under sbatch/srun."
  exit 1
fi
FIRST_NODE=$(scontrol show hostnames "$NODELIST" | head -1)
SECOND_NODE=$(scontrol show hostnames "$NODELIST" | tail -1)

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
NODE_SUFFIX=$(echo "${FIRST_NODE}_${SECOND_NODE}" | tr -c '[:alnum:]_-' '_')
if [[ -n "${TEST_NAME:-}" ]]; then
  TEST_NAME_SANITIZED=$(echo "${TEST_NAME}" | tr -c '[:alnum:]_-' '_' | sed 's/^_*\|_*$//g')
  [[ -n "$TEST_NAME_SANITIZED" ]] && FILE_SUFFIX="${JOB_ID}_${TIMESTAMP}_${TEST_NAME_SANITIZED}_${NODE_SUFFIX}" || FILE_SUFFIX="${JOB_ID}_${TIMESTAMP}_${NODE_SUFFIX}"
else
  FILE_SUFFIX="${JOB_ID}_${TIMESTAMP}_subset_${NODE_SUFFIX}"
fi
if [[ -n "${RESUME_OUTPUT_DIR:-}" ]]; then
  RESUME_OUTPUT_DIR="$(cd "$RESUME_OUTPUT_DIR" && pwd)"
  OUTPUT_DIR="$RESUME_OUTPUT_DIR"
  # Find log and results files (match rdma_tests_full_*.log pattern in that dir)
  LOG_FILE=$(ls "$OUTPUT_DIR"/rdma_tests_full_*.log 2>/dev/null | head -1)
  RESULTS_FILE=$(ls "$OUTPUT_DIR"/rdma_tests_results_*.txt 2>/dev/null | head -1)
  [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]] && { echo "ERROR: RESUME_OUTPUT_DIR must contain rdma_tests_full_*.log"; exit 1; }
  [[ -z "$RESULTS_FILE" || ! -f "$RESULTS_FILE" ]] && { echo "ERROR: RESUME_OUTPUT_DIR must contain rdma_tests_results_*.txt"; exit 1; }
  RESUME_MODE=1
  SLURM_OUT="$OUTPUT_DIR/rdma_tests_resume_${JOB_ID}.out"
  SLURM_ERR="$OUTPUT_DIR/rdma_tests_resume_${JOB_ID}.err"
else
  OUTPUT_DIR="$OUTPUT_BASE/BW_latency_tests/$FILE_SUFFIX"
  mkdir -p "$OUTPUT_DIR"
  LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/rdma_tests_full_${FILE_SUFFIX}.log}"
  RESULTS_FILE="${RESULTS_FILE:-$OUTPUT_DIR/rdma_tests_results_${FILE_SUFFIX}.txt}"
  RESUME_MODE=0
  SLURM_OUT="$OUTPUT_DIR/rdma_tests_${FILE_SUFFIX}.out"
  SLURM_ERR="$OUTPUT_DIR/rdma_tests_${FILE_SUFFIX}.err"
fi
exec > "$SLURM_OUT" 2> "$SLURM_ERR"

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
  if [[ -z "$RECEIVER_IP" ]] && command -v getent &>/dev/null; then
    RECEIVER_IP=$(getent hosts "$FIRST_NODE" 2>/dev/null | awk '{print $1}' | head -1)
  fi
fi
if [[ -z "$RECEIVER_IP" ]]; then
  echo "ERROR: Could not get receiver IB IP. Set RECEIVER_IB_IP or ensure an ib* interface has an address on the first node."
  exit 1
fi

echo "=== RDMA multi-QP test suite (subset) ===" | tee -a "$LOG_FILE"
echo "Configs: src=0 tc=8 | src=0,1 tc=8 | src=0 tc=4 | src=0,1 tc=4 | src=0..7 (default)" | tee -a "$LOG_FILE"
echo "Modes: direct_alltoall | allow_nvlink | allow_nvlink_reassembly_alltoall" | tee -a "$LOG_FILE"
echo "Transport buffers: 8M 16M 32M 64M 128M 256M 1G (each with and without --double-buffer)" | tee -a "$LOG_FILE"
echo "First node (receiver): $FIRST_NODE" | tee -a "$LOG_FILE"
echo "Second node (sender):  $SECOND_NODE" | tee -a "$LOG_FILE"
echo "Receiver IP:           $RECEIVER_IP" | tee -a "$LOG_FILE"
echo "Log file:              $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results file:          $RESULTS_FILE" | tee -a "$LOG_FILE"
USE_SINGLE_SRUN_STEP="${USE_SINGLE_SRUN_STEP:-1}"
RDMA_SENDER_DELAY="${RDMA_SENDER_DELAY:-10}"
echo "Use single srun step:  $USE_SINGLE_SRUN_STEP" | tee -a "$LOG_FILE"
echo "Sender delay (s):      $RDMA_SENDER_DELAY" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Configs: "src_gpus|target_count"  (target_count empty = no --target-count flag, uses all 8 NICs)
ALL_CONFIGS=(
  "0|8"
  "0,1|8"
  "0|4"
  "0,1|4"
  "0,1,2,3,4,5,6,7|"
)

TRANSPORT_BUFFERS=("8M" "16M" "32M" "64M" "128M" "256M" "1G")

# Resume: determine which configs to run and pre-populate results from existing data
COMPLETED_CONFIGS=()
SRC0_TC4_PARTIAL_BW=()
SRC0_TC4_PARTIAL_LAT=()
if [[ "$RESUME_MODE" = "1" ]]; then
  # Parse existing results file for completed config rows (tab format from pipe-separated)
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*(src[0-9,_]+_tc[0-9]+|src[0-9,_]+)[[:space:]]*\| ]] || continue
    row=$(echo "$line" | awk -F'|' '{
      gsub(/^[ \t]+|[ \t]+$/, "", $1);
      printf "%s", $1;
      for (i=2; i<=NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i);
        if ($i != "" && $i !~ /^-+$/) printf "\t%s", $i;
      }
      printf "\n";
    }')
    cfg=$(echo "$row" | cut -f1)
    COMPLETED_CONFIGS+=("$cfg")
  done < <(grep -E '^[[:space:]]*src' "$RESULTS_FILE" 2>/dev/null || true)

  # Parse log for partial src0_tc4: 29 tests per config, so src0_tc4 starts at index 58 (0-based)
  # 9 tests completed for src0_tc4: direct + 8M dbl/nodbl + 16M dbl/nodbl + 32M dbl/nodbl + 64M dbl/nodbl
  if grep -q "config: src0_tc4" "$LOG_FILE" 2>/dev/null; then
    mapfile -t ALL_BW < <(grep -oE 'Total bandwidth: [0-9.]+' "$LOG_FILE" 2>/dev/null | sed 's/Total bandwidth: //' | sed 's/\x1b\[[0-9;]*m//g')
    mapfile -t ALL_LAT < <(grep 'Latency (per iteration)' "$LOG_FILE" 2>/dev/null | sed -E 's/.*\(([0-9.]+) ms\).*/\1/')
    # src0_tc8 (29) + src0_1_tc8 (29) = 58, so src0_tc4 starts at 58
    for i in $(seq 58 66); do
      [[ -n "${ALL_BW[$i]:-}" ]] && SRC0_TC4_PARTIAL_BW+=("${ALL_BW[$i]}")
      [[ -n "${ALL_LAT[$i]:-}" ]] && SRC0_TC4_PARTIAL_LAT+=("${ALL_LAT[$i]}")
    done
  fi

  # Configs to run: those not in COMPLETED_CONFIGS, plus src0_tc4 if partial (need to complete)
  CONFIGS=()
  for cfg in "${ALL_CONFIGS[@]}"; do
    src_gpus="${cfg%%|*}"
    target_count="${cfg##*|}"
    [[ -n "$target_count" ]] && cfg_label="src${src_gpus//,/_}_tc${target_count}" || cfg_label="src${src_gpus//,/_}"
    if printf '%s\n' "${COMPLETED_CONFIGS[@]}" | grep -qxF "$cfg_label"; then
      : # skip, already done
    else
      CONFIGS+=("$cfg")
    fi
  done
  # If src0_tc4 has partial data, we need to run it (to complete the row)
  if [[ ${#SRC0_TC4_PARTIAL_BW[@]} -gt 0 ]] && [[ ${#SRC0_TC4_PARTIAL_BW[@]} -lt 29 ]]; then
    # Ensure 0|4 is in CONFIGS (might have been skipped if we mis-parsed)
    printf '%s\n' "${CONFIGS[@]}" | grep -qxF "0|4" || CONFIGS=("0|4" "${CONFIGS[@]}")
  fi

  echo "=== RESUME MODE ===" | tee -a "$LOG_FILE"
  echo "Resuming into: $OUTPUT_DIR" | tee -a "$LOG_FILE"
  echo "Completed configs: ${COMPLETED_CONFIGS[*]:-none}" | tee -a "$LOG_FILE"
  echo "Configs to run: ${CONFIGS[*]:-none}" | tee -a "$LOG_FILE"
  echo "src0_tc4 partial tests from log: ${#SRC0_TC4_PARTIAL_BW[@]}" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
else
  CONFIGS=("${ALL_CONFIGS[@]}")
fi

RESULTS_DIR=$(mktemp -d)
RESULTS_TAB=""
RESULTS_TAB_LAT=""
trap 'rm -rf "$RESULTS_DIR" ${RESULTS_TAB:+"$RESULTS_TAB"} ${RESULTS_TAB_LAT:+"$RESULTS_TAB_LAT"}' EXIT

# run_one_test mode src_gpus target_count transport_buf double_buf
#   mode:         direct_alltoall | allow_nvlink | allow_nvlink_reassembly_alltoall
#   target_count: "4", "8", or "" (no flag)
#   transport_buf: "8M" .. "1G", or "" (no piping; only for nvlink modes)
#   double_buf:   "1" (with) or "0" (without --double-buffer)
# Prints: "bw<TAB>lat"
run_one_test() {
  local mode="$1"
  local src_gpus="$2"
  local target_count="$3"
  local transport_buf="$4"
  local double_buf="$5"

  local recv_args="" send_args=""
  case "$mode" in
    direct_alltoall)
      recv_args="--direct --all-to-all --source-gpus $src_gpus"
      send_args="--direct --all-to-all --source-gpus $src_gpus"
      ;;
    allow_nvlink)
      recv_args="--allow-nvlink --source-gpus $src_gpus"
      send_args="--allow-nvlink --source-gpus $src_gpus"
      ;;
    allow_nvlink_reassembly_alltoall)
      recv_args="--allow-nvlink --reassembly --all-to-all --source-gpus $src_gpus"
      send_args="--allow-nvlink --reassembly --all-to-all --source-gpus $src_gpus"
      ;;
    *) echo "Unknown mode: $mode"; return 1 ;;
  esac

  if [[ -n "$target_count" ]]; then
    recv_args="${recv_args} --target-count ${target_count}"
    send_args="${send_args} --target-count ${target_count}"
  fi

  if [[ -n "$transport_buf" ]] && [[ "$mode" != "direct_alltoall" ]]; then
    recv_args="${recv_args} --transport-buffer ${transport_buf}"
    send_args="${send_args} --transport-buffer ${transport_buf}"
    if [[ "$double_buf" = "1" ]]; then
      recv_args="${recv_args} --double-buffer"
      send_args="${send_args} --double-buffer"
    fi
  fi

  # Build output filename
  local src_safe="${src_gpus//,/_}"
  local tc_suffix="${target_count:+_tc${target_count}}"
  local tb_suffix="${transport_buf:+_${transport_buf}}"
  local dbl_suffix=""
  if [[ -n "$transport_buf" ]] && [[ "$mode" != "direct_alltoall" ]]; then
    [[ "$double_buf" = "0" ]] && dbl_suffix="_nodbl"
  fi
  local outfile="$RESULTS_DIR/src${src_safe}_${mode}${tc_suffix}${tb_suffix}${dbl_suffix}.out"

  local bw="" lat=""

  if [[ "${USE_SINGLE_SRUN_STEP:-0}" = "1" ]]; then
    local step_suffix
    step_suffix=$(echo "src${src_safe}_${mode}${tc_suffix}${tb_suffix}${dbl_suffix}" | tr -c '[:alnum:]_.-' '_')
    local step_outfile_host="$OUTPUT_DIR/.rdma_out_$$_${step_suffix}.out"
    local step_outfile_inside
    if [[ -n "$CONTAINER_IMAGE" ]]; then
      # Container mounts BINDIR at CONTAINER_WORKDIR; use canonical paths so symlinks match
      local out_canon bindir_canon output_rel
      out_canon=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd -P) || out_canon="$OUTPUT_DIR"
      bindir_canon=$(cd "$BINDIR" 2>/dev/null && pwd -P) || bindir_canon="$BINDIR"
      if [[ "$out_canon" == "$bindir_canon"/* ]] || [[ "$out_canon" == "$bindir_canon" ]]; then
        output_rel="${out_canon#$bindir_canon}"
        [[ "$output_rel" != /* ]] && output_rel="/$output_rel"
        step_outfile_inside="${CONTAINER_WORKDIR}${output_rel}/.rdma_out_$$_${step_suffix}.out"
      else
        # OUTPUT_DIR not under BINDIR (e.g. resume with different path); write under CONTAINER_WORKDIR
        step_outfile_inside="${CONTAINER_WORKDIR}/.rdma_out_$$_${step_suffix}.out"
        step_outfile_host="$BINDIR/.rdma_out_$$_${step_suffix}.out"
      fi
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
    sleep "${RECEIVER_STARTUP_SLEEP:-8}"
    local port_wait="${RECEIVER_PORT_WAIT:-60}"
    local recv_port="${RECEIVER_PORT:-48515}"
    if [[ -n "$port_wait" ]] && [[ "$port_wait" -gt 0 ]]; then
      local waited=0
      while (( waited < port_wait )); do
        if (echo >/dev/tcp/"$RECEIVER_IP"/"$recv_port") 2>/dev/null; then break; fi
        sleep 2
        waited=$((waited + 2))
      done
    fi
    if [[ -n "$CONTAINER_IMAGE" ]]; then
      srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$SECOND_NODE" --gres=gpu:8 bash -c "cd '$CONTAINER_WORKDIR' && ./multi_nic_send_buffer '$RECEIVER_IP' $send_args" > "$outfile" 2>&1
    else
      srun "${SRUN_OPTS[@]}" -N 1 -n 1 -w "$SECOND_NODE" --gres=gpu:8 ./multi_nic_send_buffer "$RECEIVER_IP" $send_args > "$outfile" 2>&1
    fi
    kill $recv_pid 2>/dev/null || true
    wait $recv_pid 2>/dev/null || true
  fi

  if [[ -f "$outfile" ]] && [[ -r "$outfile" ]]; then
    bw=$(grep -oE 'Total bandwidth: [0-9.]+' "$outfile" 2>/dev/null | sed 's/Total bandwidth: //' || true)
    bw=$(echo "$bw" | sed 's/\x1b\[[0-9;]*m//g')
    lat=$(grep -oE 'Latency \(per iteration\): [0-9.]+ s \([0-9.]+ ms\)' "$outfile" 2>/dev/null | sed -E -n 's/.*\(([0-9.]+) ms\).*/\1/p' || true)
  fi

  sleep "${POST_TEST_SLEEP:-5}"
  cat "$outfile" >> "$LOG_FILE" 2>/dev/null || true
  echo "${bw}	${lat}"
}

# Build column headers
# Row key: "src|tc" label
# Columns: direct_alltoall | nvlink_8M nvlink_8M_nodbl ... | nvlink_ra_8M ...
COL_HEADERS="config"
COL_HEADERS="${COL_HEADERS}	direct_alltoall"
for tb in "${TRANSPORT_BUFFERS[@]}"; do
  COL_HEADERS="${COL_HEADERS}	nvlink_${tb}_dbl	nvlink_${tb}_nodbl"
done
for tb in "${TRANSPORT_BUFFERS[@]}"; do
  COL_HEADERS="${COL_HEADERS}	nvlink_ra_${tb}_dbl	nvlink_ra_${tb}_nodbl"
done

RESULTS_TAB=$(mktemp)
RESULTS_TAB_LAT=$(mktemp)
RESULTS_OUTPUT_DONE=""

format_results_table() {
  local f="$1" unit="$2"
  {
    read -r line
    IFS=$'\t' read -ra COLS <<< "$line"
    local ncol=${#COLS[@]} w1=18 w=14
    printf " %-*s" $((w1-1)) "${COLS[0]}"
    for ((i=1; i<ncol; i++)); do printf " | %*s" "$w" "${COLS[$i]}"; done
    printf "\n"
    local sep; sep=$(printf '%0.s-' $(seq 1 $w1))
    for ((i=1; i<ncol; i++)); do sep="${sep}+$(printf '%0.s-' $(seq 1 $((w+2))))"; done
    echo "$sep"
    while IFS=$'\t' read -ra COLS; do
      [[ -z "${COLS[0]}" ]] && continue
      printf " %-*s" $((w1-1)) "${COLS[0]}"
      for ((i=1; i<ncol; i++)); do printf " | %*s" "$w" "${COLS[$i]:---}"; done
      printf "\n"
    done
    echo ""
    echo "All values in $unit."
  } < "$f"
}

output_results_on_exit() {
  [[ -n "$RESULTS_OUTPUT_DONE" ]] && return 0
  {
    echo "=== Bandwidth (GB/s) ==="
    format_results_table "$RESULTS_TAB" "GB/s"
    echo ""
    echo "=== Latency (ms per iteration) ==="
    format_results_table "$RESULTS_TAB_LAT" "ms per iteration"
  } > "$RESULTS_FILE" 2>/dev/null || true
  echo "" | tee -a "$LOG_FILE"
  echo "=== Results table ===" | tee -a "$LOG_FILE"
  cat "$RESULTS_FILE" 2>/dev/null | tee -a "$LOG_FILE" || true
  echo "" | tee -a "$LOG_FILE"
  echo "Done. Full log: $LOG_FILE  Results: $RESULTS_FILE" | tee -a "$LOG_FILE"
  RESULTS_OUTPUT_DONE=1
}
trap 'output_results_on_exit; rm -rf "$RESULTS_DIR" ${RESULTS_TAB:+"$RESULTS_TAB"} ${RESULTS_TAB_LAT:+"$RESULTS_TAB_LAT"}' EXIT

echo "$COL_HEADERS" > "$RESULTS_TAB"
echo "$COL_HEADERS" > "$RESULTS_TAB_LAT"

# Resume: pre-populate with completed config rows from existing results
if [[ "$RESUME_MODE" = "1" ]] && [[ -f "$RESULTS_FILE" ]] && [[ ${#COMPLETED_CONFIGS[@]} -gt 0 ]]; then
  # Parse Bandwidth section (between "=== Bandwidth" and "All values" or "===")
  in_bw=0
  while IFS= read -r line; do
    [[ "$line" =~ "=== Bandwidth" ]] && in_bw=1 && continue
    [[ "$in_bw" = "1" ]] && { [[ "$line" =~ "All values" ]] || [[ "$line" =~ "===" ]]; } && in_bw=0
    [[ "$in_bw" = "1" ]] && [[ "$line" =~ ^[[:space:]]*src ]] || continue
    row=$(echo "$line" | awk -F'|' '{ gsub(/^[ \t]+|[ \t]+$/, "", $1); r=$1; for(i=2;i<=NF;i++){ gsub(/^[ \t]+|[ \t]+$/, "", $i); if($i!="" && $i!~/^-+$/) r=r"\t"$i } print r }')
    cfg=$(echo "$row" | cut -f1)
    if printf '%s\n' "${COMPLETED_CONFIGS[@]}" | grep -qxF "$cfg" 2>/dev/null; then
      echo "$row" >> "$RESULTS_TAB"
    fi
  done < "$RESULTS_FILE"
  # Parse Latency section
  in_lat=0
  while IFS= read -r line; do
    [[ "$line" =~ "=== Latency" ]] && in_lat=1 && continue
    [[ "$in_lat" = "1" ]] && { [[ "$line" =~ "All values" ]] || [[ "$line" =~ "===" ]]; } && in_lat=0
    [[ "$in_lat" = "1" ]] && [[ "$line" =~ ^[[:space:]]*src ]] || continue
    row=$(echo "$line" | awk -F'|' '{ gsub(/^[ \t]+|[ \t]+$/, "", $1); r=$1; for(i=2;i<=NF;i++){ gsub(/^[ \t]+|[ \t]+$/, "", $i); if($i!="" && $i!~/^-+$/) r=r"\t"$i } print r }')
    cfg=$(echo "$row" | cut -f1)
    if printf '%s\n' "${COMPLETED_CONFIGS[@]}" | grep -qxF "$cfg" 2>/dev/null; then
      echo "$row" >> "$RESULTS_TAB_LAT"
    fi
  done < "$RESULTS_FILE"
fi

for cfg in "${CONFIGS[@]}"; do
  src_gpus="${cfg%%|*}"
  target_count="${cfg##*|}"   # empty string when no target-count

  [[ -n "$target_count" ]] && cfg_label="src${src_gpus//,/_}_tc${target_count}" \
                            || cfg_label="src${src_gpus//,/_}"

  # src0_tc4 with partial data: use cached for first 9 tests
  USE_PARTIAL=0
  PARTIAL_IDX_FILE=""
  if [[ "$cfg_label" = "src0_tc4" ]] && [[ ${#SRC0_TC4_PARTIAL_BW[@]} -gt 0 ]]; then
    USE_PARTIAL=1
    PARTIAL_IDX_FILE=$(mktemp)
    echo "0" > "$PARTIAL_IDX_FILE"
  fi

  echo "" | tee -a "$LOG_FILE"
  echo "========== config: $cfg_label ==========" | tee -a "$LOG_FILE"

  row="$cfg_label"
  row_lat="$cfg_label"

  run_or_use_cached() {
    if [[ "$USE_PARTIAL" = "1" ]] && [[ -f "$PARTIAL_IDX_FILE" ]]; then
      local idx=$(cat "$PARTIAL_IDX_FILE")
      if [[ "$idx" -lt ${#SRC0_TC4_PARTIAL_BW[@]} ]]; then
        echo "${SRC0_TC4_PARTIAL_BW[$idx]}	${SRC0_TC4_PARTIAL_LAT[$idx]}"
        echo $((idx + 1)) > "$PARTIAL_IDX_FILE"
        return
      fi
    fi
    run_one_test "$1" "$2" "$3" "$4" "$5" || echo ""
  }

  # --- Direct all-to-all (no transport buffer) ---
  echo "  --- direct_alltoall ---" | tee -a "$LOG_FILE"
  result=$(run_or_use_cached "direct_alltoall" "$src_gpus" "$target_count" "" "0")
  bw=$(echo "$result" | cut -f1)
  lat=$(echo "$result" | cut -f2)
  row="${row}	${bw:-}"
  row_lat="${row_lat}	${lat:-}"

  # --- Allow-nvlink: each transport buffer with and without double-buffer ---
  for tb in "${TRANSPORT_BUFFERS[@]}"; do
    for dbl in 1 0; do
      [[ "$dbl" = "1" ]] && dbl_label="dbl" || dbl_label="nodbl"
      echo "  --- allow_nvlink tb=${tb} (${dbl_label}) ---" | tee -a "$LOG_FILE"
      result=$(run_or_use_cached "allow_nvlink" "$src_gpus" "$target_count" "$tb" "$dbl")
      bw=$(echo "$result" | cut -f1)
      lat=$(echo "$result" | cut -f2)
      row="${row}	${bw:-}"
      row_lat="${row_lat}	${lat:-}"
    done
  done

  # --- Allow-nvlink reassembly all-to-all: each transport buffer with and without double-buffer ---
  for tb in "${TRANSPORT_BUFFERS[@]}"; do
    for dbl in 1 0; do
      [[ "$dbl" = "1" ]] && dbl_label="dbl" || dbl_label="nodbl"
      echo "  --- allow_nvlink_reassembly_alltoall tb=${tb} (${dbl_label}) ---" | tee -a "$LOG_FILE"
      result=$(run_or_use_cached "allow_nvlink_reassembly_alltoall" "$src_gpus" "$target_count" "$tb" "$dbl")
      bw=$(echo "$result" | cut -f1)
      lat=$(echo "$result" | cut -f2)
      row="${row}	${bw:-}"
      row_lat="${row_lat}	${lat:-}"
    done
  done

  echo "$row" >> "$RESULTS_TAB"
  echo "$row_lat" >> "$RESULTS_TAB_LAT"
  [[ -n "$PARTIAL_IDX_FILE" ]] && rm -f "$PARTIAL_IDX_FILE"
done

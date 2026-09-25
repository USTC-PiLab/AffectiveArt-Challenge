#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
OUT="${OUT_DIR:-./results}"
TEST_DATA="${TEST_DATA:-./dataset/test/test.jsonl}"
CKPT="${CKPT:-./checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  echo "Usage: bash scripts/generate_all.sh [--num-gpus N | --devices ID[,ID...]]" >&2
  echo "Examples: --num-gpus 4 | --devices 1,3,5" >&2
}

devices=()
while (( $# > 0 )); do
  case "$1" in
    --num-gpus)
      if (( $# < 2 )) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then echo "--num-gpus must be a positive integer" >&2; exit 2; fi
      devices=(); for ((gpu=0; gpu<$2; gpu++)); do devices+=("$gpu"); done; shift 2 ;;
    --devices)
      if (( $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then echo "--devices must be comma-separated GPU IDs" >&2; exit 2; fi
      IFS="," read -r -a devices <<< "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
(( ${#devices[@]} > 0 )) || devices=(0)

mkdir -p "$OUT/images"
export PYTHONPATH="./tools:./diffsynth-studio:${PYTHONPATH:-}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false

buckets=(impressionism gongbi ukiyoe abstract baroque inkwash renaissance sovrealism)
task_buckets=(); task_suffixes=(); task_shards=(); task_indices=()
for bucket in "${buckets[@]}"; do
  if [[ "$bucket" == renaissance && ${#devices[@]} -ge 2 ]]; then
    task_buckets+=(renaissance renaissance); task_suffixes+=(_s0 _s1); task_shards+=(2 2); task_indices+=(0 1)
  else
    task_buckets+=("$bucket"); task_suffixes+=(""); task_shards+=(1); task_indices+=(0)
  fi
done

run_one() {
  local bucket="$1" device="$2" suffix="$3" shards="$4" shard_index="$5"
  "$PYTHON_BIN" ./tools/generate.py --test-data "$TEST_DATA" --checkpoint-dir "$CKPT" --out-dir "$OUT" --devices "$device" --buckets "$bucket" --num-shards "$shards" --shard-index "$shard_index" --steps "${STEPS:-50}" --cfg-scale "${CFG_SCALE:-4.0}"
}

active_pids=()
declare -A pid_task=() pid_device=()
available_devices=("${devices[@]}")
next_task=0
failed=0
start_tasks() {
  while (( next_task < ${#task_buckets[@]} && ${#available_devices[@]} > 0 )); do
    task="$next_task"
    device="${available_devices[0]}"
    available_devices=("${available_devices[@]:1}")
    run_one "${task_buckets[$task]}" "$device" "${task_suffixes[$task]}" "${task_shards[$task]}" "${task_indices[$task]}" &
    pid=$!
    active_pids+=("$pid")
    pid_task["$pid"]="$task"
    pid_device["$pid"]="$device"
    echo "[generate] started ${task_buckets[$task]} on GPU $device (pid=$pid)"
    next_task=$((next_task+1))
  done
}
start_tasks
while (( ${#active_pids[@]} > 0 )); do
  finished_pid=""; if wait -n -p finished_pid "${active_pids[@]}"; then code=0; else code=$?; fi
  [[ -n "$finished_pid" ]] || { finished_pid="${active_pids[0]}"; if wait "$finished_pid"; then code=0; else code=$?; fi; }
  task="${pid_task[$finished_pid]}"
  if (( code == 0 )); then echo "[generate] finished ${task_buckets[$task]}"; else echo "[generate] failed ${task_buckets[$task]} (exit=$code)" >&2; failed=$((failed+1)); fi
  remaining=(); for pid in "${active_pids[@]}"; do [[ "$pid" == "$finished_pid" ]] || remaining+=("$pid"); done
  active_pids=("${remaining[@]}")
  available_devices+=("${pid_device[$finished_pid]}")
  unset "pid_task[$finished_pid]" "pid_device[$finished_pid]"
  start_tasks
done
(( failed == 0 )) || { echo "[generate] $failed task(s) failed" >&2; exit 1; }

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
SCRIPT_DIR="./scripts"
TRAIN_DIR="${TRAIN_DIR:-./dataset/train/text}"
PORT_BASE="${PORT_BASE:-29531}"

usage() {
  echo "Usage: bash scripts/train_multi.sh bucket:gpu[,gpu...] [bucket:gpu[,gpu...] ...]" >&2
  echo "Example: bash scripts/train_multi.sh gongbi:0 ukiyoe:1 baroque:2,3" >&2
}

if (( $# == 0 )); then
  usage
  exit 2
fi
if [[ ! "${PORT_BASE}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "PORT_BASE must be a nonnegative integer" >&2
  exit 2
fi

declare -a buckets=() devices=() started=() active_pids=()
declare -A seen_buckets=() gpu_owner=() reserved_gpu=() active_bucket=() active_devices=()

for spec in "$@"; do
  if [[ "${spec}" != *:* ]]; then
    usage
    exit 2
  fi
  bucket="${spec%%:*}"
  device="${spec#*:}"
  if [[ ! "${bucket}" =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "Invalid bucket name: ${bucket}" >&2
    exit 2
  fi
  if [[ -n "${seen_buckets[${bucket}]:-}" ]]; then
    echo "Duplicate bucket: ${bucket}" >&2
    exit 2
  fi
  if [[ ! "${device}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "Invalid GPU list for ${bucket}: ${device}" >&2
    exit 2
  fi
  IFS=',' read -r -a gpu_list <<< "${device}"
  for i in "${!gpu_list[@]}"; do
    for ((j = 0; j < i; j++)); do
      if [[ "${gpu_list[i]}" == "${gpu_list[j]}" ]]; then
        echo "GPU ${gpu_list[i]} is repeated for ${bucket}" >&2
        exit 2
      fi
    done
  done
  metadata="${TRAIN_DIR}/train_${bucket}.jsonl"
  if [[ ! -f "${metadata}" ]]; then
    echo "Missing training data: ${metadata}" >&2
    exit 3
  fi
  seen_buckets["${bucket}"]=1
  buckets+=("${bucket}")
  devices+=("${device}")
done

remaining=${#buckets[@]}
failed=0

while (( remaining > 0 || ${#active_pids[@]} > 0 )); do
  reserved_gpu=()
  for index in "${!buckets[@]}"; do
    if [[ "${started[index]:-0}" == 1 ]]; then
      continue
    fi
    IFS=',' read -r -a gpu_list <<< "${devices[index]}"
    available=1
    for gpu in "${gpu_list[@]}"; do
      if [[ -n "${gpu_owner[${gpu}]:-}" || -n "${reserved_gpu[${gpu}]:-}" ]]; then
        available=0
        break
      fi
    done
    if (( ! available )); then
      for gpu in "${gpu_list[@]}"; do
        reserved_gpu["${gpu}"]=1
      done
      continue
    fi

    bucket="${buckets[index]}"
    device="${devices[index]}"
    port=$((PORT_BASE + gpu_list[0]))
    PORT="${port}" bash "${SCRIPT_DIR}/train.sh" "${bucket}" "${device}" &
    pid=$!
    active_pids+=("${pid}")
    active_bucket["${pid}"]="${bucket}"
    active_devices["${pid}"]="${device}"
    for gpu in "${gpu_list[@]}"; do
      gpu_owner["${gpu}"]="${pid}"
    done
    started[index]=1
    remaining=$((remaining - 1))
    echo "[multi] started ${bucket} on GPU ${device} (pid=${pid})"
  done

  if (( ${#active_pids[@]} == 0 )); then
    echo "No runnable jobs remain" >&2
    exit 1
  fi

  finished_pid=
  if wait -n -p finished_pid "${active_pids[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  if [[ -z "${finished_pid}" ]]; then
    finished_pid="${active_pids[0]}"
    if wait "${finished_pid}"; then
      exit_code=0
    else
      exit_code=$?
    fi
  fi

  bucket="${active_bucket[${finished_pid}]}"
  IFS=',' read -r -a gpu_list <<< "${active_devices[${finished_pid}]}"
  for gpu in "${gpu_list[@]}"; do
    unset "gpu_owner[${gpu}]"
  done
  unset "active_bucket[${finished_pid}]" "active_devices[${finished_pid}]"
  remaining_pids=()
  for pid in "${active_pids[@]}"; do
    if [[ "${pid}" != "${finished_pid}" ]]; then
      remaining_pids+=("${pid}")
    fi
  done
  active_pids=("${remaining_pids[@]}")

  if (( exit_code == 0 )); then
    echo "[multi] finished ${bucket}"
  else
    echo "[multi] failed ${bucket} (exit=${exit_code})" >&2
    failed=$((failed + 1))
  fi
done

if (( failed > 0 )); then
  echo "[multi] ${failed} training job(s) failed" >&2
  exit 1
fi
echo "[multi] all training jobs finished"

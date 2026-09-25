#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
ZIMAGE_ROOT="${ZIMAGE_ROOT:-./model/Zimage}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
BUCKET="${1:?bucket is required}"
DEVICE="${2:-0}"
IFS=',' read -r -a DEVICE_LIST <<< "${DEVICE}"
NUM_PROCESSES="${NUM_PROCESSES:-${#DEVICE_LIST[@]}}"

if [[ ! "${BUCKET}" =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "Invalid bucket name: ${BUCKET}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-./checkpoints}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
SAVE_STEPS="${SAVE_STEPS:-}"
DATASET_WORKERS="${DATASET_WORKERS:-4}"

METADATA="./dataset/train/text/train_${BUCKET}.jsonl"
TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT:-./dataset/train/dataset}"

if [[ ! -f "${METADATA}" ]]; then
  echo "Missing training data: ${METADATA}" >&2
  exit 3
fi

MODEL_PATHS="[[\"${ZIMAGE_ROOT}/transformer/diffusion_pytorch_model-00001-of-00002.safetensors\",\"${ZIMAGE_ROOT}/transformer/diffusion_pytorch_model-00002-of-00002.safetensors\"],[\"${ZIMAGE_ROOT}/text_encoder/model-00001-of-00003.safetensors\",\"${ZIMAGE_ROOT}/text_encoder/model-00002-of-00003.safetensors\",\"${ZIMAGE_ROOT}/text_encoder/model-00003-of-00003.safetensors\"],\"${ZIMAGE_ROOT}/vae/diffusion_pytorch_model.safetensors\"]"

export PYTHONPATH="./diffsynth-studio:${PYTHONPATH:-}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${DEVICE}"

PORT="${PORT:-$((29531 + ${DEVICE_LIST[0]}))}"
SAVE_ARGS=()
if [[ -n "${SAVE_STEPS}" ]]; then
  SAVE_ARGS+=(--save_steps "${SAVE_STEPS}")
fi

echo "[train] bucket=${BUCKET} device=${DEVICE} processes=${NUM_PROCESSES} grad_acc=${GRADIENT_ACCUMULATION_STEPS:-1} train_data=${METADATA} image_root=${TRAIN_IMAGE_ROOT} output=${OUT_DIR}/${BUCKET}"
"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "${PORT}" \
  "./diffsynth-studio/examples/z_image/model_training/train.py" \
  --dataset_base_path "${TRAIN_IMAGE_ROOT}" \
  --dataset_metadata_path "${METADATA}" \
  --data_file_keys "image" \
  --max_pixels 4194304 \
  --min_pixels 0 \
  --model_paths "${MODEL_PATHS}" \
  --tokenizer_path "${ZIMAGE_ROOT}/tokenizer" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUT_DIR}/${BUCKET}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --enable_resolution_dependent_timestep_sampling \
  --logit_normal_n_min 256 \
  --logit_normal_n_max 4096 \
  --logit_normal_mu_min 0.5 \
  --logit_normal_mu_max 1.15 \
  --logit_normal_std 1.0 \
  --dataset_num_workers "${DATASET_WORKERS}" \
  "${SAVE_ARGS[@]}" 

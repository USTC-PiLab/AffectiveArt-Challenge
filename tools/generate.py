from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "./diffsynth-studio")

import torch
from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

from common import BUCKETS, MODEL_ROOT, load_jsonl


def local_model_configs() -> list[ModelConfig]:
    transformer = sorted((MODEL_ROOT / "transformer").glob("diffusion_pytorch_model*.safetensors"))
    text_encoder = sorted((MODEL_ROOT / "text_encoder").glob("model-*.safetensors"))
    vae = MODEL_ROOT / "vae" / "diffusion_pytorch_model.safetensors"
    return [
        ModelConfig(path=[str(path) for path in transformer]),
        ModelConfig(path=[str(path) for path in text_encoder]),
        ModelConfig(path=str(vae)),
    ]


def load_test_data(path: Path) -> list[dict]:
    rows = load_jsonl(path)
    required = {"sample_id", "caption", "bucket", "predicted_aspect_ratio", "width", "height"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Missing fields in {path}: {sorted(missing)}")
    return rows


def bucket_rows(test_rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {bucket: [] for bucket in BUCKETS}
    for index, row in enumerate(test_rows):
        bucket = row["bucket"]
        grouped[bucket].append(
            {
                "index": index,
                "sample_id": row["sample_id"],
                "caption": row["caption"],
                "style_bucket": bucket,
                "aspect_ratio": row["predicted_aspect_ratio"],
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )
    return grouped


def generate_bucket(
    bucket: str,
    rows: list[dict],
    device: str,
    out_dir: str,
    checkpoint_dir: str,
    steps: int,
    cfg_scale: float,
    seed_base: int,
    quality: int,
    limit: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    runtime_device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except AttributeError:
        pass

    out_path = Path(out_dir)
    image_dir = out_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    bucket_checkpoint_dir = Path(checkpoint_dir) / bucket
    lora_path = bucket_checkpoint_dir / "epoch-2.safetensors"
    if not lora_path.exists():
        candidates = sorted(bucket_checkpoint_dir.glob("step-*.safetensors"))
        if candidates:
            lora_path = candidates[-1]
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing LoRA checkpoint for {bucket}: {lora_path}")

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=runtime_device,
        model_configs=local_model_configs(),
        tokenizer_config=ModelConfig(path=str(MODEL_ROOT / "tokenizer")),
        enable_npu_patch=False,
    )
    pipe.load_lora(pipe.dit, str(lora_path), alpha=1.0)

    selected_rows = rows[:limit] if limit else rows
    start = time.time()
    for done, row in enumerate(selected_rows, start=1):
        file_path = image_dir / f"{row['sample_id']}.jpg"
        if file_path.exists():
            status = "exists"
        else:
            image = pipe(
                prompt=row["caption"],
                height=row["height"],
                width=row["width"],
                seed=seed_base + row["index"],
                rand_device=runtime_device,
                num_inference_steps=steps,
                cfg_scale=cfg_scale,
                progress_bar_cmd=lambda value: value,
            )
            image.convert("RGB").save(file_path, format="JPEG", quality=quality, optimize=True)
            status = "generated"
        record = {
            **row,
            "file": file_path.relative_to(out_path).as_posix(),
            "status": status,
            "device": device,
            "steps": steps,
            "cfg_scale": cfg_scale,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if done % 5 == 0:
            elapsed = time.time() - start
            print(f"[generate:{bucket}] {done}/{len(selected_rows)} avg={elapsed / done:.1f}s", flush=True)
    print(f"[generate:{bucket}] done {len(selected_rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=Path, default=Path("./dataset/test/test.jsonl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--out-dir", type=Path, default=Path("./results"))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument("--buckets", default=",".join(BUCKETS))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed-base", type=int, default=53100)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--limit-per-bucket", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    test_rows = load_test_data(args.test_data)
    grouped = bucket_rows(test_rows)
    devices = [item for item in args.devices.split(",") if item]
    selected_buckets = [item for item in args.buckets.split(",") if item]
    for bucket in selected_buckets:
        if bucket not in BUCKETS:
            raise ValueError(f"Unknown bucket: {bucket}")
    if not devices:
        raise ValueError("At least one CUDA device must be provided.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn", force=True)
    processes = []
    for bucket_index, bucket in enumerate(selected_buckets):
        rows = [row for row_index, row in enumerate(grouped[bucket]) if row_index % args.num_shards == args.shard_index]
        device = devices[bucket_index % len(devices)]
        process = mp.Process(
            target=generate_bucket,
            args=(
                bucket,
                rows,
                device,
                str(args.out_dir),
                str(args.checkpoint_dir),
                args.steps,
                args.cfg_scale,
                args.seed_base,
                args.quality,
                args.limit_per_bucket,
            ),
        )
        process.start()
        processes.append(process)

    exit_code = 0
    for process in processes:
        process.join()
        if process.exitcode:
            exit_code = process.exitcode
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

from PIL import Image


ASPECT_CONFIG = json.loads(Path("./configs/ratio.json").read_text(encoding="utf-8"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "./model/Zimage"))

BUCKETS = (
    "gongbi",
    "ukiyoe",
    "abstract",
    "inkwash",
    "impressionism",
    "renaissance",
    "baroque",
    "sovrealism",
)

SHORT_EDGE_BY_BUCKET = {key: int(value) for key, value in ASPECT_CONFIG["short_edge"].items()}

ASPECT_RATIOS = {key: float(value) for key, value in ASPECT_CONFIG["candidate_ratios"].items()}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def portable_path(path: str | Path, base: Path = Path(".")) -> str:
    """Return a path suitable for metadata by making it relative to the project."""
    candidate = Path(path)
    if candidate.is_absolute():
        return os.path.relpath(candidate, base).replace(os.sep, "/")
    return candidate.as_posix()


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def nearest_aspect_bucket(width: int, height: int) -> str:
    ratio = width / height
    return min(ASPECT_RATIOS, key=lambda label: abs(math.log(ratio / ASPECT_RATIOS[label])))


def aspect_bucket_ratio(label: str) -> float:
    if label not in ASPECT_RATIOS:
        raise KeyError(f"Unknown aspect ratio bucket: {label}")
    return ASPECT_RATIOS[label]


def dimensions_for_bucket(style_bucket: str, aspect_label: str, multiple: int = 16) -> tuple[int, int]:
    short_edge = SHORT_EDGE_BY_BUCKET[style_bucket]
    ratio = aspect_bucket_ratio(aspect_label)
    if ratio >= 1:
        height = short_edge
        width = short_edge * ratio
    else:
        width = short_edge
        height = short_edge / ratio
    width = max(multiple, int(round(width / multiple)) * multiple)
    height = max(multiple, int(round(height / multiple)) * multiple)
    return width, height


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

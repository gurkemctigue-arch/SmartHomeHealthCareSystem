# -*- coding: utf-8 -*-
"""从明确不含药品的图片池挖掘误检，并构建固定验证集的困难负样本数据集。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_BASE = ROOT / "data" / "training" / "medicine_copypaste" / "medicine_copypaste.yaml"
DEFAULT_WEIGHTS = ROOT / "models" / "best_obb.pt"
DEFAULT_DESTINATION = ROOT / "data" / "training" / "medicine_hardneg_next"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建药盒 OBB 困难负样本数据集")
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--scan-conf", type=float, default=0.01)
    parser.add_argument("--select-conf", type=float, default=0.10)
    parser.add_argument("--max-fraction", type=float, default=0.03)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _read_yaml(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("r", encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream) or {}
    raw_root = Path(str(data.get("path", path.parent)))
    root = raw_root if raw_root.is_absolute() else path.parent / raw_root
    return data, root.resolve()


def _image_dir(data: dict[str, Any], root: Path, split: str) -> Path:
    value = data[split]
    if isinstance(value, (list, tuple)):
        raise ValueError(f"{split} 暂不支持多个路径")
    path = Path(str(value))
    return (path if path.is_absolute() else root / path).resolve()


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"图片路径中缺少 images 层级：{image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _clone_split(source_dir: Path, destination: Path, split: str) -> tuple[int, set[str]]:
    count = 0
    digests: set[str] = set()
    for image in _images(source_dir):
        relative = image.relative_to(source_dir)
        target_image = destination / split / "images" / relative
        target_label = destination / split / "labels" / relative.with_suffix(".txt")
        label = _label_for_image(image)
        _link_or_copy(image, target_image)
        if label.is_file():
            _link_or_copy(label, target_label)
        else:
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.touch()
        digests.add(_sha256(image))
        count += 1
    return count, digests


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.scan_conf <= args.select_conf <= 1:
        raise ValueError("需要满足 0 < scan-conf <= select-conf <= 1")
    if not 0 < args.max_fraction <= 0.10:
        raise ValueError("--max-fraction 必须在 (0, 0.10] 内")

    base_path = _absolute(args.base_data)
    weights = _absolute(args.weights)
    destination = _absolute(args.destination)
    candidate_dirs = [_absolute(path) for path in args.candidate_dir]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"目标目录非空：{destination}")
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")
    for path in candidate_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"候选目录不存在：{path}")

    data, base_root = _read_yaml(base_path)
    train_dir = _image_dir(data, base_root, "train")
    val_dir = _image_dir(data, base_root, "val")
    candidates = sorted({path for directory in candidate_dirs for path in _images(directory)})
    if not candidates:
        raise ValueError("候选目录中没有图片")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    detections: list[dict[str, Any]] = []
    # Some Ultralytics versions materialize a list source as one large CUDA
    # workload even when ``batch`` is small. Submit one image at a time so the
    # candidate pool cannot exhaust VRAM before the first result is consumed.
    for candidate_index, candidate in enumerate(candidates, start=1):
        results = model.predict(
            source=str(candidate),
            imgsz=args.imgsz,
            conf=args.scan_conf,
            batch=1,
            device=args.device,
            verbose=False,
        )
        if candidate_index == 1 or candidate_index % 50 == 0 or candidate_index == len(candidates):
            print(f"Scanning hard negatives: {candidate_index}/{len(candidates)}", flush=True)
        result = results[0]
        if result.obb is None or len(result.obb) == 0:
            continue
        confidences = result.obb.conf.detach().cpu().numpy()
        classes = result.obb.cls.detach().cpu().numpy().astype(int)
        index = int(confidences.argmax())
        score = float(confidences[index])
        if score >= args.select_conf:
            class_id = int(classes[index])
            detections.append(
                {
                    "source_image": str(Path(result.path).resolve()),
                    "max_conf": score,
                    "predicted_class_id": class_id,
                    "predicted_class_name": str(result.names[class_id]),
                    "prediction_count": len(result.obb),
                }
            )

    detections.sort(key=lambda row: float(row["max_conf"]), reverse=True)
    destination.mkdir(parents=True, exist_ok=True)
    train_count, base_hashes = _clone_split(train_dir, destination, "train")
    val_count, val_hashes = _clone_split(val_dir, destination, "valid")
    max_backgrounds = max(1, round(train_count * args.max_fraction))
    selected: list[dict[str, Any]] = []
    for row in detections:
        source = Path(str(row["source_image"]))
        digest = _sha256(source)
        if digest in base_hashes or digest in val_hashes:
            continue
        suffix = source.suffix.lower()
        name = f"hardneg_{digest[:16]}{suffix}"
        target_image = destination / "train" / "images" / name
        target_label = destination / "train" / "labels" / f"{Path(name).stem}.txt"
        _link_or_copy(source, target_image)
        target_label.touch()
        row = {**row, "output_image": str(target_image), "sha256": digest}
        selected.append(row)
        base_hashes.add(digest)
        if len(selected) >= max_backgrounds:
            break

    config = {
        "path": destination.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "nc": int(data.get("nc", len(data["names"]))),
        "names": data["names"],
    }
    output_yaml = destination / "medicine_hardneg.yaml"
    with output_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    report_path = destination / "hard_negative_report.csv"
    fieldnames = (
        "source_image",
        "output_image",
        "sha256",
        "max_conf",
        "predicted_class_id",
        "predicted_class_name",
        "prediction_count",
    )
    with report_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    decisions = base_root / "ocr_review_decisions.yaml"
    if decisions.is_file():
        shutil.copy2(decisions, destination / decisions.name)
    summary = {
        "base_data": str(base_path),
        "weights": str(weights),
        "candidate_images": len(candidates),
        "false_positive_candidates": len(detections),
        "selected_backgrounds": len(selected),
        "base_train_images": train_count,
        "validation_images": val_count,
        "output_yaml": str(output_yaml),
    }
    (destination / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

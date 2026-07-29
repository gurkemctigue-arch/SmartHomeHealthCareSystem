# -*- coding: utf-8 -*-
"""构建去重、纠错并增强稀有类的药盒 OBB 数据集。

默认保留原验证集，确保优化前后的 mAP50 可以直接比较。需要诊断全部类别时，
可显式使用 ``--split-policy stratified`` 重新分层划分。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = ROOT / "data" / "yolo" / "medicine" / "medicine_obb.yaml"
DEFAULT_CORRECTIONS = SCRIPT_DIR / "medicine_label_corrections.yaml"
DEFAULT_DESTINATION = ROOT / "data" / "training" / "medicine_map50_fixed"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Record:
    image: Path
    label: Path
    source_split: str
    digest: str
    labels: list[tuple[int, list[float]]]
    corrections: list[str] = field(default_factory=list)

    @property
    def classes(self) -> set[int]:
        return {class_id for class_id, _ in self.labels}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成面向 mAP50 的药盒 OBB 纠错增强数据集。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--split-policy",
        choices=("preserve", "stratified"),
        default="preserve",
        help="preserve 固定原验证集用于公平比较；stratified 重划全类别验证集",
    )
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--min-val-per-class", type=int, default=2)
    parser.add_argument("--min-train-per-class", type=int, default=2)
    parser.add_argument("--augment-classes", default="3,4,6,8,10,14,15,16")
    parser.add_argument("--rare-target", type=int, default=80, help="增强后稀有类训练目标数")
    parser.add_argument("--max-variants", type=int, default=6, help="每张原图最多增强版本数")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _class_ids(value: str, nc: int) -> set[int]:
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(class_id for class_id in result if not 0 <= class_id < nc)
    if invalid:
        raise ValueError(f"增强类别超出 0..{nc - 1}：{invalid}")
    return result


def _read_dataset(path: Path) -> tuple[dict[str, Any], dict[int, str], Path]:
    with path.open("r", encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream) or {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError(f"YAML 缺少 names：{path}")
    raw_root = Path(str(data.get("path", path.parent)))
    root = raw_root if raw_root.is_absolute() else path.parent / raw_root
    return data, names, root.resolve()


def _split_dirs(data: dict[str, Any], root: Path, split: str) -> tuple[Path, Path]:
    image_dir = Path(str(data[split]))
    image_dir = image_dir if image_dir.is_absolute() else root / image_dir
    image_dir = image_dir.resolve()
    parts = list(image_dir.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"图像路径中缺少 images 层级：{image_dir}")
    parts[indexes[-1]] = "labels"
    return image_dir, Path(*parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_labels(path: Path) -> list[tuple[int, list[float]]]:
    output: list[tuple[int, list[float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number} 不是合法 OBB 九列标签")
        output.append((int(values[0]), values[1:]))
    return output


def _write_labels(path: Path, labels: Sequence[tuple[int, Sequence[float]]]) -> None:
    lines = [
        f"{class_id} " + " ".join(f"{float(value):.6f}" for value in coords)
        for class_id, coords in labels
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_records(data: dict[str, Any], root: Path) -> list[Record]:
    records: list[Record] = []
    for split in ("train", "val"):
        image_dir, label_dir = _split_dirs(data, root, split)
        source_split = "valid" if split == "val" else "train"
        for image in sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            relative = image.relative_to(image_dir)
            label = (label_dir / relative).with_suffix(".txt")
            if not label.is_file():
                continue
            labels = _read_labels(label)
            if labels:
                records.append(Record(image, label, source_split, _sha256(image), labels))
    if not records:
        raise ValueError("没有找到带标签的训练/验证图像")
    return records


def _apply_corrections(
    records: Sequence[Record],
    path: Path,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        document = yaml.safe_load(stream) or {}
    by_key = {(record.source_split, record.image.name): record for record in records}
    report: list[dict[str, Any]] = []
    for item in document.get("corrections", []):
        key = (str(item["split"]), str(item["image"]))
        record = by_key.get(key)
        if record is None:
            raise FileNotFoundError(f"修正清单中的图像不存在：{key[0]}/{key[1]}")
        old_id, new_id = int(item["from"]), int(item["to"])
        changed = sum(1 for class_id, _ in record.labels if class_id == old_id)
        if not changed:
            raise ValueError(f"{key[0]}/{key[1]} 中没有待修正类别 {old_id}")
        record.labels = [
            (new_id if class_id == old_id else class_id, coords)
            for class_id, coords in record.labels
        ]
        tag = f"{old_id}->{new_id}"
        record.corrections.append(tag)
        report.append(
            {
                "source_split": key[0],
                "image": key[1],
                "from": old_id,
                "to": new_id,
                "objects_changed": changed,
                "evidence": str(item.get("evidence", "")),
            }
        )
    return report


def _deduplicate(records: Sequence[Record]) -> tuple[list[Record], list[dict[str, Any]]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.digest].append(record)
    unique: list[Record] = []
    report: list[dict[str, Any]] = []
    for digest, group in grouped.items():
        # 跨集合重复时保留验证样本并移除训练副本，避免评估数据泄漏。
        winner = max(
            group,
            key=lambda record: (
                record.source_split == "valid",
                bool(record.corrections),
                len(record.labels),
            ),
        )
        unique.append(winner)
        if len(group) > 1:
            signatures = {
                tuple((class_id, *(round(value, 5) for value in coords)) for class_id, coords in item.labels)
                for item in group
            }
            report.append(
                {
                    "sha256": digest,
                    "kept": str(winner.image),
                    "removed": "|".join(str(item.image) for item in group if item is not winner),
                    "copies": len(group),
                    "label_conflict": len(signatures) > 1,
                }
            )
    return sorted(unique, key=lambda record: (record.source_split, record.image.name)), report


def _stratified_split(
    records: Sequence[Record],
    nc: int,
    val_ratio: float,
    min_val: int,
    min_train: int,
    seed: int,
) -> tuple[list[Record], list[Record], dict[int, int]]:
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    totals = Counter(class_id for record in pool for class_id in record.classes)
    targets: dict[int, int] = {}
    for class_id in range(nc):
        total = totals[class_id]
        if total < 2:
            raise ValueError(f"类别 {class_id} 只有 {total} 张图，无法同时划分训练和验证")
        lower = min(min_val, max(1, total - min_train))
        targets[class_id] = min(max(round(total * val_ratio), lower), total - min_train)

    val: list[Record] = []
    val_counts: Counter[int] = Counter()

    def can_select(record: Record) -> bool:
        return all(totals[class_id] - val_counts[class_id] - 1 >= min_train for class_id in record.classes)

    while any(val_counts[class_id] < targets[class_id] for class_id in range(nc)):
        scored: list[tuple[float, float, Record]] = []
        for record in pool:
            if not can_select(record):
                continue
            score = sum(
                max(targets[class_id] - val_counts[class_id], 0) / max(1, targets[class_id])
                for class_id in record.classes
            )
            if score > 0:
                scored.append((score, rng.random(), record))
        if not scored:
            break
        selected = max(scored, key=lambda item: (item[0], item[1]))[2]
        pool.remove(selected)
        val.append(selected)
        val_counts.update(selected.classes)

    desired_val = round(len(records) * val_ratio)
    while len(val) < desired_val:
        candidates = [record for record in pool if can_select(record)]
        if not candidates:
            break

        def penalty(record: Record) -> tuple[float, float]:
            overshoot = sum(
                max(val_counts[class_id] + 1 - targets[class_id], 0)
                / max(1, targets[class_id])
                for class_id in record.classes
            )
            return overshoot, rng.random()

        selected = min(candidates, key=penalty)
        pool.remove(selected)
        val.append(selected)
        val_counts.update(selected.classes)

    train = pool
    train_counts = Counter(class_id for record in train for class_id in record.classes)
    missing = [class_id for class_id in range(nc) if not train_counts[class_id] or not val_counts[class_id]]
    if missing:
        raise RuntimeError(f"分层划分后仍有类别缺失：{missing}")
    return train, val, targets


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def _write_jpg(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"无法编码图像：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _ordered(coords: Sequence[float]) -> list[float]:
    points = np.asarray(coords, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    return np.clip(points[order], 0.0, 1.0).reshape(-1).astype(float).tolist()


def _transform(
    image: np.ndarray,
    labels: Sequence[tuple[int, Sequence[float]]],
    variant: int,
) -> tuple[np.ndarray, list[tuple[int, list[float]]], str]:
    mode = variant % 6
    transformed = [(class_id, list(coords)) for class_id, coords in labels]
    if mode == 0:
        return cv2.convertScaleAbs(image, alpha=0.85, beta=8), transformed, "exposure_dark"
    if mode == 1:
        bright = cv2.convertScaleAbs(image, alpha=1.12, beta=6)
        return cv2.GaussianBlur(bright, (3, 3), 0.35), transformed, "bright_mild_blur"

    if mode == 2:
        output = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mapper = lambda x, y: (1.0 - y, x)
        name = "rotate90"
    elif mode == 3:
        output = cv2.rotate(image, cv2.ROTATE_180)
        mapper = lambda x, y: (1.0 - x, 1.0 - y)
        name = "rotate180"
    elif mode == 4:
        output = cv2.flip(image, 1)
        mapper = lambda x, y: (1.0 - x, y)
        name = "flip_horizontal"
    else:
        output = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mapper = lambda x, y: (y, 1.0 - x)
        name = "rotate270"

    mapped_labels: list[tuple[int, list[float]]] = []
    for class_id, coords in labels:
        points = np.asarray(coords, dtype=np.float32).reshape(4, 2)
        mapped = np.asarray([mapper(float(x), float(y)) for x, y in points], dtype=np.float32)
        mapped_labels.append((class_id, _ordered(mapped.reshape(-1))))
    return output, mapped_labels, name


def _copy_split(
    records: Sequence[Record],
    destination: Path,
    split: str,
    provenance: list[dict[str, Any]],
) -> list[tuple[Record, Path]]:
    image_dir = destination / split / "images"
    label_dir = destination / split / "labels"
    copied: list[tuple[Record, Path]] = []
    for record in records:
        name = f"{record.source_split}_{record.image.stem}_{record.digest[:10]}{record.image.suffix.lower()}"
        target_image = image_dir / name
        target_label = label_dir / f"{Path(name).stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, target_image)
        _write_labels(target_label, record.labels)
        copied.append((record, target_image))
        provenance.append(
            {
                "output_image": str(target_image),
                "output_split": split,
                "source_image": str(record.image),
                "source_split": record.source_split,
                "sha256": record.digest,
                "augmented": False,
                "transform": "none",
                "corrections": "|".join(record.corrections),
            }
        )
    return copied


def _augment_training(
    copied: Sequence[tuple[Record, Path]],
    destination: Path,
    augment_classes: set[int],
    target: int,
    max_variants: int,
    provenance: list[dict[str, Any]],
) -> tuple[int, Counter[int]]:
    object_counts = Counter(class_id for record, _ in copied for class_id, _ in record.labels)
    uses: Counter[str] = Counter()
    added = 0
    while True:
        deficits = {
            class_id: target - object_counts[class_id]
            for class_id in augment_classes
            if object_counts[class_id] < target
        }
        if not deficits:
            break
        candidates = [
            (record, image)
            for record, image in copied
            if uses[record.digest] < max_variants and not record.classes.isdisjoint(deficits)
        ]
        if not candidates:
            break

        def value(item: tuple[Record, Path]) -> tuple[int, int, str]:
            record, _ = item
            score = sum(deficits.get(class_id, 0) for class_id, _ in record.labels)
            return score, -uses[record.digest], record.digest

        record, source_image = max(candidates, key=value)
        variant = uses[record.digest]
        image, labels, transform_name = _transform(_read_image(source_image), record.labels, variant)
        stem = f"aug_{record.digest[:12]}_{variant + 1}"
        target_image = destination / "train" / "images" / f"{stem}.jpg"
        target_label = destination / "train" / "labels" / f"{stem}.txt"
        _write_jpg(target_image, image)
        _write_labels(target_label, labels)
        object_counts.update(class_id for class_id, _ in labels)
        uses[record.digest] += 1
        added += 1
        provenance.append(
            {
                "output_image": str(target_image),
                "output_split": "train",
                "source_image": str(record.image),
                "source_split": record.source_split,
                "sha256": record.digest,
                "augmented": True,
                "transform": transform_name,
                "corrections": "|".join(record.corrections),
            }
        )
    return added, object_counts


def _distribution(
    destination: Path,
    split: str,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    label_dir = destination / split / "labels"
    objects: Counter[int] = Counter()
    images: Counter[int] = Counter()
    for label in label_dir.glob("*.txt"):
        labels = _read_labels(label)
        objects.update(class_id for class_id, _ in labels)
        images.update({class_id for class_id, _ in labels})
    return [
        {
            "split": split,
            "class_id": class_id,
            "class_name": names[class_id],
            "images": images[class_id],
            "objects": objects[class_id],
        }
        for class_id in names
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.05 <= args.val_ratio <= 0.5:
        raise ValueError("--val-ratio 必须在 [0.05, 0.5] 内")
    if min(args.min_val_per_class, args.min_train_per_class, args.rare_target, args.max_variants) < 1:
        raise ValueError("最小样本数、增强目标和最大变体数必须大于 0")

    data_path = _absolute(args.data)
    corrections_path = _absolute(args.corrections)
    destination = _absolute(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"目标目录非空，为避免覆盖请换一个 --destination：{destination}")
    destination.mkdir(parents=True, exist_ok=True)

    data, names, dataset_root = _read_dataset(data_path)
    records = _load_records(data, dataset_root)
    correction_report = _apply_corrections(records, corrections_path)
    with corrections_path.open("r", encoding="utf-8-sig") as stream:
        correction_document = yaml.safe_load(stream) or {}
    reviewed_keep_count = len(correction_document.get("reviewed_keep", []))
    records, duplicate_report = _deduplicate(records)
    if args.split_policy == "preserve":
        train = [record for record in records if record.source_split == "train"]
        val = [record for record in records if record.source_split == "valid"]
        targets = Counter(class_id for record in val for class_id in record.classes)
    else:
        train, val, targets = _stratified_split(
            records,
            len(names),
            args.val_ratio,
            args.min_val_per_class,
            args.min_train_per_class,
            args.seed,
        )

    provenance: list[dict[str, Any]] = []
    copied_train = _copy_split(train, destination, "train", provenance)
    _copy_split(val, destination, "valid", provenance)
    augment_classes = _class_ids(args.augment_classes, len(names))
    added, augmented_counts = _augment_training(
        copied_train,
        destination,
        augment_classes,
        args.rare_target,
        args.max_variants,
        provenance,
    )

    config = {
        "path": destination.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": [names[class_id] for class_id in range(len(names))],
    }
    output_name = (
        "medicine_map50_fixed.yaml"
        if args.split_policy == "preserve"
        else "medicine_map50_stratified.yaml"
    )
    output_yaml = destination / output_name
    with output_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

    distribution = _distribution(destination, "train", names) + _distribution(destination, "valid", names)
    _write_csv(
        destination / "label_corrections.csv",
        correction_report,
        ("source_split", "image", "from", "to", "objects_changed", "evidence"),
    )
    shutil.copy2(corrections_path, destination / "ocr_review_decisions.yaml")
    _write_csv(
        destination / "duplicates.csv",
        duplicate_report,
        ("sha256", "kept", "removed", "copies", "label_conflict"),
    )
    _write_csv(
        destination / "provenance.csv",
        provenance,
        (
            "output_image",
            "output_split",
            "source_image",
            "source_split",
            "sha256",
            "augmented",
            "transform",
            "corrections",
        ),
    )
    _write_csv(
        destination / "class_distribution.csv",
        distribution,
        ("split", "class_id", "class_name", "images", "objects"),
    )
    summary = {
        "source_data": str(data_path),
        "output_yaml": str(output_yaml),
        "split_policy": args.split_policy,
        "unique_source_images": len(records),
        "train_original_images": len(train),
        "validation_images": len(val),
        "augmented_train_images": added,
        "corrections": len(correction_report),
        "corrected_objects": sum(row["objects_changed"] for row in correction_report),
        "reviewed_keep": reviewed_keep_count,
        "ocr_conflicts_reviewed": len(correction_report) + reviewed_keep_count,
        "duplicate_groups_removed": len(duplicate_report),
        "validation_targets": dict(sorted(targets.items())),
        "augmented_object_counts": dict(sorted(augmented_counts.items())),
    }
    (destination / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== medicine_map50 dataset ==========")
    print(f"  划分策略     : {args.split_policy}")
    print(f"  原始去重图像 : {len(records)}")
    print(f"  train/val    : {len(train)} / {len(val)}")
    print(f"  标签修正     : {len(correction_report)} 条，{summary['corrected_objects']} 个目标")
    print(f"  OCR 冲突复核 : {summary['ocr_conflicts_reviewed']} 张（保留 {reviewed_keep_count} 张）")
    print(f"  稀有类增强   : {added} 张")
    print(f"  数据 YAML    : {output_yaml}")
    print(f"  分布报告     : {destination / 'class_distribution.csv'}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

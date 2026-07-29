# -*- coding: utf-8 -*-
"""Build a leakage-safe OBB Copy-Paste training dataset.

Only the training split is augmented. Donor objects are cut out with their
oriented box polygon, rotated mildly, and pasted without mirroring product
text. The validation split is cloned unchanged from the base dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_BASE = ROOT / "data" / "training" / "medicine_hardneg_next" / "medicine_hardneg.yaml"
DEFAULT_DESTINATION = ROOT / "data" / "training" / "medicine_copypaste_next"


@dataclass(frozen=True)
class Donor:
    image: Path
    class_id: int
    points: np.ndarray
    key: str


@dataclass(frozen=True)
class Background:
    image: Path
    labels: list[tuple[int, list[float]]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an OBB-aware Copy-Paste medicine dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--target-classes", default="3,6,8,11,12,14,16")
    parser.add_argument("--target-objects", type=int, default=120)
    parser.add_argument("--max-composites", type=int, default=180)
    parser.add_argument("--max-pastes-per-image", type=int, default=2)
    parser.add_argument("--max-uses-per-instance", type=int, default=4)
    parser.add_argument("--background-max-objects", type=int, default=3)
    parser.add_argument("--max-overlap", type=float, default=0.10)
    parser.add_argument("--rotation", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
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
        if len(value) != 1:
            raise ValueError(f"Multiple {split} paths are not supported")
        value = value[0]
    path = Path(str(value))
    return (path if path.is_absolute() else root / path).resolve()


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"Image path has no images directory: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _read_labels(path: Path) -> list[tuple[int, list[float]]]:
    if not path.is_file():
        return []
    output: list[tuple[int, list[float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid OBB label at {path}:{line_number}")
        output.append((int(values[0]), values[1:]))
    return output


def _write_labels(path: Path, labels: Sequence[tuple[int, Sequence[float]]]) -> None:
    lines = [
        f"{class_id} " + " ".join(f"{float(value):.6f}" for value in coords)
        for class_id, coords in labels
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _write_jpg(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _clone_split(source_dir: Path, destination: Path, split: str) -> int:
    count = 0
    for image in _images(source_dir):
        relative = image.relative_to(source_dir)
        target_image = destination / split / "images" / relative
        target_label = destination / split / "labels" / relative.with_suffix(".txt")
        _link_or_copy(image, target_image)
        label = _label_for_image(image)
        if label.is_file():
            _link_or_copy(label, target_label)
        else:
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.touch()
        count += 1
    return count


def _parse_classes(value: str, nc: int) -> set[int]:
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(item for item in result if not 0 <= item < nc)
    if invalid:
        raise ValueError(f"Class IDs outside 0..{nc - 1}: {invalid}")
    return result


def _pixel_points(coords: Sequence[float], width: int, height: int) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float32).reshape(4, 2).copy()
    points[:, 0] *= width
    points[:, 1] *= height
    return cv2.convexHull(points).reshape(-1, 2).astype(np.float32)


def _ordered_normalized(points: np.ndarray, width: int, height: int) -> list[float]:
    normalized = points.astype(np.float32).copy()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    center = normalized.mean(axis=0)
    angles = np.arctan2(normalized[:, 1] - center[1], normalized[:, 0] - center[0])
    normalized = normalized[np.argsort(angles)]
    return np.clip(normalized, 0.0, 1.0).reshape(-1).astype(float).tolist()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_training_pool(
    train_dir: Path,
    target_classes: set[int],
    background_max_objects: int,
) -> tuple[list[Donor], list[Background], Counter[int]]:
    donors: list[Donor] = []
    backgrounds: list[Background] = []
    counts: Counter[int] = Counter()
    for image_path in _images(train_dir):
        labels = _read_labels(_label_for_image(image_path))
        counts.update(class_id for class_id, _ in labels)
        is_original = not image_path.name.startswith(("aug_", "hardneg_", "copypaste_"))
        if is_original and len(labels) <= background_max_objects:
            backgrounds.append(Background(image=image_path, labels=labels))
        donor_labels = [item for item in labels if item[0] in target_classes]
        if not is_original or not donor_labels:
            continue
        image = _read_image(image_path)
        height, width = image.shape[:2]
        for instance_index, (class_id, coords) in enumerate(labels):
            if class_id not in target_classes:
                continue
            points = _pixel_points(coords, width, height)
            if abs(cv2.contourArea(points)) < 64:
                continue
            key = _sha256_text(f"{image_path.resolve()}:{instance_index}:{class_id}")[:20]
            donors.append(Donor(image=image_path, class_id=class_id, points=points, key=key))
    if not donors:
        raise ValueError("No eligible donor instances found")
    if not backgrounds:
        raise ValueError("No eligible background images found")
    return donors, backgrounds, counts


def _extract_and_transform(
    donor: Donor,
    target_width: int,
    target_height: int,
    rotation: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    source = _read_image(donor.image)
    x, y, width, height = cv2.boundingRect(donor.points.astype(np.float32))
    pad = 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(source.shape[1], x + width + pad), min(source.shape[0], y + height + pad)
    patch = source[y0:y1, x0:x1].copy()
    local_points = donor.points - np.asarray([x0, y0], dtype=np.float32)
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local_points).astype(np.int32), 255)

    max_scale = min(0.44 * target_width / patch.shape[1], 0.44 * target_height / patch.shape[0])
    scale = min(rng.uniform(0.75, 1.18), max_scale)
    if scale < 0.22:
        return None
    angle = rng.uniform(-rotation, rotation)
    center = ((patch.shape[1] - 1) / 2.0, (patch.shape[0] - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    corners = np.asarray(
        [[0, 0], [patch.shape[1], 0], [patch.shape[1], patch.shape[0]], [0, patch.shape[0]]],
        dtype=np.float32,
    )
    transformed_corners = cv2.transform(corners[None, :, :], matrix)[0]
    minimum = transformed_corners.min(axis=0)
    maximum = transformed_corners.max(axis=0)
    matrix[:, 2] += 2.0 - minimum
    output_width = max(1, int(math.ceil(maximum[0] - minimum[0] + 4)))
    output_height = max(1, int(math.ceil(maximum[1] - minimum[1] + 4)))
    transformed_patch = cv2.warpAffine(
        patch,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    transformed_mask = cv2.warpAffine(
        mask,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    transformed_points = cv2.transform(local_points[None, :, :], matrix)[0]
    if abs(cv2.contourArea(transformed_points)) < 100:
        return None
    return transformed_patch, transformed_mask, transformed_points


def _overlap_ratio(candidate: np.ndarray, existing: np.ndarray) -> float:
    candidate_hull = cv2.convexHull(candidate.astype(np.float32))
    existing_hull = cv2.convexHull(existing.astype(np.float32))
    intersection, _ = cv2.intersectConvexConvex(candidate_hull, existing_hull)
    area = max(abs(cv2.contourArea(candidate_hull)), 1.0)
    return float(intersection) / area


def _paste_donor(
    canvas: np.ndarray,
    donor: Donor,
    occupied: Sequence[np.ndarray],
    max_overlap: float,
    rotation: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = canvas.shape[:2]
    transformed = _extract_and_transform(donor, width, height, rotation, rng)
    if transformed is None:
        return None
    patch, mask, local_points = transformed
    if patch.shape[1] + 4 >= width or patch.shape[0] + 4 >= height:
        return None
    for _ in range(35):
        left = rng.randint(2, width - patch.shape[1] - 2)
        top = rng.randint(2, height - patch.shape[0] - 2)
        offset = np.asarray([left, top], dtype=np.float32)
        candidate = local_points + offset
        if any(_overlap_ratio(candidate, polygon) > max_overlap for polygon in occupied):
            continue
        region = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
        alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
        blended = patch.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
        canvas[top : top + patch.shape[0], left : left + patch.shape[1]] = np.clip(
            blended, 0, 255
        ).astype(np.uint8)
        return canvas, candidate
    return None


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min(
        args.target_objects,
        args.max_composites,
        args.max_pastes_per_image,
        args.max_uses_per_instance,
    ) < 1:
        raise ValueError("Count arguments must be positive")
    if not 0 <= args.max_overlap <= 0.5:
        raise ValueError("--max-overlap must be in [0, 0.5]")
    if not 0 <= args.rotation <= 45:
        raise ValueError("--rotation must be in [0, 45]")

    base_path = _absolute(args.base_data)
    destination = _absolute(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    data, root = _read_yaml(base_path)
    names = data["names"]
    nc = int(data.get("nc", len(names)))
    target_classes = _parse_classes(args.target_classes, nc)
    train_dir = _image_dir(data, root, "train")
    val_dir = _image_dir(data, root, "val")

    donors, backgrounds, object_counts = _load_training_pool(
        train_dir, target_classes, args.background_max_objects
    )
    donors_by_class = {
        class_id: [donor for donor in donors if donor.class_id == class_id]
        for class_id in target_classes
    }
    destination.mkdir(parents=True, exist_ok=True)
    train_count = _clone_split(train_dir, destination, "train")
    val_count = _clone_split(val_dir, destination, "valid")

    rng = random.Random(args.seed)
    uses: Counter[str] = Counter()
    generated_counts: Counter[int] = Counter()
    paste_report: list[dict[str, Any]] = []
    composites = 0
    failures = 0
    while composites < args.max_composites:
        deficits = {
            class_id: args.target_objects - object_counts[class_id] - generated_counts[class_id]
            for class_id in target_classes
            if object_counts[class_id] + generated_counts[class_id] < args.target_objects
        }
        available = {
            class_id: [
                donor
                for donor in donors_by_class[class_id]
                if uses[donor.key] < args.max_uses_per_instance
            ]
            for class_id in deficits
        }
        active = [class_id for class_id, pool in available.items() if pool]
        if not active:
            break

        background = rng.choice(backgrounds)
        canvas = _read_image(background.image)
        height, width = canvas.shape[:2]
        output_labels = [(class_id, list(coords)) for class_id, coords in background.labels]
        occupied = [_pixel_points(coords, width, height) for _, coords in output_labels]
        placed: list[tuple[Donor, np.ndarray]] = []
        for _ in range(args.max_pastes_per_image):
            deficits = {
                class_id: args.target_objects - object_counts[class_id] - generated_counts[class_id]
                for class_id in active
                if object_counts[class_id] + generated_counts[class_id] < args.target_objects
            }
            active = [class_id for class_id in active if deficits.get(class_id, 0) > 0]
            if not active:
                break
            weights = [float(deficits[class_id]) for class_id in active]
            class_id = rng.choices(active, weights=weights, k=1)[0]
            candidates = [
                donor
                for donor in available[class_id]
                if uses[donor.key] < args.max_uses_per_instance
                and donor.image.resolve() != background.image.resolve()
            ]
            rng.shuffle(candidates)
            result = None
            selected = None
            for donor in candidates[:20]:
                result = _paste_donor(
                    canvas.copy(), donor, occupied, args.max_overlap, args.rotation, rng
                )
                if result is not None:
                    selected = donor
                    break
            if result is None or selected is None:
                continue
            canvas, points = result
            occupied.append(points)
            output_labels.append((selected.class_id, _ordered_normalized(points, width, height)))
            placed.append((selected, points))
            generated_counts[selected.class_id] += 1
            uses[selected.key] += 1

        if not placed:
            failures += 1
            if failures >= 100:
                break
            continue
        failures = 0
        composites += 1
        output_name = f"copypaste_{composites:04d}.jpg"
        output_image = destination / "train" / "images" / output_name
        output_label = destination / "train" / "labels" / f"copypaste_{composites:04d}.txt"
        _write_jpg(output_image, canvas)
        _write_labels(output_label, output_labels)
        for donor, points in placed:
            paste_report.append(
                {
                    "output_image": str(output_image),
                    "background_image": str(background.image),
                    "donor_image": str(donor.image),
                    "donor_key": donor.key,
                    "class_id": donor.class_id,
                    "class_name": str(names[donor.class_id]),
                    "donor_use": uses[donor.key],
                    "polygon_pixels": " ".join(f"{value:.2f}" for value in points.reshape(-1)),
                }
            )

    config = {
        "path": destination.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "nc": nc,
        "names": names,
    }
    output_yaml = destination / "medicine_copypaste.yaml"
    with output_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    _write_csv(
        destination / "copypaste_report.csv",
        paste_report,
        (
            "output_image",
            "background_image",
            "donor_image",
            "donor_key",
            "class_id",
            "class_name",
            "donor_use",
            "polygon_pixels",
        ),
    )
    for metadata_name in ("ocr_review_decisions.yaml", "hard_negative_report.csv"):
        source = root / metadata_name
        if source.is_file():
            shutil.copy2(source, destination / metadata_name)
    summary = {
        "base_data": str(base_path),
        "output_yaml": str(output_yaml),
        "base_train_images": train_count,
        "validation_images": val_count,
        "eligible_donors": len(donors),
        "donors_by_class": {
            class_id: len(donors_by_class[class_id]) for class_id in sorted(target_classes)
        },
        "eligible_backgrounds": len(backgrounds),
        "target_classes": sorted(target_classes),
        "target_objects": args.target_objects,
        "composite_images": composites,
        "pasted_objects": sum(generated_counts.values()),
        "pasted_by_class": dict(sorted(generated_counts.items())),
        "final_target_class_counts": {
            class_id: object_counts[class_id] + generated_counts[class_id]
            for class_id in sorted(target_classes)
        },
        "max_donor_use": max(uses.values(), default=0),
        "mirroring": False,
    }
    (destination / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

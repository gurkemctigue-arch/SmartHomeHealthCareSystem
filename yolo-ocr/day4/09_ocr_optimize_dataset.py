# -*- coding: utf-8 -*-
"""用 OBB 透视矫正 + OCR 生成药盒标签质检报告。

该脚本只标记人工复核候选，不会自动修改 YOLO 标签。已有 OCR JSON 会直接复用，
缺失结果可用 ``--run-ocr`` 补齐。建议先补齐稀有类：

    python day4/09_ocr_optimize_dataset.py --run-ocr --ocr-classes 4,6,10,15
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml

from medicine_ocr import (
    MEDICINE_NAMES,
    load_ocr_json,
    read_obb_labels,
    run_obb_ocr,
    suggest_class,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = ROOT / "data" / "yolo" / "medicine" / "medicine_obb.yaml"
DEFAULT_REUSE_JSON = ROOT / "output" / "ocr_results" / "json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _disk_free(path: Path) -> int:
    return shutil.disk_usage(Path(path.anchor) if path.anchor else path).free


def _default_artifact_root() -> Path:
    configured = os.environ.get("YOLO_OBB_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser()
    if _disk_free(ROOT) >= 1024**3:
        return ROOT
    if os.name == "nt":
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            drive = Path(f"{letter}:/")
            if drive.exists() and _disk_free(drive) >= 1024**3:
                return drive / "ai-vision-training"
    return ROOT


def _class_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    invalid = sorted(class_id for class_id in result if not 0 <= class_id < len(MEDICINE_NAMES))
    if invalid:
        raise argparse.ArgumentTypeError(f"类别编号超出 0..17：{invalid}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 OCR 辅助的药盒 OBB 标签质检报告。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="药盒 OBB YAML")
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
    parser.add_argument("--reuse-json", type=Path, default=DEFAULT_REUSE_JSON, help="已有 OCR JSON 目录")
    parser.add_argument("--classes", type=_class_ids, default=None, help="仅报告指定类别，如 4,6,10,15")
    parser.add_argument(
        "--ocr-classes",
        type=_class_ids,
        default=None,
        help="仅对这些类别补跑 OCR；已有结果仍全部进入报告",
    )
    parser.add_argument("--run-ocr", action="store_true", help="对缺失 JSON 的图像运行 RapidOCR")
    parser.add_argument("--overwrite", action="store_true", help="重跑已有的新 OCR 缓存")
    parser.add_argument("--limit", type=int, default=None, help="最多新增多少张 OCR，便于试跑")
    parser.add_argument("--angles", default="0,180", help="透视矫正后尝试的文字方向")
    parser.add_argument("--min-score", type=int, default=6, help="关键词建议类别最低分")
    parser.add_argument("--min-margin", type=int, default=2, help="第一与第二候选最低分差")
    return parser.parse_args(argv)


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _dataset_train_dirs(yaml_path: Path) -> tuple[Path, Path]:
    with yaml_path.open("r", encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream) or {}
    raw_root = Path(os.path.expandvars(str(data.get("path", yaml_path.parent))))
    dataset_root = raw_root if raw_root.is_absolute() else yaml_path.parent / raw_root
    image_dir = Path(str(data["train"]))
    image_dir = image_dir if image_dir.is_absolute() else dataset_root / image_dir
    image_dir = image_dir.resolve()
    parts = list(image_dir.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"训练图像路径中没有 images 层级：{image_dir}")
    parts[indexes[-1]] = "labels"
    return image_dir, Path(*parts)


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _result_row(
    image_path: Path,
    label_path: Path,
    result: dict[str, Any],
    min_score: int,
    min_margin: int,
    result_path: Path,
) -> dict[str, Any]:
    labels = sorted({class_id for class_id, _ in read_obb_labels(label_path)})
    suggestion = suggest_class(result.get("text", ""), min_score=min_score, min_margin=min_margin)
    suggested = suggestion["class_id"]
    label_score = max((suggestion["scores"].get(class_id, 0) for class_id in labels), default=0)
    if result.get("line_count", 0) == 0:
        status = "ocr_failed"
    elif suggested in labels:
        status = "supported"
    elif suggested is not None and suggestion["score"] >= label_score + min_margin:
        status = "review"
    elif label_score >= min_score:
        status = "supported"
    elif suggestion["score"] > 0:
        status = "weak_signal"
    else:
        status = "no_signal"

    matched = suggestion["matched_keywords"]
    return {
        "status": status,
        "image": str(image_path),
        "label": str(label_path),
        "label_ids": ",".join(map(str, labels)),
        "label_names": "|".join(MEDICINE_NAMES[class_id] for class_id in labels),
        "suggested_id": "" if suggested is None else suggested,
        "suggested_name": "" if suggested is None else MEDICINE_NAMES[suggested],
        "suggested_score": suggestion["score"],
        "label_score": label_score,
        "score_margin": suggestion["margin"],
        "matched_keywords": "|".join(matched),
        "line_count": result.get("line_count", 0),
        "average_confidence": result.get("average_confidence", 0.0),
        "text": str(result.get("text", "")).replace("\r", " ").replace("\n", " | "),
        "ocr_json": str(result_path),
        "ocr_method": result.get("ocr_method", "existing"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = _absolute(args.data)
    artifact_root = _absolute(args.artifact_root)
    reuse_json = _absolute(args.reuse_json)
    output_root = artifact_root / "output" / "ocr"
    json_root = output_root / "json" / "train"
    report_path = artifact_root / "output" / "ocr_label_audit.csv"
    summary_path = artifact_root / "output" / "ocr_label_audit_summary.json"
    review_dir = output_root / "review"
    for directory in (json_root, report_path.parent, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_dir, label_dir = _dataset_train_dirs(data_path)
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    angles = tuple(int(value.strip()) for value in args.angles.split(",") if value.strip())
    engine = None
    added = 0
    rows: list[dict[str, Any]] = []
    selected_images = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(image_dir)
        label_path = (label_dir / relative).with_suffix(".txt")
        if not label_path.is_file():
            continue
        label_ids = {class_id for class_id, _ in read_obb_labels(label_path)}
        if args.classes is not None and label_ids.isdisjoint(args.classes):
            continue
        selected_images += 1

        cached_path = (json_root / relative).with_suffix(".json")
        legacy_path = reuse_json / f"{image_path.stem}.json"
        result_path = cached_path if cached_path.is_file() else legacy_path
        result = None
        if result_path.is_file() and not (args.overwrite and result_path == cached_path):
            result = load_ocr_json(result_path)
        elif args.run_ocr and (
            args.ocr_classes is None or not label_ids.isdisjoint(args.ocr_classes)
        ):
            if args.limit is not None and added >= args.limit:
                continue
            if engine is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError as exc:
                    raise RuntimeError(
                        "缺少 rapidocr-onnxruntime，请先在当前环境安装该包"
                    ) from exc
                engine = RapidOCR()
            result = run_obb_ocr(image_path, label_path, engine, angles=angles)
            _save_json(cached_path, result)
            result_path = cached_path
            added += 1
            print(f"OCR {added}: {image_path.name} -> {result['line_count']} 行")
        if result is None:
            continue

        row = _result_row(
            image_path,
            label_path,
            result,
            args.min_score,
            args.min_margin,
            result_path,
        )
        rows.append(row)
        if row["status"] == "review":
            target = review_dir / image_path.name
            if not target.exists():
                shutil.copy2(image_path, target)
            shutil.copy2(label_path, target.with_suffix(".txt"))

    fieldnames = list(rows[0]) if rows else ["status", "image"]
    with report_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["status"] for row in rows)
    summary = {
        "data": str(data_path),
        "selected_images": selected_images,
        "reported_images": len(rows),
        "new_ocr_images": added,
        "statuses": dict(statuses),
        "report": str(report_path),
        "review_dir": str(review_dir),
        "note": "OCR 结果仅用于人工质检，不会自动修改标签。",
    }
    _save_json(summary_path, summary)
    print("\n========== OCR 标签质检 ==========")
    print(f"  已有/新增报告: {len(rows)} / {added}")
    print(f"  状态统计     : {dict(statuses)}")
    print(f"  报告         : {report_path}")
    print(f"  复核图片     : {review_dir}")
    print("  注意：review 是候选，不是自动判错。")
    print("==================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

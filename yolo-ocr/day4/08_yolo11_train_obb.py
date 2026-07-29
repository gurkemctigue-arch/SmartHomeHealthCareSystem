# -*- coding: utf-8 -*-
"""训练 YOLO11-OBB 药盒检测模型。

脚本包含数据预检、硬件自适应、断点续训、最佳指标汇总和训练曲线导出。
当项目所在磁盘空间不足时，会自动把运行记录和最终产物写到其他可用磁盘，
避免训练到 epoch 结束后才因 ``last.pt`` 无法保存而失败。

示例（可在任意工作目录运行）：

    python day4/08_yolo11_train_obb.py
    python day4/08_yolo11_train_obb.py --dry-run
    python day4/08_yolo11_train_obb.py --resume auto
    python day4/08_yolo11_train_obb.py --artifact-root D:/ai-vision-training
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import shutil
import stat
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROJECT_DATA = ROOT / "data" / "yolo" / "medicine" / "medicine_obb.yaml"
BUNDLED_DATA = ROOT / "data" / "training" / "medicine_copypaste" / "medicine_copypaste.yaml"
FIXED_ENHANCED_DATA = Path(
    "D:/ai-vision-training/datasets/medicine_map50_fixed/medicine_map50_fixed.yaml"
)
ENHANCED_DATA = Path("D:/ai-vision-training/datasets/medicine_map50_v2/medicine_map50_v2.yaml")
PREFERRED_DATA = next(
    (
        path
        for path in (BUNDLED_DATA, FIXED_ENHANCED_DATA, ENHANCED_DATA, PROJECT_DATA)
        if path.is_file()
    ),
    PROJECT_DATA,
)
DEFAULT_DATA = Path(
    os.environ.get(
        "YOLO_OBB_DATA",
        str(PREFERRED_DATA),
    )
)
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11s-obb.pt"
DEFAULT_EPOCHS = 60
DEFAULT_IMGSZ = 640
MIN_RUN_FREE_BYTES = 1 * 1024**3
MIN_EXPORT_FREE_BYTES = 64 * 1024**2
IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _absolute(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _disk_free(path: Path) -> int:
    """获取 path 所在卷的可用字节数，path 尚未创建时也可用。"""
    if path.anchor:
        probe = Path(path.anchor)
    else:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
    return shutil.disk_usage(probe).free


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _default_artifact_root() -> Path:
    configured = os.environ.get("YOLO_OBB_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser()
    if _disk_free(ROOT) >= MIN_RUN_FREE_BYTES:
        return ROOT
    if os.name == "nt":
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            drive = Path(f"{letter}:/")
            if drive.exists() and _disk_free(drive) >= MIN_RUN_FREE_BYTES:
                return drive / "ai-vision-training"
    return ROOT


DEFAULT_ARTIFACT_ROOT = _default_artifact_root()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 YOLO11-OBB 药盒检测模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集 YAML")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="OBB 预训练权重")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="最大训练轮数")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=None, help="GPU 默认 -1 自动估算，CPU 默认 2")
    parser.add_argument("--device", default="auto", help="auto、0、0,1 或 cpu")
    parser.add_argument("--workers", type=int, default=None, help="Windows 默认 0")
    parser.add_argument("--patience", type=int, default=12, help="早停等待轮数，0 表示关闭")
    parser.add_argument(
        "--optimizer",
        choices=("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"),
        default="auto",
        help="优化器",
    )
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率相对 lr0 的比例")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--cls-loss", type=float, default=0.5, help="分类损失权重")
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    parser.add_argument("--save-period", type=int, default=5, help="检查点保存间隔，-1 只存 last/best")
    parser.add_argument(
        "--final-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练后用 best.pt 重跑验证并导出逐类别指标",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        metavar="N",
        help="Stop normally after validating and saving epoch N without changing the LR schedule",
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="训练集使用比例")
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        metavar="N",
        help="冻结模型前 N 个模块；两阶段微调可使用 11 冻结 YOLO11 Backbone",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="最终权重、曲线和默认 runs 的存储根目录",
    )
    parser.add_argument("--project", type=Path, default=None, help="运行目录，默认 artifact-root/runs/obb")
    parser.add_argument("--name", default="medicine_obb", help="运行名称")
    parser.add_argument("--exist-ok", action="store_true", help="允许复用同名运行目录")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="从 last.pt 续训；省略路径时自动查找最新有效断点",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 OBB 数据增强",
    )
    parser.add_argument("--degrees", type=float, default=20.0, help="随机旋转角度")
    parser.add_argument("--mosaic", type=float, default=0.25, help="Mosaic 概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp 概率")
    parser.add_argument(
        "--multi-scale",
        type=float,
        default=0.0,
        help="多尺度训练幅度，0 表示关闭",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("map50", "map50-95"),
        default="map50",
        help="早停及 best.pt 使用的验证指标",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用重复采样清单温和增加稀有类出现频率",
    )
    parser.add_argument(
        "--balance-max-repeat",
        type=int,
        default=6,
        help="单张稀有类图像最多重复次数",
    )
    parser.add_argument(
        "--ocr-audit",
        type=Path,
        default=None,
        help="OCR 标签质检 CSV；默认自动读取 artifact-root/output/ocr_label_audit.csv",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CUDA 默认开启，CPU 默认关闭",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅执行完整预检")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs 必须大于 0")
    if args.imgsz < 32:
        raise ValueError("--imgsz 必须至少为 32")
    if args.batch == 0:
        raise ValueError("--batch 不能为 0；使用 -1 可自动估算")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.epochs:
        raise ValueError("--stop-after must be between 1 and --epochs")
    if args.workers is not None and args.workers < 0:
        raise ValueError("--workers 不能小于 0")
    if args.patience < 0:
        raise ValueError("--patience 不能小于 0")
    if args.save_period == 0 or args.save_period < -1:
        raise ValueError("--save-period 应为 -1 或正整数")
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction 必须在 (0, 1] 内")
    if not 0 <= args.mosaic <= 1:
        raise ValueError("--mosaic 必须在 [0, 1] 内")
    if not 0 <= args.mixup <= 1:
        raise ValueError("--mixup 必须在 [0, 1] 内")
    if not 0 <= args.multi_scale <= 1:
        raise ValueError("--multi-scale 必须在 [0, 1] 内")
    if not 0 <= args.degrees <= 180:
        raise ValueError("--degrees 必须在 [0, 180] 内")
    if args.lr0 <= 0:
        raise ValueError("--lr0 必须大于 0")
    if not 0 < args.lrf <= 1:
        raise ValueError("--lrf 必须在 (0, 1] 内")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay 不能小于 0")
    if args.cls_loss <= 0:
        raise ValueError("--cls-loss 必须大于 0")
    if args.freeze is not None and args.freeze < 0:
        raise ValueError("--freeze 不能小于 0")
    if args.balance_max_repeat < 1:
        raise ValueError("--balance-max-repeat 必须大于 0")


def _verify_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".yolo_write_test_{os.getpid()}"
    try:
        probe.write_bytes(b"ok")
    except OSError as exc:
        raise OSError(f"目录不可写：{path}（{exc}）") from exc
    finally:
        probe.unlink(missing_ok=True)


def _prepare_storage(artifact_root: Path, project: Path) -> tuple[Path, Path, Path]:
    run_free = _disk_free(project)
    export_free = _disk_free(artifact_root)
    if run_free < MIN_RUN_FREE_BYTES:
        raise OSError(
            f"训练盘 {project.anchor} 仅剩 {_format_bytes(run_free)}，"
            f"至少需要 {_format_bytes(MIN_RUN_FREE_BYTES)}；请使用 --artifact-root 指定其他磁盘。"
        )
    if export_free < MIN_EXPORT_FREE_BYTES:
        raise OSError(
            f"产物盘 {artifact_root.anchor} 仅剩 {_format_bytes(export_free)}，"
            f"至少需要 {_format_bytes(MIN_EXPORT_FREE_BYTES)}。"
        )

    models_dir = artifact_root / "models"
    output_dir = artifact_root / "output"
    temp_dir = artifact_root / "tmp"
    for directory in (project, models_dir, output_dir, temp_dir):
        _verify_writable_directory(directory)

    if _disk_free(ROOT) < MIN_EXPORT_FREE_BYTES:
        for variable in ("TEMP", "TMP", "TMPDIR"):
            os.environ[variable] = str(temp_dir)
        matplotlib_config = temp_dir / "matplotlib"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_config)

    return (
        models_dir / "best_obb.pt",
        output_dir / "results_obb.csv",
        output_dir / "yolo_obb_curve.png",
    )


def _read_dataset(yaml_path: Path) -> tuple[dict[str, Any], dict[int, str], Path]:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"数据集 YAML 不存在：{yaml_path}")
    with yaml_path.open("r", encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream) or {}
    missing = [key for key in ("train", "val", "names") if key not in data]
    if missing:
        raise ValueError(f"数据集 YAML 缺少字段：{', '.join(missing)}")

    raw_names = data["names"]
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("names 必须是列表或字典")
    nc = int(data.get("nc", len(names)))
    if len(names) != nc or sorted(names) != list(range(nc)):
        raise ValueError(f"类别配置不连续：nc={nc}，names={sorted(names)}")

    raw_root = Path(os.path.expandvars(str(data.get("path", yaml_path.parent))))
    dataset_root = raw_root if raw_root.is_absolute() else yaml_path.parent / raw_root
    return data, names, dataset_root.resolve()


def _split_dirs(dataset_root: Path, value: Any, split: str) -> tuple[Path, Path]:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"预检暂不支持多个 {split} 路径")
        value = value[0]
    image_dir = Path(os.path.expandvars(str(value)))
    image_dir = image_dir if image_dir.is_absolute() else dataset_root / image_dir
    image_dir = image_dir.resolve()
    if image_dir.name.lower() != "images" and (image_dir / "images").is_dir():
        image_dir /= "images"

    parts = list(image_dir.parts)
    image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indexes:
        raise ValueError(f"{split} 路径中缺少 images 目录：{image_dir}")
    parts[image_indexes[-1]] = "labels"
    return image_dir, Path(*parts)


def audit_dataset(yaml_path: Path) -> None:
    """检查图像/标签配对、OBB 九列格式、数值范围和类别覆盖。"""
    data, names, root = _read_dataset(yaml_path)
    split_counts: dict[str, Counter[int]] = {}
    print("\n========== 数据集预检 ==========")
    for split in ("train", "val"):
        image_dir, label_dir = _split_dirs(root, data[split], split)
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"{split} 目录不存在：{image_dir} / {label_dir}")
        images = [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        labels = list(label_dir.rglob("*.txt"))
        image_keys = {path.relative_to(image_dir).with_suffix("") for path in images}
        label_keys = {path.relative_to(label_dir).with_suffix("") for path in labels}
        orphan = label_keys - image_keys
        if orphan:
            raise ValueError(f"{split} 有 {len(orphan)} 个标签找不到同名图像")

        counts: Counter[int] = Counter()
        issues: list[str] = []
        objects = 0
        for label in labels:
            for line_number, line in enumerate(
                label.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 9:
                    issues.append(f"{label.name}:{line_number} 不是 9 列")
                    continue
                try:
                    values = [float(field) for field in fields]
                except ValueError:
                    issues.append(f"{label.name}:{line_number} 含非数字字段")
                    continue
                class_id = int(values[0])
                if (
                    not all(math.isfinite(value) for value in values)
                    or values[0] != class_id
                    or not 0 <= class_id < len(names)
                    or any(not 0 <= value <= 1 for value in values[1:])
                ):
                    issues.append(f"{label.name}:{line_number} 类别或坐标非法")
                    continue
                counts[class_id] += 1
                objects += 1
        if issues:
            samples = "\n".join(f"  - {issue}" for issue in issues[:8])
            raise ValueError(f"{split} 有 {len(issues)} 个非法 OBB 标签：\n{samples}")
        split_counts[split] = counts
        print(
            f"  {split:5s}: {len(images)} 张图像，{len(labels)} 个标签文件，"
            f"{objects} 个目标，{len(counts)}/{len(names)} 类"
        )
        missing_labels = len(image_keys - label_keys)
        if missing_labels:
            print(f"         提示：{missing_labels} 张图像没有标签，将按背景处理")

    missing_val = [class_id for class_id in names if split_counts["val"][class_id] == 0]
    if missing_val:
        text = ", ".join(f"{class_id}:{names[class_id]}" for class_id in missing_val)
        print(f"  警告：验证集缺少类别 {text}；总 mAP 不能代表全部类别")
    train_counts = split_counts["train"]
    rare_limit = max(10, round(sum(train_counts.values()) * 0.005))
    rare = [class_id for class_id in names if 0 < train_counts[class_id] < rare_limit]
    if rare:
        text = ", ".join(
            f"{class_id}:{names[class_id]}={train_counts[class_id]}" for class_id in rare
        )
        print(f"  警告：稀有类别（少于 {rare_limit} 个目标）：{text}")
    present = [count for count in train_counts.values() if count]
    if present and max(present) / min(present) >= 10:
        print(f"  警告：类别最大/最小目标数约为 {max(present) / min(present):.1f}:1")
    print("================================\n")


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"图像路径中没有 images 层级：{image_path}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _build_balanced_training_yaml(
    data_path: Path,
    artifact_root: Path,
    max_repeat: int,
) -> tuple[Path, int, int]:
    """生成含重复图像路径的训练清单；原图和标签不会被复制或修改。"""
    data, _, dataset_root = _read_dataset(data_path)
    image_dir, _ = _split_dirs(dataset_root, data["train"], "train")
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    records: list[tuple[Path, set[int]]] = []
    class_counts: Counter[int] = Counter()
    for image_path in image_paths:
        label_path = _label_path_for_image(image_path)
        class_ids: set[int] = set()
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    class_id = int(float(line.split()[0]))
                    class_ids.add(class_id)
                    class_counts[class_id] += 1
        records.append((image_path, class_ids))

    positive_counts = sorted(count for count in class_counts.values() if count > 0)
    if not positive_counts:
        raise ValueError("训练集没有可用于类别均衡的目标")
    target = float(np.median(positive_counts))
    manifest_lines: list[str] = []
    repeat_histogram: Counter[int] = Counter()
    for image_path, class_ids in records:
        if class_ids:
            rarest_count = min(class_counts[class_id] for class_id in class_ids)
            repeat = min(max_repeat, max(1, math.ceil(math.sqrt(target / rarest_count))))
        else:
            repeat = 1
        repeat_histogram[repeat] += 1
        manifest_lines.extend([image_path.resolve().as_posix()] * repeat)

    derived_dir = artifact_root / "data"
    derived_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = derived_dir / "medicine_obb_balanced_train.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    derived = dict(data)
    derived["path"] = dataset_root.as_posix()
    derived["train"] = manifest_path.resolve().as_posix()
    derived_path = derived_dir / "medicine_obb_balanced.yaml"
    with derived_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(derived, stream, allow_unicode=True, sort_keys=False)
    print(
        f"类别均衡清单：{len(image_paths)} 张原图 -> {len(manifest_lines)} 个训练条目；"
        f"重复分布 {dict(sorted(repeat_histogram.items()))}"
    )
    return derived_path, len(image_paths), len(manifest_lines)


def _print_ocr_audit(report_path: Path, decisions_path: Path | None = None) -> None:
    if not report_path.is_file():
        print(f"OCR 质检：未找到 {report_path}，本次仅使用人工标签")
        return
    with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    statuses = Counter(row.get("status", "unknown") for row in rows)
    print(f"OCR 质检：{len(rows)} 张，状态 {dict(statuses)}")
    review_count = statuses.get("review", 0)
    resolved = 0
    if decisions_path is not None and decisions_path.is_file():
        with decisions_path.open("r", encoding="utf-8-sig") as stream:
            decisions = yaml.safe_load(stream) or {}
        corrected = len(decisions.get("corrections", []))
        kept = len(decisions.get("reviewed_keep", []))
        resolved = corrected + kept
        unresolved = max(0, review_count - resolved)
        print(
            f"  冲突处理：已改标 {corrected} 张，人工确认保留 {kept} 张，"
            f"未处理 {unresolved} 张"
        )
    if review_count > resolved:
        print(
            f"  警告：仍有 {review_count - resolved} 张 OCR/人工标签疑似冲突；"
            "训练仍保留人工标签，请先查看 OCR review 目录"
        )


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _series(rows: list[dict[str, str]], *candidates: str) -> list[float]:
    if not rows:
        return []
    key = next((candidate for candidate in candidates if candidate in rows[0]), None)
    if key is None:
        return []
    output: list[float] = []
    for row in rows:
        try:
            output.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            output.append(float("nan"))
    return output


def _metric_series(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    return {
        "precision": _series(rows, "metrics/precision(B)", "metrics/precision(O)", "metrics/precision"),
        "recall": _series(rows, "metrics/recall(B)", "metrics/recall(O)", "metrics/recall"),
        "mAP50": _series(rows, "metrics/mAP50(B)", "metrics/mAP50(O)", "metrics/mAP50"),
        "mAP50-95": _series(
            rows,
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(O)",
            "metrics/mAP50-95",
        ),
    }


def _epoch_axis(rows: list[dict[str, str]]) -> np.ndarray:
    epochs = _series(rows, "epoch")
    if not epochs:
        return np.arange(1, len(rows) + 1)
    offset = 1 if int(epochs[0]) == 0 else 0
    return np.asarray([int(value) + offset for value in epochs])


def print_final_metrics(
    csv_path: Path,
    selection_metric: str = "map50",
) -> dict[str, float | int | None]:
    rows = _read_results(csv_path)
    if not rows:
        print("results.csv 为空，无法汇总指标")
        return {}
    metrics = _metric_series(rows)
    ranking_name = "mAP50" if selection_metric == "map50" else "mAP50-95"
    ranking = metrics[ranking_name] or metrics["mAP50"]
    finite = [index for index, value in enumerate(ranking) if np.isfinite(value)]
    best = max(finite, key=lambda index: ranking[index]) if finite else len(rows) - 1
    values = {
        name: float(series[best]) if series and np.isfinite(series[best]) else None
        for name, series in metrics.items()
    }
    precision, recall = values["precision"], values["recall"]
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    epoch = int(_epoch_axis(rows)[best])

    def display(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "N/A"

    print(f"\n========== 最佳验证指标（按 {ranking_name}，epoch {epoch}）==========")
    print(f"  F1               : {display(f1)}")
    print(f"  Precision        : {display(precision)}")
    print(f"  Recall           : {display(recall)}")
    print(f"  mAP50            : {display(values['mAP50'])}")
    print(f"  mAP50-95         : {display(values['mAP50-95'])}")
    print("================================================\n")
    return {"epoch": epoch, "f1": f1, **values}


def _ema(values: list[float], smoothing: float = 0.25) -> np.ndarray:
    output = np.full(len(values), np.nan)
    current: float | None = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        current = value if current is None else smoothing * value + (1 - smoothing) * current
        output[index] = current
    return output


def plot_training_curves(csv_path: Path, output_path: Path) -> Path | None:
    rows = _read_results(csv_path)
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _epoch_axis(rows)
    metric_values = _metric_series(rows)
    train_losses = {
        "train box": _series(rows, "train/box_loss"),
        "train cls": _series(rows, "train/cls_loss"),
        "train dfl": _series(rows, "train/dfl_loss"),
        "train angle": _series(rows, "train/angle_loss"),
    }
    val_losses = {
        "val box": _series(rows, "val/box_loss"),
        "val cls": _series(rows, "val/cls_loss"),
        "val dfl": _series(rows, "val/dfl_loss"),
        "val angle": _series(rows, "val/angle_loss"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = (
        ("#1565c0", "#00897b", "#ef6c00", "#7b1fa2"),
        ("#1565c0", "#ef6c00", "#2e7d32", "#6a1b9a"),
        ("#c62828", "#ad1457", "#6a1b9a", "#00838f"),
    )

    def draw(axis: Any, values_map: dict[str, list[float]], palette: Sequence[str]) -> None:
        for (label, values), color in zip(values_map.items(), palette):
            if not values:
                continue
            axis.plot(x, values, color=color, linewidth=1, alpha=0.25)
            axis.plot(x, _ema(values), color=color, linewidth=2, label=label)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8, loc="best")

    draw(axes[0], metric_values, colors[0])
    axes[0].set(title="Validation metrics", ylabel="score", ylim=(0, 1.02))
    draw(axes[1], train_losses, colors[1])
    axes[1].set(title="Training losses", ylabel="loss")
    draw(axes[2], val_losses, colors[2])
    axes[2].set(title="Validation losses", ylabel="loss")
    fig.suptitle("YOLO11-OBB training curves (raw + EMA)", fontsize=13)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())
    return output_path


def export_validation_metrics(
    metrics: Any,
    names: dict[int, str],
    per_class_path: Path,
    summary_path: Path,
) -> None:
    """导出全部配置类别；验证集缺失类别保留空指标。"""
    box = metrics.box
    position = {int(class_id): index for index, class_id in enumerate(metrics.ap_class_index)}
    rows: list[dict[str, Any]] = []
    for class_id, class_name in names.items():
        index = position.get(class_id)
        present = index is not None
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "images": int(metrics.nt_per_image[class_id]),
                "instances": int(metrics.nt_per_class[class_id]),
                "precision": float(box.p[index]) if present else "",
                "recall": float(box.r[index]) if present else "",
                "f1": float(box.f1[index]) if present else "",
                "map50": float(box.ap50[index]) if present else "",
                "map50_95": float(box.ap[index]) if present else "",
            }
        )
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    with per_class_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overall = {
        key: float(value) if isinstance(value, (int, float, np.number)) else value
        for key, value in metrics.results_dict.items()
    }
    overall["evaluated_classes"] = len(position)
    overall["configured_classes"] = len(names)
    summary_path.write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cache_value(option: str) -> bool | str:
    return {"none": False, "ram": True, "disk": "disk"}[option]


def _find_resume(project: Path, name: str) -> Path:
    candidates = [project / name / "weights" / "last.pt"]
    candidates.extend(project.glob(f"{name}*/weights/last.pt"))
    valid = [path for path in candidates if path.is_file() and path.stat().st_size > 1024]
    if not valid:
        raise FileNotFoundError(f"{project} 中没有找到有效的 last.pt，无法续训")
    return max(valid, key=lambda path: path.stat().st_mtime).resolve()


def _resolve_resume(value: str | None, project: Path, name: str) -> Path | None:
    if value is None:
        return None
    checkpoint = _find_resume(project, name) if value == "auto" else _absolute(Path(value))
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 1024:
        raise FileNotFoundError(f"断点不存在或不完整：{checkpoint}")
    return checkpoint


def _device_and_batch(torch_module: Any, requested: str, batch: int | None) -> tuple[str, int, bool]:
    device = "0" if requested.lower() == "auto" and torch_module.cuda.is_available() else requested
    if requested.lower() == "auto" and not torch_module.cuda.is_available():
        device = "cpu"
    using_cuda = device.lower() != "cpu" and torch_module.cuda.is_available()
    if device.lower() != "cpu" and not using_cuda:
        raise RuntimeError(f"请求 device={device}，但 PyTorch 未检测到 CUDA")
    return device, batch if batch is not None else (-1 if using_cuda else 2), using_cuda


def _map50_trainer_class() -> type:
    """让 Ultralytics 的早停与 best.pt 真正按 mAP50 选择。"""
    from ultralytics.models.yolo.obb.train import OBBTrainer

    class Map50OBBTrainer(OBBTrainer):
        def validate(self):
            metrics, fallback_fitness = super().validate()
            if metrics is None:
                return metrics, fallback_fitness
            key = next(
                (
                    candidate
                    for candidate in ("metrics/mAP50(B)", "metrics/mAP50(O)", "metrics/mAP50")
                    if candidate in metrics
                ),
                None,
            )
            if key is None:
                raise KeyError(f"验证结果没有 mAP50：{sorted(metrics)}")
            score = float(metrics[key])
            if not np.isfinite(score):
                return metrics, fallback_fitness

            if not hasattr(self, "_best_map50"):
                history = _read_results(Path(self.csv)) if Path(self.csv).is_file() else []
                previous = [
                    value
                    for value in _metric_series(history)["mAP50"]
                    if np.isfinite(value)
                ]
                self._best_map50 = max(previous, default=float("-inf"))
            self._best_map50 = max(self._best_map50, score)
            self.best_fitness = self._best_map50
            return metrics, score

    return Map50OBBTrainer


def _training_args(
    args: argparse.Namespace,
    data: Path,
    project: Path,
    device: str,
    batch: int,
    amp: bool,
    resume: Path | None,
) -> dict[str, Any]:
    close_mosaic = min(10, max(1, args.epochs // 5)) if args.augment and args.mosaic else 0
    output: dict[str, Any] = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "workers": args.workers,
        "device": device,
        "project": str(project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "cls": args.cls_loss,
        "cos_lr": True,
        "warmup_epochs": min(3.0, max(1.0, args.epochs * 0.05)),
        "amp": amp,
        "cache": _cache_value(args.cache),
        "seed": args.seed,
        "deterministic": True,
        "fraction": args.fraction,
        "save": True,
        "save_period": args.save_period,
        "plots": False,
        "val": True,
        "degrees": args.degrees if args.augment else 0.0,
        "translate": 0.10 if args.augment else 0.0,
        "scale": 0.35 if args.augment else 0.0,
        "shear": 2.0 if args.augment else 0.0,
        "perspective": 0.0002 if args.augment else 0.0,
        "flipud": 0.05 if args.augment else 0.0,
        "fliplr": 0.50 if args.augment else 0.0,
        "hsv_h": 0.01 if args.augment else 0.0,
        "hsv_s": 0.40 if args.augment else 0.0,
        "hsv_v": 0.30 if args.augment else 0.0,
        "mosaic": args.mosaic if args.augment else 0.0,
        "mixup": args.mixup if args.augment else 0.0,
        "multi_scale": args.multi_scale if args.augment else 0.0,
        "auto_augment": None,
        "close_mosaic": close_mosaic,
    }
    if resume is not None:
        output["resume"] = str(resume)
    if args.freeze is not None:
        output["freeze"] = args.freeze
    return output


def _make_writable(path: Path) -> None:
    if path.exists() and not path.stat().st_mode & stat.S_IWRITE:
        path.chmod(path.stat().st_mode | stat.S_IWRITE)


def _copy_artifact(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"训练产物不存在或为空：{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _make_writable(destination)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    data_path = _absolute(args.data)
    weights_path = _absolute(args.weights)
    artifact_root = _absolute(args.artifact_root)
    project = _absolute(args.project) if args.project is not None else artifact_root / "runs" / "obb"
    if not weights_path.is_file():
        raise FileNotFoundError(f"未找到 OBB 预训练权重：{weights_path}")

    best_export, results_export, curve_export = _prepare_storage(artifact_root, project)
    audit_dataset(data_path)
    ocr_report = (
        _absolute(args.ocr_audit)
        if args.ocr_audit is not None
        else artifact_root / "output" / "ocr_label_audit.csv"
    )
    _print_ocr_audit(ocr_report, data_path.parent / "ocr_review_decisions.yaml")
    training_data = data_path
    balanced_entries: tuple[int, int] | None = None
    if args.balance_classes:
        training_data, original_count, entry_count = _build_balanced_training_yaml(
            data_path,
            artifact_root,
            args.balance_max_repeat,
        )
        balanced_entries = (original_count, entry_count)
    print("正在加载 PyTorch 与 Ultralytics，首次启动可能需要数秒……")
    import torch
    from ultralytics import YOLO

    device, batch, using_cuda = _device_and_batch(torch, str(args.device), args.batch)
    args.workers = (
        args.workers
        if args.workers is not None
        else (0 if os.name == "nt" else min(4, max(1, (os.cpu_count() or 2) // 2)))
    )
    amp = args.amp if args.amp is not None else using_cuda
    resume = _resolve_resume(args.resume, project, args.name)
    load_path = resume or weights_path
    model = YOLO(str(load_path))
    if model.task != "obb":
        raise ValueError(f"权重任务为 {model.task!r}，需要 OBB 权重：{load_path}")
    train_args = _training_args(args, training_data, project, device, batch, amp, resume)

    hardware = torch.cuda.get_device_name(0) if using_cuda else "CPU"
    print("========== 训练配置 ==========")
    print(f"  硬件       : {hardware}（device={device}）")
    print(f"  权重       : {load_path}")
    print(f"  原始数据   : {data_path}")
    print(f"  训练配置   : {training_data}")
    if balanced_entries is not None:
        print(f"  类别均衡   : {balanced_entries[0]} -> {balanced_entries[1]} 条")
    print(f"  epochs     : {args.epochs}（patience={args.patience}）")
    print(f"  imgsz/batch: {args.imgsz}/{batch}")
    print(f"  workers/AMP: {args.workers}/{amp}")
    print(f"  冻结模块   : {args.freeze if args.freeze is not None else 0}")
    print(
        f"  优化器     : {args.optimizer}，lr0={args.lr0:g}，lrf={args.lrf:g}，"
        f"weight_decay={args.weight_decay:g}"
    )
    print(f"  最佳模型指标: {args.selection_metric}")
    print(
        f"  增强       : degrees={train_args['degrees']}, mosaic={train_args['mosaic']}, "
        f"mixup={train_args['mixup']}, multi_scale={train_args['multi_scale']}"
    )
    print(f"  运行目录   : {project / args.name}")
    print(f"  最终产物   : {artifact_root}")
    print(f"  可用空间   : {_format_bytes(_disk_free(project))}")
    if artifact_root != ROOT:
        print(
            f"  存储迁移   : 项目盘仅剩 {_format_bytes(_disk_free(ROOT))}，"
            f"已自动改用 {artifact_root}"
        )
    print("==============================\n")
    if args.dry_run:
        print("预检通过：数据、磁盘、环境和 OBB 权重均可用；未开始训练。")
        return 0

    if args.stop_after is not None:
        stop_after = args.stop_after

        def stop_at_epoch_limit(trainer: Any) -> None:
            if trainer.epoch + 1 >= stop_after:
                trainer.stop = True

        model.add_callback("on_train_epoch_end", stop_at_epoch_limit)

    trainer_class = _map50_trainer_class() if args.selection_metric == "map50" else None
    if trainer_class is None:
        model.train(**train_args)
    else:
        model.train(trainer=trainer_class, **train_args)
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("训练结束但无法取得 trainer")
    save_dir = Path(trainer.save_dir)
    best = Path(trainer.best)
    last = Path(trainer.last)
    selected = best if best.is_file() and best.stat().st_size else last
    source_csv = save_dir / "results.csv"
    _copy_artifact(selected, best_export)
    if source_csv.is_file():
        _copy_artifact(source_csv, results_export)
        print_final_metrics(results_export, args.selection_metric)
        try:
            plot_training_curves(results_export, curve_export)
            print(f"训练曲线：{curve_export}")
        except Exception as exc:
            print(f"警告：训练完成，但曲线生成失败：{exc}")
    else:
        print(f"警告：未找到 {source_csv}")
    if args.final_val:
        print("正在用最佳权重重跑验证并导出逐类别指标……")
        validation = YOLO(str(best_export)).val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=batch,
            workers=args.workers,
            device=device,
            project=str(artifact_root / "output" / "validation"),
            name=args.name,
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        per_class_path = artifact_root / "output" / f"{args.name}_per_class.csv"
        validation_path = artifact_root / "output" / f"{args.name}_validation.json"
        export_validation_metrics(
            validation,
            {int(index): str(name) for index, name in validation.names.items()},
            per_class_path,
            validation_path,
        )
        print(f"逐类别指标：{per_class_path}")
        print(f"验证摘要：{validation_path}")
    print(f"训练运行目录：{save_dir}")
    print(f"最佳 OBB 权重：{best_export}")
    print(f"断点权重：{last}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

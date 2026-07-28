from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import yaml

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

ROOT = Path(__file__).resolve().parent
HERE = ROOT
YAML_PATH = ROOT / "data" / "medicine" / "medicine_obb.yaml"
OUTPUT_DIR = HERE / "output"
PRETRAINED_PATH = ROOT / "yolo11s-obb.pt"

EPOCHS = 40
IMAGE_SIZE = 960
GPU_BATCH_SIZE = 8
CPU_BATCH_SIZE = 1
EXPERIMENT_NAME = "result"
EXIST_OK = True
PATIENCE = 60
SEED = 42

EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / EXPERIMENT_NAME
BEST_MODEL_PATH = EXPERIMENT_OUTPUT_DIR / "weights" / "best.pt"
LAST_MODEL_PATH = EXPERIMENT_OUTPUT_DIR / "weights" / "last.pt"
RESULTS_CSV_PATH = EXPERIMENT_OUTPUT_DIR / "results.csv"
ULTRALYTICS_PLOT_PATH = EXPERIMENT_OUTPUT_DIR / "yolo_results.png"
CUSTOM_CURVE_PATH = EXPERIMENT_OUTPUT_DIR / "yolo_curve.png"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def configure_chinese_font() -> None:
    """尝试配置 Matplotlib 中文字体。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        installed = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"):
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        pass


def normalize_csv_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """去除CSV列名前后空格，兼容不同Ultralytics版本。"""
    return [{str(key).strip(): value for key, value in row.items()} for row in rows]


def get_metric_series(rows: list[dict[str, str]], candidates: Iterable[str]) -> list[float]:
    """按照候选列名读取指标序列。"""
    if not rows:
        return []

    key = next((candidate for candidate in candidates if candidate in rows[0]), None)
    if key is None:
        return []

    values: list[float] = []
    for row in rows:
        try:
            values.append(float(str(row.get(key, "")).strip()))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def smooth_curve(values: list[float], window_size: int = 5) -> list[float]:
    """移动平均平滑，自动处理NaN。"""
    if len(values) < 3:
        return values

    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    if not valid.any():
        return values

    if not valid.all():
        indexes = np.where(valid)[0]
        array = np.interp(np.arange(len(array)), indexes, array[indexes])

    window_size = max(1, min(window_size, len(array)))
    if window_size == 1:
        return array.tolist()

    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(array, kernel, mode="same")
    half = window_size // 2
    if half:
        smoothed[:half] = array[:half]
        smoothed[-half:] = array[-half:]
    return smoothed.tolist()


def plot_metric(axis, epochs: list[int], values: list[float], label: str) -> None:
    """绘制原始散点和平滑曲线。"""
    if not values:
        return
    axis.scatter(epochs, values, s=16, alpha=0.35, label=f"{label}原始值")
    axis.plot(epochs, smooth_curve(values), linewidth=2, label=label)


def plot_training_curves(csv_path: Path, output_path: Path | None = None) -> Path | None:
    """根据results.csv绘制训练曲线。"""
    output_path = output_path or CUSTOM_CURVE_PATH
    if not csv_path.is_file():
        print(f"[提示] 未找到训练指标文件：{csv_path}")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[提示] 未安装Matplotlib，跳过自定义曲线绘制。")
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = normalize_csv_rows(list(csv.DictReader(file)))

    if not rows:
        print("[提示] results.csv内容为空，无法绘图。")
        return None

    epoch_raw = get_metric_series(rows, ["epoch"])
    epochs = [
        int(value) + 1 if np.isfinite(value) else index + 1
        for index, value in enumerate(epoch_raw or range(len(rows)))
    ]

    map50 = get_metric_series(rows, ["metrics/mAP50(B)", "metrics/mAP50(O)", "metrics/mAP50"])
    map50_95 = get_metric_series(
        rows,
        ["metrics/mAP50-95(B)", "metrics/mAP50-95(O)", "metrics/mAP50-95"],
    )
    precision = get_metric_series(
        rows,
        ["metrics/precision(B)", "metrics/precision(O)", "metrics/precision"],
    )
    recall = get_metric_series(
        rows,
        ["metrics/recall(B)", "metrics/recall(O)", "metrics/recall"],
    )

    train_box = get_metric_series(rows, ["train/box_loss"])
    val_box = get_metric_series(rows, ["val/box_loss"])
    train_cls = get_metric_series(rows, ["train/cls_loss"])
    val_cls = get_metric_series(rows, ["val/cls_loss"])
    train_dfl = get_metric_series(rows, ["train/dfl_loss"])
    val_dfl = get_metric_series(rows, ["val/dfl_loss"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    for values, label in (
        (map50, "mAP@50"),
        (map50_95, "mAP@50-95"),
        (precision, "Precision"),
        (recall, "Recall"),
    ):
        plot_metric(axes[0], epochs, values, label)

    axes[0].set_title("验证集检测指标")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plot_metric(axes[1], epochs, train_box, "Train Box Loss")
    plot_metric(axes[1], epochs, val_box, "Val Box Loss")
    axes[1].set_title("边界框损失")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    for values, label in (
        (train_cls, "Train Cls Loss"),
        (val_cls, "Val Cls Loss"),
        (train_dfl, "Train DFL Loss"),
        (val_dfl, "Val DFL Loss"),
    ):
        plot_metric(axes[2], epochs, values, label)

    axes[2].set_title("分类与DFL损失")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    figure.suptitle(f"YOLO11-OBB训练曲线—{EXPERIMENT_NAME}", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def copy_if_exists(source: Path, destination: Path) -> bool:
    """源文件存在时复制到目标路径。"""
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return True


def get_pretrained_model() -> str:
    """优先使用本地权重，不存在时使用官方模型名。"""
    if PRETRAINED_PATH.is_file():
        print(f"使用本地预训练权重：{PRETRAINED_PATH}")
        return str(PRETRAINED_PATH)

    print(
        f"[提示] 本地未找到：{PRETRAINED_PATH}\n"
        "将使用yolo11m-obb.pt，首次运行时由Ultralytics自动下载。"
    )
    return "yolo11m-obb.pt"


def resolve_dataset_root(config: dict) -> Path:
    """解析数据集根目录。"""
    raw_path = str(config.get("path", "")).strip()
    dataset_root = Path(raw_path) if raw_path else YAML_PATH.parent
    if not dataset_root.is_absolute():
        dataset_root = (YAML_PATH.parent / dataset_root).resolve()
    return dataset_root


def labels_dir_from_images_dir(image_dir: Path) -> Path:
    """从images目录推导对应labels目录。"""
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels"


def check_label_line(
    line: str,
    label_path: Path,
    line_number: int,
    class_count: int | None,
) -> list[str]:
    """检查单行OBB标签。"""
    issues: list[str] = []
    parts = line.strip().split()
    if len(parts) != 9:
        return [f"{label_path}第{line_number}行：共有{len(parts)}列，OBB应为9列"]

    try:
        class_id = int(float(parts[0]))
        coordinates = np.asarray([float(value) for value in parts[1:]], dtype=float)
    except ValueError:
        return [f"{label_path}第{line_number}行：存在无法解析的数值"]

    if class_count is not None and not 0 <= class_id < class_count:
        issues.append(
            f"{label_path}第{line_number}行：类别ID={class_id}超出0～{class_count - 1}"
        )

    if not np.isfinite(coordinates).all():
        issues.append(f"{label_path}第{line_number}行：坐标包含NaN或无穷值")
    elif np.any((coordinates < 0.0) | (coordinates > 1.0)):
        issues.append(f"{label_path}第{line_number}行：归一化坐标不在0～1范围内")
    else:
        points = coordinates.reshape(4, 2)
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < 1e-6:
            issues.append(f"{label_path}第{line_number}行：旋转框面积接近0")

    return issues


def check_dataset() -> None:
    """检查YAML、图片、标签、类别ID和OBB坐标。"""
    if not YAML_PATH.is_file():
        raise FileNotFoundError(f"数据集配置文件不存在：{YAML_PATH}")

    try:
        with YAML_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except Exception as error:
        raise RuntimeError(f"读取数据集配置失败：{error}") from error

    class_count: int | None = None
    names = config.get("names")
    if isinstance(names, (list, tuple, dict)):
        class_count = len(names)
    elif config.get("nc") is not None:
        class_count = int(config["nc"])

    dataset_root = resolve_dataset_root(config)
    fatal_issues: list[str] = []
    warnings: list[str] = []

    for split in ("train", "val", "test"):
        split_path = config.get(split)
        if not split_path:
            continue

        image_dir = Path(split_path)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        label_dir = labels_dir_from_images_dir(image_dir)

        image_files = sorted(
            path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if image_dir.is_dir() else []
        label_files = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []

        print(f"[{split}] 图片：{len(image_files)}，标签：{len(label_files)}")

        if not image_files:
            fatal_issues.append(f"{split}图片目录为空或不存在：{image_dir}")
            continue
        if not label_files:
            fatal_issues.append(f"{split}标签目录为空或不存在：{label_dir}")
            continue

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)

        if missing_labels:
            warnings.append(f"{split}有{len(missing_labels)}张图片缺少同名标签")
        if orphan_labels:
            warnings.append(f"{split}有{len(orphan_labels)}个标签没有对应图片")

        for label_path in label_files:
            try:
                lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            except Exception as error:
                fatal_issues.append(f"无法读取标签：{label_path}，原因：{error}")
                continue

            for line_number, line in enumerate(lines, 1):
                if line.strip():
                    fatal_issues.extend(
                        check_label_line(line, label_path, line_number, class_count)
                    )

    for warning in warnings:
        print(f"[警告] {warning}")

    if fatal_issues:
        print("\n[错误] 数据集检查未通过：")
        for issue in fatal_issues[:30]:
            print(f"  - {issue}")
        if len(fatal_issues) > 30:
            print(f"  ...另有{len(fatal_issues) - 30}项问题")
        raise SystemExit("请先修复数据集或OBB标注后再训练。")


def print_validation_metrics(metrics: "Results") -> None:
    """输出最佳模型的验证指标。"""
    print("\n" + "=" * 60)
    print("最佳模型验证结果")
    print("=" * 60)

    box = getattr(metrics, "box", None)
    if box is None:
        print("未能读取box指标，请查看Ultralytics输出。")
        return

    for label, attribute in (
        ("Precision", "mp"),
        ("Recall", "mr"),
        ("mAP@50", "map50"),
        ("mAP@50-95", "map"),
    ):
        value = getattr(box, attribute, None)
        if value is not None:
            try:
                print(f"{label:<12}: {float(value):.4f}")
            except (TypeError, ValueError):
                print(f"{label:<12}: {value}")

    speed = getattr(metrics, "speed", None)
    if speed:
        print("\n推理速度：")
        for key in ("preprocess", "inference", "loss", "postprocess"):
            if key in speed:
                print(f"  {key:<12}: {speed[key]:.2f} ms")


def main() -> None:
    configure_chinese_font()
    check_dataset()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        print(f"缺少依赖：{error}")
        print("请执行：python -m pip install ultralytics matplotlib numpy pyyaml")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device: int | str = 0
        batch_size = GPU_BATCH_SIZE
        amp = True
        device_info = f"GPU：{torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        batch_size = CPU_BATCH_SIZE
        amp = False
        device_info = "未检测到CUDA GPU，将使用CPU训练"

    workers = 0 if os.name == "nt" else 4

    print("=" * 60)
    print(device_info)
    print(f"输入尺寸：{IMAGE_SIZE}")
    print(f"Batch size：{batch_size}")
    print(f"混合精度训练：{'开启' if amp else '关闭'}")
    print(f"训练轮数：{EPOCHS}")
    print(f"早停耐心值：{PATIENCE}")
    print("=" * 60)

    model = YOLO(get_pretrained_model())
    print("开始训练YOLO11-OBB药盒检测模型...\n")

    try:
        train_results = model.train(
            data=str(YAML_PATH),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=batch_size,
            workers=workers,
            device=device,
            project=str(OUTPUT_DIR),
            name=EXPERIMENT_NAME,
            exist_ok=EXIST_OK,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,
            weight_decay=0.0005,
            warmup_epochs=3,
            label_smoothing=0.1,
            patience=PATIENCE,
            seed=SEED,
            deterministic=True,
            save=True,
            save_period=-1,
            plots=True,
            verbose=True,
            amp=amp,
            hsv_h=0.015,
            hsv_s=0.50,
            hsv_v=0.20,
            degrees=30.0,
            translate=0.10,
            scale=0.50,
            shear=0.0,
            perspective=0.0,
            flipud=0.20,
            fliplr=0.50,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.0,
            erasing=0.0,
            close_mosaic=15,
            cache=False,
        )

        save_dir = Path(train_results.save_dir)
        best_source = save_dir / "weights" / "best.pt"
        last_source = save_dir / "weights" / "last.pt"
        csv_source = save_dir / "results.csv"
        plot_source = save_dir / "results.png"

        print("\n" + "=" * 60)
        print("训练完成，正在整理结果")
        print("=" * 60)
        print(f"训练目录：{save_dir}")

        if copy_if_exists(best_source, BEST_MODEL_PATH):
            print(f"最佳权重：{BEST_MODEL_PATH}")
        else:
            print(f"[警告] 未找到最佳权重：{best_source}")

        if copy_if_exists(last_source, LAST_MODEL_PATH):
            print(f"最后权重：{LAST_MODEL_PATH}")

        if copy_if_exists(csv_source, RESULTS_CSV_PATH):
            print(f"训练指标：{RESULTS_CSV_PATH}")
        else:
            print(f"[警告] 未找到results.csv：{csv_source}")

        if copy_if_exists(plot_source, ULTRALYTICS_PLOT_PATH):
            print(f"官方训练曲线：{ULTRALYTICS_PLOT_PATH}")

        if RESULTS_CSV_PATH.is_file():
            curve_path = plot_training_curves(RESULTS_CSV_PATH)
            if curve_path:
                print(f"自定义训练曲线：{curve_path}")

        if BEST_MODEL_PATH.is_file():
            print("\n正在使用best.pt重新评估验证集...")
            best_model = YOLO(str(BEST_MODEL_PATH))
            validation_metrics = best_model.val(
                data=str(YAML_PATH),
                imgsz=IMAGE_SIZE,
                batch=batch_size,
                device=device,
                workers=workers,
                split="val",
                plots=True,
                verbose=True,
            )
            print_validation_metrics(validation_metrics)

    except Exception as error:
        print("\n" + "=" * 60)
        print("执行过程中出现错误❌")
        print("=" * 60)
        print(f"错误信息：{error}")
        print("\n常见原因：")
        print("  - 数据集路径配置错误")
        print("  - GPU显存不足，可降低GPU_BATCH_SIZE或IMAGE_SIZE")
        print("  - OBB标签格式、类别ID或坐标存在问题")
        print("  - 当前Ultralytics版本不支持某个训练参数")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("所有任务已完成🎯")
    print("=" * 60)
    print(f"最佳模型：{BEST_MODEL_PATH}")
    print("\n测试单张图片：")
    print(
        "yolo predict "
        f'model="{BEST_MODEL_PATH}" '
        'source="你的图片路径.jpg" '
        "conf=0.25 save=True"
    )


if __name__ == "__main__":
    main()
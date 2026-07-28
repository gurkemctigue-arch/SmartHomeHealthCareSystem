from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HERE = ROOT
YAML_PATH = ROOT / "data" / "medicine" / "medicine_obb.yaml"
OUTPUT_DIR = HERE / "output"

EXPERIMENT_NAME = "result"
IMAGE_SIZE = 960
GPU_BATCH_SIZE = 4
CPU_BATCH_SIZE = 1
SAVE_RESULT_IMAGES = True

MODEL_PATH = OUTPUT_DIR / EXPERIMENT_NAME / "weights" / "best.pt"


def print_metrics(metrics) -> None:
    print("\n" + "=" * 60)
    print("验证集评估结果")
    print("=" * 60)

    box = getattr(metrics, "box", None)
    if box is None:
        print("未能读取 box 指标，请查看上方 Ultralytics 输出。")
        return

    for label, attr in [
        ("Precision", "mp"),
        ("Recall", "mr"),
        ("mAP@50", "map50"),
        ("mAP@50-95", "map"),
    ]:
        val = getattr(box, attr, None)
        if val is not None:
            try:
                print(f"{label:<12}: {float(val):.4f}")
            except (TypeError, ValueError):
                print(f"{label:<12}: {val}")

    speed = getattr(metrics, "speed", None)
    if speed:
        print("\n推理速度：")
        for key in ("preprocess", "inference", "loss", "postprocess"):
            if key in speed:
                print(f"  {key:<12}: {speed[key]:.2f} ms")


def main() -> None:
    if not MODEL_PATH.is_file():
        print(f"模型文件不存在：{MODEL_PATH}")
        print("请先运行 train.py 训练模型，或修改 MODEL_PATH 指向已有权重。")
        sys.exit(1)

    if not YAML_PATH.is_file():
        print(f"数据集配置文件不存在：{YAML_PATH}")
        sys.exit(1)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        print(f"缺少依赖：{error}")
        print("\n请在 cv 环境中执行：")
        print("python -m pip install ultralytics")
        sys.exit(1)

    if torch.cuda.is_available():
        device: int | str = 0
        batch_size = GPU_BATCH_SIZE
        device_info = f"GPU：{torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        batch_size = CPU_BATCH_SIZE
        device_info = "CPU"

    workers = 0 if os.name == "nt" else 4

    print("=" * 60)
    print(f"模型：{MODEL_PATH}")
    print(f"设备：{device_info}")
    print(f"Batch size：{batch_size}")
    print(f"输入尺寸：{IMAGE_SIZE}")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    # 评估时不传 conf/iou，使用 Ultralytics 默认值（conf=0.001, iou=0.7）
    # 确保不因阈值过滤而损失 Recall，获得真实的 mAP 分数
    metrics = model.val(
        data=str(YAML_PATH),
        imgsz=IMAGE_SIZE,
        batch=batch_size,
        device=device,
        workers=workers,
        split="val",
        plots=SAVE_RESULT_IMAGES,
        verbose=True,
    )

    print_metrics(metrics)

    print("\n" + "=" * 60)
    print("验证完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

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
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 设为 None 则自动从 YAML 解析（优先 test，其次 val）
SOURCE: Path | str | None = None

MODEL_PATH = OUTPUT_DIR / EXPERIMENT_NAME / "weights" / "best.pt"
PREDICT_OUTPUT_DIR = OUTPUT_DIR / EXPERIMENT_NAME / "predictions"


def _resolve_source() -> Path:
    """从 YAML 自动解析测试来源：优先 test，其次 val。"""
    if SOURCE is not None:
        source = Path(SOURCE)
        if source.exists():
            return source
        print(f"[警告] 指定的 SOURCE 不存在：{SOURCE}")

    if not YAML_PATH.is_file():
        raise FileNotFoundError(f"数据集配置文件不存在：{YAML_PATH}")

    import yaml
    with open(YAML_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ds_root = Path(cfg.get("path", ""))
    if not ds_root.is_absolute():
        ds_root = YAML_PATH.parent / ds_root

    for key in ("test", "val"):
        sub = cfg.get(key, "")
        if sub:
            source = ds_root / sub
            if source.exists():
                print(f"自动使用 [{key}] 集：{source}")
                return source
            print(f"[警告] YAML 中 [{key}] 路径不存在：{source}")

    raise FileNotFoundError(
        "无法自动解析测试来源。请修改 SOURCE 变量指定图片路径或目录。"
    )


def main() -> None:
    if not MODEL_PATH.is_file():
        print(f"模型文件不存在：{MODEL_PATH}")
        print("请先运行 train.py 训练模型，或修改 EXPERIMENT_NAME。")
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

    try:
        source = _resolve_source()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("=" * 60)
    print(f"模型：{MODEL_PATH}")
    print(f"设备：{device_info}")
    print(f"Batch size：{batch_size}")
    print(f"Conf 阈值：{CONF_THRESHOLD}")
    print(f"IoU 阈值：{IOU_THRESHOLD}")
    print(f"测试来源：{source}")
    print(f"输出目录：{PREDICT_OUTPUT_DIR}")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(source),
        imgsz=IMAGE_SIZE,
        batch=batch_size,
        device=device,
        workers=workers,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(PREDICT_OUTPUT_DIR.parent),
        name=PREDICT_OUTPUT_DIR.name,
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"预测完成 ✅")
    print("=" * 60)

    total = len(results)
    detected = sum(1 for r in results if r.boxes is not None and len(r.boxes) > 0)
    print(f"\n图片总数：{total}")
    print(f"检测到目标：{detected} 张")
    print(f"结果保存至：{PREDICT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

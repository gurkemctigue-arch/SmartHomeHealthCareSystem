"""摄像头实时 YOLO 药盒检测。

加载 output/result/weights/best.pt，OOM 时自动降尺寸或切 CPU。
运行: python camera.py  按 q 退出。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "output" / "result" / "weights" / "best.pt"

CONF = 0.10
IOU = 0.5
IMGSZ = 416
PRINT_EVERY = 30


def _n_det(r) -> int:
    obb = getattr(r, "obb", None)
    if obb is not None and len(obb) > 0:
        return len(obb)
    boxes = getattr(r, "boxes", None)
    return len(boxes) if boxes is not None else 0


def _pick_device() -> str | int:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            name = torch.cuda.get_device_name(0)
            print(f"GPU: {name}")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"GPU 不可用（{e}），改用 CPU")
    return "cpu"


def _predict(model, frame, device, imgsz: int):
    """推理；显存不足时返回 (None, err)。"""
    try:
        return model.predict(
            frame,
            conf=CONF,
            iou=IOU,
            imgsz=imgsz,
            device=device,
            verbose=False,
        ), None
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, e
        raise


def main():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"未找到权重文件：{WEIGHTS_PATH}\n"
            "请先运行 train.py 训练模型，或修改 WEIGHTS_PATH 指向已有权重。"
        )

    device = _pick_device()
    imgsz = IMGSZ

    print("正在加载模型（第一次较慢，请稍等）…", flush=True)
    model = YOLO(str(WEIGHTS_PATH))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 下 DSHOW 往往更稳
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头（是否被其它程序占用？）")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"模型: {WEIGHTS_PATH}")
    print(f"device={device} conf={CONF} iou={IOU} imgsz={imgsz}  按 q 退出", flush=True)
    print("提示: 药盒占画面约 1/3～1/2；若卡住请关掉其它占 GPU 的训练窗口", flush=True)

    # 预热一帧，把报错提前暴露
    ok, warm = cap.read()
    if ok:
        results, err = _predict(model, warm, device, imgsz)
        if results is None:
            print(f"预热 OOM/CUDA 失败: {err} → 改 imgsz=320 或 CPU", flush=True)
            imgsz = 320
            results, err = _predict(model, warm, device, imgsz)
            if results is None:
                device = "cpu"
                imgsz = 320
                print(f"改用 CPU imgsz={imgsz}", flush=True)
                results, err = _predict(model, warm, device, imgsz)
                if results is None:
                    raise RuntimeError(f"推理失败: {err}")
        print(f"预热成功，检出 {_n_det(results[0])} 个框", flush=True)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头失败")
            break

        results, err = _predict(model, frame, device, imgsz)
        if results is None:
            print(f"[跳过帧] 推理失败: {err}", flush=True)
            continue

        r = results[0]
        if frame_i % PRINT_EVERY == 0:
            print(f"[frame {frame_i}] 检出 {_n_det(r)} 个框", flush=True)

        # 使用 ultralytics 内置绘图（自动绘制中文标签）
        annotated = r.plot()
        cv2.imshow("YOLO medicine detect", annotated)
        frame_i += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()

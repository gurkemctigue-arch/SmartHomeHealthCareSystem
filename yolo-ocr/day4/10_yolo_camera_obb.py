# -*- coding: utf-8 -*-
"""YOLO medicine-box localization + RapidOCR + SQLite category lookup.

YOLO is used only to locate a medicine box. Its predicted class is deliberately
ignored: the rectified box is read by OCR and the medicine category comes from
the SQLite catalog.

Examples (run under ai-training/ with the ai-vision-nmgdx environment):
  python day4/10_yolo_camera_obb.py --check
  python day4/10_yolo_camera_obb.py
  python day4/10_yolo_camera_obb.py --image data/yolo/medicine/images/train/images/1.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_path import imread_color, imwrite_color  # noqa: E402
from medicine_db import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    DEFAULT_DB_PATH,
    MedicineDatabase,
    MedicineMatch,
)
from medicine_ocr import perspective_crop, recognize_crop  # noqa: E402

MODELS_DIR = ROOT / "models"
CONF = 0.15
IOU = 0.5
IMGSZ = 640
PRINT_EVERY = 30
DEFAULT_CAPTURE_DIR = os.environ.get(
    "YOLO_OBB_CAPTURE_DIR",
    str(ROOT / "output" / "camera_hard_cases"),
)
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
)


@dataclass
class Detection:
    polygon_norm: np.ndarray
    polygon: np.ndarray
    confidence: float
    state_id: int = -1

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.polygon[:, 0].min()),
            float(self.polygon[:, 1].min()),
            float(self.polygon[:, 0].max()),
            float(self.polygon[:, 1].max()),
        )

    @property
    def area(self) -> float:
        return abs(float(cv2.contourArea(self.polygon.astype(np.float32))))


@dataclass
class TrackState:
    state_id: int
    bbox: tuple[float, float, float, float]
    last_seen: int
    last_submitted: int = -1_000_000
    last_completed: int = -1_000_000
    medicine_name: str = "识别中..."
    category_name: str = "待查询"
    raw_text: str = ""
    ocr_confidence: float = 0.0
    match_score: float = 0.0
    match_type: str = "pending"
    status: str = "pending"


def _parse_angles(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",") if value.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("OCR angles must be comma-separated integers") from exc
    if not values or any(value not in {0, 90, 180, 270} for value in values):
        raise argparse.ArgumentTypeError("OCR angles may contain only 0, 90, 180, 270")
    return values


def _candidate_weights() -> tuple[Path, ...]:
    candidates: list[Path] = []
    artifact_root = os.environ.get("YOLO_OBB_ARTIFACT_ROOT")
    if artifact_root:
        candidates.append(Path(artifact_root).expanduser() / "models/best_obb.pt")
    candidates.extend((MODELS_DIR / "best_obb.pt", MODELS_DIR / "best.pt"))
    if os.name == "nt":
        candidates.append(Path("D:/ai-vision-training/models/best_obb.pt"))
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _resolve_weights(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"指定的权重不存在：{path}")
    candidates = _candidate_weights()
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("未找到权重。已查找：\n  - " + "\n  - ".join(map(str, candidates)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 药盒定位 + OCR 药名 + SQLite 18 类查询")
    parser.add_argument("--weights", help="YOLO OBB/detect 权重路径")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH, help="SQLite 药品数据库")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH, help="药品种子目录 YAML")
    parser.add_argument("--check", action="store_true", help="验证模型、OCR 和数据库后退出")
    parser.add_argument("--image", type=Path, help="识别一张静态图片，不打开摄像头")
    parser.add_argument("--output", type=Path, help="静态图片识别结果保存路径")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=CONF, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU, help="YOLO NMS IoU threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--ocr-interval", type=int, default=15, help="同一药盒重新 OCR 的最小帧间隔")
    parser.add_argument("--ocr-max-boxes", type=int, default=3, help="每批最多 OCR 的药盒数")
    parser.add_argument("--ocr-angles", type=_parse_angles, default=(0, 180), help="OCR 方向，例如 0,180")
    parser.add_argument("--min-ocr-area", type=float, default=2500.0, help="启用 OCR 的最小框面积（像素）")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.78, help="药名模糊匹配阈值")
    parser.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR, help="困难样本保存目录")
    args = parser.parse_args()
    if args.imgsz < 320:
        parser.error("--imgsz must be at least 320")
    if not 0.01 <= args.conf <= 1.0:
        parser.error("--conf must be in [0.01, 1.0]")
    if not 0.01 <= args.iou <= 1.0:
        parser.error("--iou must be in [0.01, 1.0]")
    if args.ocr_interval < 1 or args.ocr_max_boxes < 1:
        parser.error("--ocr-interval and --ocr-max-boxes must be positive")
    if args.min_ocr_area < 0:
        parser.error("--min-ocr-area must not be negative")
    if not 0.5 <= args.fuzzy_threshold <= 1.0:
        parser.error("--fuzzy-threshold must be in [0.5, 1.0]")
    return args


def _n_det(result: Any) -> int:
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        return len(obb)
    boxes = getattr(result, "boxes", None)
    return len(boxes) if boxes is not None else 0


def _pick_device() -> str | int:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"GPU 不可用（{exc}），改用 CPU")
    return "cpu"


def _predict(model: YOLO, frame: np.ndarray, device: str | int, imgsz: int, conf: float, iou: float):
    try:
        return model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            agnostic_nms=True,
            verbose=False,
        ), None
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, exc
        raise


def _predict_resilient(
    model: YOLO,
    frame: np.ndarray,
    device: str | int,
    imgsz: int,
    conf: float,
    iou: float,
) -> tuple[Any, str | int, int]:
    results, error = _predict(model, frame, device, imgsz, conf, iou)
    if results is not None:
        return results, device, imgsz
    if imgsz > 320:
        print(f"推理显存不足（{error}），降低 imgsz 到 320", flush=True)
        imgsz = 320
        results, error = _predict(model, frame, device, imgsz, conf, iou)
    if results is None and device != "cpu":
        print(f"GPU 推理失败（{error}），改用 CPU", flush=True)
        device = "cpu"
        imgsz = 320
        results, error = _predict(model, frame, device, imgsz, conf, iou)
    if results is None:
        raise RuntimeError(f"推理失败：{error}")
    return results, device, imgsz


def _polygon_iou(left: Detection, right: Detection) -> float:
    left_points = left.polygon.astype(np.float32)
    right_points = right.polygon.astype(np.float32)
    left_area = abs(float(cv2.contourArea(left_points)))
    right_area = abs(float(cv2.contourArea(right_points)))
    try:
        intersection, _ = cv2.intersectConvexConvex(left_points, right_points)
    except cv2.error:
        return _bbox_iou(left.bbox, right.bbox)
    return float(intersection) / max(1.0, left_area + right_area - float(intersection))


def _extract_detections(result: Any, image_shape: Sequence[int]) -> list[Detection]:
    height, width = image_shape[:2]
    scale = np.asarray([width, height], dtype=np.float32)
    detections: list[Detection] = []
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        polygons = obb.xyxyxyxyn.detach().cpu().numpy().reshape(-1, 4, 2)
        confidences = obb.conf.detach().cpu().numpy()
        for polygon_norm, confidence in zip(polygons, confidences):
            polygon_norm = np.clip(polygon_norm.astype(np.float32), 0.0, 1.0)
            detections.append(Detection(polygon_norm, polygon_norm * scale, float(confidence)))
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            rectangles = boxes.xyxyn.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            for (x1, y1, x2, y2), confidence in zip(rectangles, confidences):
                polygon_norm = np.asarray(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
                polygon_norm = np.clip(polygon_norm, 0.0, 1.0)
                detections.append(Detection(polygon_norm, polygon_norm * scale, float(confidence)))
    # The source model has 18 classes, so one package can receive overlapping
    # boxes from different classes. Once classification belongs to SQLite those
    # are duplicates; keep only the strongest location.
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    for detection in ordered:
        if all(_polygon_iou(detection, previous) < 0.55 for previous in kept):
            kept.append(detection)
    return kept


def _bbox_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, left_area + right_area - intersection)


def _assign_track_states(
    detections: list[Detection],
    states: dict[int, TrackState],
    frame_index: int,
    next_state_id: int,
) -> int:
    unused = {state_id for state_id, state in states.items() if frame_index - state.last_seen <= 45}
    for detection in detections:
        best_state_id = None
        best_score = 0.0
        for state_id in unused:
            state = states[state_id]
            iou = _bbox_iou(detection.bbox, state.bbox)
            dx = (detection.bbox[0] + detection.bbox[2] - state.bbox[0] - state.bbox[2]) / 2
            dy = (detection.bbox[1] + detection.bbox[3] - state.bbox[1] - state.bbox[3]) / 2
            state_scale = max(
                20.0,
                detection.bbox[2] - detection.bbox[0],
                detection.bbox[3] - detection.bbox[1],
                state.bbox[2] - state.bbox[0],
                state.bbox[3] - state.bbox[1],
            )
            center_score = max(0.0, 1.0 - (dx * dx + dy * dy) ** 0.5 / state_scale)
            score = max(iou, center_score * 0.65)
            if score > best_score:
                best_score = score
                best_state_id = state_id
        if best_state_id is None or best_score < 0.25:
            best_state_id = next_state_id
            next_state_id += 1
            states[best_state_id] = TrackState(best_state_id, detection.bbox, frame_index)
        else:
            unused.remove(best_state_id)
            states[best_state_id].bbox = detection.bbox
            states[best_state_id].last_seen = frame_index
        detection.state_id = best_state_id

    stale_ids = [state_id for state_id, state in states.items() if frame_index - state.last_seen > 60]
    for state_id in stale_ids:
        del states[state_id]
    return next_state_id


def _best_ocr_candidate(lines: Sequence[dict[str, Any]]) -> str:
    product_suffix = re.compile(
        r"(片|胶囊|颗粒|口服液|注射液|混悬液|糖浆|软膏|乳膏|滴眼液|散|丸|贴|绷带)$"
    )
    generic = re.compile(r"(说明书|有限公司|功能主治|国药准字|购买和使用|规格|生产日期)")
    candidates: list[tuple[float, str]] = []
    for line in lines:
        value = re.sub(r"\s+", "", str(line.get("text", ""))).strip()
        useful_length = len(re.sub(r"[^\w\u4e00-\u9fff]", "", value))
        if useful_length < 2:
            continue
        score = float(line.get("confidence", 0.0)) * 5.0 + min(useful_length, 14) * 0.2
        if product_suffix.search(value):
            score += 5.0
        if generic.search(value) or useful_length > 24:
            score -= 6.0
        candidates.append((score, value))
    if not candidates:
        return "未识别"
    value = max(candidates, key=lambda item: item[0])[1]
    return value if len(value) <= 22 else value[:21] + "..."


def _apply_ocr_result(
    state: TrackState,
    recognized: dict[str, Any],
    database: MedicineDatabase,
    fuzzy_threshold: float,
    other_category: str,
) -> MedicineMatch | None:
    lines = recognized.get("lines") or []
    line_texts = [str(line.get("text", "")) for line in lines]
    match = database.lookup(
        str(recognized.get("text", "")),
        lines=line_texts,
        fuzzy_threshold=fuzzy_threshold,
    )
    state.raw_text = str(recognized.get("text", ""))
    state.ocr_confidence = float(recognized.get("average_confidence", 0.0))
    if match is not None:
        state.medicine_name = match.medicine_name
        state.category_name = match.category_name
        state.match_score = match.score
        state.match_type = match.match_type
        state.status = "known"
    else:
        state.medicine_name = _best_ocr_candidate(lines)
        state.category_name = other_category
        state.match_score = 0.0
        state.match_type = "none"
        state.status = "unknown"
    return match


def _ocr_batch(
    engine: RapidOCR,
    tasks: Sequence[tuple[int, np.ndarray]],
    angles: Sequence[int],
) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    output = []
    for state_id, crop in tasks:
        try:
            output.append((state_id, recognize_crop(crop, engine, angles), None))
        except Exception as exc:  # noqa: BLE001
            output.append((state_id, None, str(exc)))
    return output


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection],
    states: dict[int, TrackState],
) -> np.ndarray:
    annotated = frame.copy()
    colors = {
        "known": (60, 200, 90),
        "unknown": (0, 210, 255),
        "error": (40, 40, 230),
        "pending": (255, 170, 30),
    }
    for detection in detections:
        state = states[detection.state_id]
        color = colors.get(state.status, colors["pending"])
        cv2.polylines(annotated, [np.round(detection.polygon).astype(np.int32)], True, color, 2)

    image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    for detection in detections:
        state = states[detection.state_id]
        color_bgr = colors.get(state.status, colors["pending"])
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        lines = (
            f"药名：{state.medicine_name}",
            f"类别：{state.category_name}  定位：{detection.confidence:.2f}",
        )
        font_size = 18
        while True:
            font = _load_font(font_size)
            boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
            text_width = max(box[2] - box[0] for box in boxes)
            if text_width <= image_width - 12 or font_size <= 12:
                break
            font_size -= 1
        line_height = max(box[3] - box[1] for box in boxes) + 3
        panel_width = min(image_width - 4, text_width + 10)
        panel_height = line_height * len(lines) + 7
        x1, y1, _, _ = detection.bbox
        panel_x = int(min(max(2, x1), max(2, image_width - panel_width - 2)))
        above_y = int(y1) - panel_height - 3
        panel_y = above_y if above_y >= 2 else min(image_height - panel_height - 2, int(y1) + 3)
        panel_y = max(2, panel_y)
        draw.rectangle(
            [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
            fill=color_rgb,
        )
        for index, line in enumerate(lines):
            draw.text(
                (panel_x + 5, panel_y + 2 + index * line_height),
                line,
                font=font,
                fill=(255, 255, 255),
            )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _prediction_records(result: Any) -> list[dict[str, Any]]:
    names = result.names
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        classes = obb.cls.detach().cpu().numpy().astype(int)
        confidences = obb.conf.detach().cpu().numpy()
        polygons = obb.xyxyxyxyn.detach().cpu().numpy()
        return [
            {
                "class_id": int(class_id),
                "class_name": str(names[int(class_id)]),
                "confidence": float(confidence),
                "obb_xyxyxyxyn": polygon.reshape(-1).astype(float).tolist(),
            }
            for class_id, confidence, polygon in zip(classes, confidences, polygons)
        ]
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        rectangles = boxes.xyxyn.detach().cpu().numpy()
        return [
            {
                "class_id": int(class_id),
                "class_name": str(names[int(class_id)]),
                "confidence": float(confidence),
                "xyxyn": rectangle.astype(float).tolist(),
            }
            for class_id, confidence, rectangle in zip(classes, confidences, rectangles)
        ]
    return []


def _save_hard_case(
    frame: np.ndarray,
    result: Any,
    capture_root: Path,
    kind: str,
    model_path: Path,
    imgsz: int,
    conf: float,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    group = "negative" if kind == "negative" else "review"
    image_dir = capture_root / group / "images"
    metadata_dir = capture_root / group / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"camera_{timestamp}.jpg"
    imwrite_color(str(image_path), frame)
    metadata = {
        "kind": kind,
        "image": str(image_path),
        "model": str(model_path),
        "imgsz": imgsz,
        "conf": conf,
        "predictions": _prediction_records(result),
    }
    metadata_path = metadata_dir / f"camera_{timestamp}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if kind == "negative":
        label_dir = capture_root / group / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / f"camera_{timestamp}.txt").write_text("", encoding="utf-8")
    return image_path


def _run_image(
    args: argparse.Namespace,
    model: YOLO,
    engine: RapidOCR,
    database: MedicineDatabase,
    model_path: Path,
    device: str | int,
) -> None:
    image_path = args.image.expanduser().resolve()
    frame = imread_color(str(image_path))
    results, device, imgsz = _predict_resilient(
        model, frame, device, args.imgsz, args.conf, args.iou
    )
    detections = _extract_detections(results[0], frame.shape)
    states: dict[int, TrackState] = {}
    other_category = str(database.category(17)["name"])
    for index, detection in enumerate(detections):
        detection.state_id = index
        state = TrackState(index, detection.bbox, 0)
        states[index] = state
        if index >= args.ocr_max_boxes or detection.area < args.min_ocr_area:
            state.medicine_name = "框过小，未执行 OCR"
            state.category_name = "待查询"
            continue
        crop = perspective_crop(frame, detection.polygon_norm.reshape(-1))
        recognized = recognize_crop(crop, engine, args.ocr_angles)
        match = _apply_ocr_result(
            state, recognized, database, args.fuzzy_threshold, other_category
        )
        print(f"框 {index + 1}: YOLO={detection.confidence:.3f}")
        print(f"  OCR: {recognized['text'].replace(chr(10), ' | ')}")
        if match is None:
            print(f"  查询: 未收录 -> {other_category}")
        else:
            print(
                f"  查询: {match.medicine_name} -> {match.category_name} "
                f"({match.match_type}, {match.score:.2f})"
            )
    annotated = _draw_overlay(frame, detections, states)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else (ROOT / "output/medicine_ocr_result.jpg").resolve()
    )
    imwrite_color(str(output_path), annotated)
    print(f"静态图检测完成：{len(detections)} 个框，device={device}, imgsz={imgsz}")
    print(f"结果图：{output_path}")


def _run_camera(
    args: argparse.Namespace,
    model: YOLO,
    engine: RapidOCR,
    database: MedicineDatabase,
    model_path: Path,
    device: str | int,
) -> None:
    capture_root = Path(args.capture_dir).expanduser().resolve()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头（是否被其他程序占用？）")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"模型：{model_path}")
    print(f"数据库：{database.db_path}")
    print(f"device={device} conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    print("YOLO 只负责药盒定位；类别由 OCR 药名查询 SQLite 得出。")
    print(f"困难样本目录：{capture_root}")
    print("按键：q=退出，s=保存复核样本，n=保存确认的负样本", flush=True)

    ok, warm_frame = cap.read()
    imgsz = args.imgsz
    if ok:
        warm_results, device, imgsz = _predict_resilient(
            model, warm_frame, device, imgsz, args.conf, args.iou
        )
        print(f"预热成功，检测到 {_n_det(warm_results[0])} 个框", flush=True)

    states: dict[int, TrackState] = {}
    next_state_id = 0
    frame_index = 0
    future: Future | None = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medicine-ocr")
    other_category = str(database.category(17)["name"])
    last_result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取摄像头失败")
                break
            results, device, imgsz = _predict_resilient(
                model, frame, device, imgsz, args.conf, args.iou
            )
            result = results[0]
            last_result = result
            detections = _extract_detections(result, frame.shape)
            next_state_id = _assign_track_states(
                detections, states, frame_index, next_state_id
            )

            if future is not None and future.done():
                try:
                    for state_id, recognized, error in future.result():
                        state = states.get(state_id)
                        if state is None:
                            continue
                        state.last_completed = frame_index
                        if error or recognized is None:
                            state.medicine_name = "OCR 失败"
                            state.category_name = "待查询"
                            state.status = "error"
                            print(f"OCR 错误（框 {state_id}）：{error}", flush=True)
                            continue
                        _apply_ocr_result(
                            state,
                            recognized,
                            database,
                            args.fuzzy_threshold,
                            other_category,
                        )
                        summary = state.raw_text.replace("\n", " | ")
                        print(
                            f"OCR 框 {state_id}: {summary} -> "
                            f"{state.medicine_name} / {state.category_name}",
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"OCR 后台任务失败：{exc}", flush=True)
                future = None

            if future is None:
                tasks: list[tuple[int, np.ndarray]] = []
                for detection in detections:
                    state = states[detection.state_id]
                    last_ocr_frame = max(state.last_submitted, state.last_completed)
                    if frame_index - last_ocr_frame < args.ocr_interval:
                        continue
                    if detection.area < args.min_ocr_area:
                        continue
                    crop = perspective_crop(frame, detection.polygon_norm.reshape(-1))
                    tasks.append((state.state_id, crop.copy()))
                    state.last_submitted = frame_index
                    if state.status in {"pending", "error"}:
                        state.medicine_name = "识别中..."
                        state.category_name = "待查询"
                        state.status = "pending"
                    if len(tasks) >= args.ocr_max_boxes:
                        break
                if tasks:
                    future = executor.submit(_ocr_batch, engine, tasks, args.ocr_angles)

            if frame_index % PRINT_EVERY == 0:
                print(f"[frame {frame_index}] 检测到 {len(detections)} 个药盒框", flush=True)
            annotated = _draw_overlay(frame, detections, states)
            cv2.imshow("YOLO + OCR medicine lookup", annotated)
            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and last_result is not None:
                saved = _save_hard_case(
                    frame, last_result, capture_root, "review", model_path, imgsz, args.conf
                )
                print(f"已保存复核样本：{saved}", flush=True)
            elif key == ord("n") and last_result is not None:
                saved = _save_hard_case(
                    frame, last_result, capture_root, "negative", model_path, imgsz, args.conf
                )
                print(f"已保存负样本：{saved}", flush=True)
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if future is not None:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    print("已退出")


def main() -> None:
    args = parse_args()
    model_path = _resolve_weights(args.weights)
    device = _pick_device()
    print("正在加载 YOLO 模型...", flush=True)
    model = YOLO(str(model_path))
    if model.task not in {"obb", "detect"}:
        raise ValueError(f"权重任务为 {model.task!r}，不是 OBB/detect 模型：{model_path}")

    print("正在加载药品数据库和 RapidOCR...", flush=True)
    with MedicineDatabase(args.database, args.catalog, initialize=True) as database:
        engine = RapidOCR()
        stats = database.stats()
        if args.check:
            print(f"权重加载成功：{model_path}")
            print(f"模型任务：{model.task}，device={device}")
            print(
                f"数据库加载成功：18 类，{stats['medicines']} 个药名，"
                f"{stats['aliases']} 个药名/别名"
            )
            print("RapidOCR 加载成功；最终类别只使用 SQLite 查询结果。")
            return
        if args.image:
            _run_image(args, model, engine, database, model_path, device)
            return
        _run_camera(args, model, engine, database, model_path, device)


if __name__ == "__main__":
    main()

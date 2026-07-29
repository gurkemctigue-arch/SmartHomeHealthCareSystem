"""YOLO-OBB medicine-box localization with OCR catalog recognition.

The trained YOLO classes are used only to localize packages. Each oriented box
is rectified and read by RapidOCR, then the medicine name and its 18-class
category are resolved from the bundled SQLite catalog.
"""

from __future__ import annotations

import atexit
import logging
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE_ROOT = PROJECT_ROOT / "yolo-ocr"
DAY4_DIR = BUNDLE_ROOT / "day4"
DEFAULT_WEIGHTS = BUNDLE_ROOT / "models" / "best_obb.pt"
DEFAULT_DATABASE = BUNDLE_ROOT / "db" / "medicine.db"
DEFAULT_CATALOG = DAY4_DIR / "medicine_catalog.yaml"


@dataclass
class Detection:
    polygon_norm: np.ndarray
    polygon: np.ndarray
    confidence: float
    state_id: int = -1
    persisted: bool = False

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
    polygon_norm: np.ndarray | None = None
    polygon: np.ndarray | None = None
    confidence: float = 0.0
    hit_count: int = 0
    confirmed: bool = False
    last_submitted: int = -1_000_000
    last_completed: int = -1_000_000
    medicine_name: str = "识别中..."
    category_name: str = "待查询"
    efficacy: str = ""
    efficacy_source: str = ""
    raw_text: str = ""
    ocr_confidence: float = 0.0
    match_score: float = 0.0
    match_type: str = "pending"
    status: str = "pending"
    medicine_id: int | None = None
    appearance_signature: np.ndarray | None = None
    appearance_change_hits: int = 0
    appearance_changed: bool = False
    last_appearance_check: int = -1_000_000
    candidate_medicine_id: int | None = None
    candidate_match_hits: int = 0


def _pick_device() -> str | int:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("药盒识别使用 GPU: %s", torch.cuda.get_device_name(0))
            return 0
    except Exception:
        pass
    logger.info("药盒识别使用 CPU")
    return "cpu"


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, left_area + right_area - intersection)


def _polygon_iou(left: Detection, right: Detection) -> float:
    left_points = left.polygon.astype(np.float32)
    right_points = right.polygon.astype(np.float32)
    left_area = abs(float(cv2.contourArea(left_points)))
    right_area = abs(float(cv2.contourArea(right_points)))
    try:
        intersection, _ = cv2.intersectConvexConvex(left_points, right_points)
    except cv2.error:
        return _bbox_iou(left.bbox, right.bbox)
    return float(intersection) / max(
        1.0, left_area + right_area - float(intersection)
    )


def _align_polygon_vertices(
    polygon: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Match cyclic order and winding before averaging two quadrilaterals."""
    candidates = []
    for ordered in (polygon, polygon[::-1]):
        candidates.extend(np.roll(ordered, shift, axis=0) for shift in range(4))
    return min(
        candidates,
        key=lambda candidate: float(np.square(candidate - reference).sum()),
    ).copy()


def _clamp_confidence(value: float, upper: float = 1.0) -> float:
    return min(upper, max(0.0, float(value)))


def _combined_recognition_confidence(
    detection_confidence: float,
    ocr_confidence: float,
    match_score: float,
    status: str,
) -> float:
    localization = _clamp_confidence(detection_confidence)
    ocr_quality = _clamp_confidence(ocr_confidence)
    name_match = _clamp_confidence(match_score)
    if status == "known":
        combined = (
            localization * 0.25
            + ocr_quality * 0.45
            + name_match * 0.30
        )
        return _clamp_confidence(combined, upper=0.99)
    if status == "unknown":
        combined = localization * 0.55 + ocr_quality * 0.45
        return _clamp_confidence(combined, upper=0.95)
    return localization


def _appearance_signature(crop: np.ndarray) -> np.ndarray:
    """Build a lighting-tolerant spatial color signature for package changes."""
    resized = cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histograms = []
    for row in range(2):
        for column in range(2):
            tile = hsv[row * 32 : (row + 1) * 32, column * 48 : (column + 1) * 48]
            histogram = cv2.calcHist(
                [tile], [0, 1], None, [18, 6], [0, 180, 0, 256]
            )
            cv2.normalize(histogram, histogram, 1.0, 0.0, cv2.NORM_L1)
            histograms.append(histogram.reshape(-1))
    return np.asarray(histograms, dtype=np.float32)


def _appearance_distance(reference: np.ndarray, current: np.ndarray) -> float:
    if reference.shape != current.shape or reference.ndim != 2:
        return 1.0
    distances = [
        cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA)
        for left, right in zip(reference, current)
    ]
    return float(np.mean(distances)) if distances else 1.0


def _blend_appearance(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    blended = reference * 0.95 + current * 0.05
    totals = blended.sum(axis=1, keepdims=True)
    return (blended / np.maximum(totals, 1e-6)).astype(np.float32)


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


class MedicineRecognizer:
    """Stateful real-time medicine recognizer used by the web video worker."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        database_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
        conf: float = 0.12,
        iou: float = 0.5,
        imgsz: int = 416,
        ocr_interval: int = 8,
        ocr_known_interval: int = 90,
        ocr_max_boxes: int = 3,
        min_ocr_area: float = 1600.0,
        fuzzy_threshold: float = 0.78,
        ocr_angles: Sequence[int] = (0, 180),
        confirm_hits: int = 2,
        hold_frames: int = 8,
        box_smoothing: float = 0.35,
        locked_box_smoothing: float = 0.18,
        immediate_conf: float = 0.22,
        switch_check_interval: int = 4,
        switch_threshold: float = 0.28,
        switch_confirm_hits: int = 2,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_WEIGHTS).expanduser().resolve()
        self.database_path = Path(database_path or DEFAULT_DATABASE).expanduser().resolve()
        self.catalog_path = Path(catalog_path or DEFAULT_CATALOG).expanduser().resolve()
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.ocr_interval = max(1, ocr_interval)
        self.ocr_known_interval = max(self.ocr_interval, ocr_known_interval)
        self.ocr_max_boxes = max(1, ocr_max_boxes)
        self.min_ocr_area = max(0.0, min_ocr_area)
        self.fuzzy_threshold = fuzzy_threshold
        self.ocr_angles = tuple(ocr_angles) or (0, 180)
        self.confirm_hits = max(1, confirm_hits)
        self.hold_frames = max(0, hold_frames)
        self.box_smoothing = min(1.0, max(0.0, box_smoothing))
        self.locked_box_smoothing = min(
            self.box_smoothing, max(0.01, locked_box_smoothing)
        )
        self.immediate_conf = max(self.conf, immediate_conf)
        self.switch_check_interval = max(1, switch_check_interval)
        self.switch_threshold = min(1.0, max(0.01, switch_threshold))
        self.switch_confirm_hits = max(1, switch_confirm_hits)

        self.model = None
        self.database = None
        self.ocr_engine = None
        self.device: str | int = "cpu"
        self.catalog_stats: dict[str, int] = {}
        self.error: str | None = None
        self.last_yolo_ms = 0.0
        self.last_ocr_ms = 0.0
        self.yolo_warmup_ms = 0.0
        self.ocr_warmup_ms = 0.0
        self._available = False
        self._lock = threading.Lock()
        self._states: dict[int, TrackState] = {}
        self._next_state_id = 0
        self._frame_index = 0
        self._future: Future | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._recognize_crop = None
        self._perspective_crop = None

        self._try_load()
        atexit.register(self.close)

    def _try_load(self) -> None:
        missing = [
            path
            for path in (self.model_path, self.catalog_path)
            if not path.is_file()
        ]
        if missing:
            self.error = "缺少文件: " + ", ".join(str(path) for path in missing)
            logger.error("yolo-ocr 初始化失败: %s", self.error)
            return

        try:
            day4_path = str(DAY4_DIR)
            if day4_path not in sys.path:
                sys.path.insert(0, day4_path)

            from medicine_db import MedicineDatabase
            from medicine_ocr import perspective_crop, recognize_crop
            from rapidocr_onnxruntime import RapidOCR
            from ultralytics import YOLO

            self.device = _pick_device()
            logger.info("加载 yolo-ocr 药盒定位模型: %s", self.model_path)
            self.model = YOLO(str(self.model_path))
            if self.model.task not in {"obb", "detect"}:
                raise ValueError(f"模型任务 {self.model.task!r} 不是 OBB/detect")
            self._warm_yolo()

            self.database = MedicineDatabase(
                self.database_path,
                self.catalog_path,
                initialize=True,
            )
            self.catalog_stats = self.database.stats()
            other_category = self.database.category(17)
            self.other_category = str(other_category["name"])
            self.other_efficacy = str(other_category["efficacy"])
            self.ocr_engine = RapidOCR()
            self._recognize_crop = recognize_crop
            self._perspective_crop = perspective_crop
            self._warm_ocr()
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="web-medicine-ocr"
            )
            self._available = True
            logger.info(
                "yolo-ocr 已就绪: %d 类, %d 个药名, %d 个药名/别名",
                self.catalog_stats.get("categories", 0),
                self.catalog_stats.get("medicines", 0),
                self.catalog_stats.get("aliases", 0),
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("yolo-ocr 初始化失败")
            self.close()

    @property
    def is_real_model(self) -> bool:
        return self._available

    @property
    def backend_name(self) -> str:
        return "YOLO-OBB + RapidOCR + SQLite" if self._available else "不可用"

    def status(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "backend": self.backend_name,
            "model": str(self.model_path),
            "database": str(self.database_path),
            "catalog": self.catalog_stats,
            "device": str(self.device),
            "performance": {
                "yolo_ms": round(self.last_yolo_ms, 1),
                "ocr_ms": round(self.last_ocr_ms, 1),
                "yolo_warmup_ms": round(self.yolo_warmup_ms, 1),
                "ocr_warmup_ms": round(self.ocr_warmup_ms, 1),
                "conf": self.conf,
                "imgsz": self.imgsz,
                "confirm_hits": self.confirm_hits,
                "hold_frames": self.hold_frames,
                "box_smoothing": self.box_smoothing,
                "locked_box_smoothing": self.locked_box_smoothing,
                "immediate_conf": self.immediate_conf,
                "switch_check_interval": self.switch_check_interval,
                "switch_threshold": self.switch_threshold,
                "switch_confirm_hits": self.switch_confirm_hits,
            },
            "error": self.error,
        }

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if not self._available or self.model is None:
            return []
        with self._lock:
            started = time.perf_counter()
            results = self._predict(frame)
            self.last_yolo_ms = (time.perf_counter() - started) * 1000
            observed_detections = self._extract_detections(results[0], frame.shape)
            stable_detections = self._stabilize_detections(observed_detections)
            self._complete_ocr()
            self._submit_ocr(
                frame,
                [
                    detection
                    for detection in observed_detections
                    if self._states[detection.state_id].confirmed
                ],
            )
            output = [self._to_record(detection) for detection in stable_detections]
            self._frame_index += 1
            return output

    def _warm_yolo(self) -> None:
        """Warm CUDA and convolution kernels with the web camera frame shape."""
        warmup = np.zeros((480, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        self._predict(warmup)
        self.yolo_warmup_ms = (time.perf_counter() - started) * 1000
        logger.info("YOLO 预热完成: %.1f ms", self.yolo_warmup_ms)

    def _warm_ocr(self) -> None:
        """Warm all RapidOCR stages so the first medicine does not pay cold-start cost."""
        warmup = np.full((320, 480, 3), 255, dtype=np.uint8)
        for index, text in enumerate(
            ("MEDICINE", "GRANULES", "OTC", "10g x 12", "LOT 202607", "TABLETS")
        ):
            cv2.putText(
                warmup,
                text,
                (16, 44 + index * 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        started = time.perf_counter()
        try:
            self.ocr_engine(warmup)
        except Exception as exc:
            logger.warning("RapidOCR 预热失败，将在首次识别时初始化: %s", exc)
        self.ocr_warmup_ms = (time.perf_counter() - started) * 1000
        logger.info("RapidOCR 预热完成: %.1f ms", self.ocr_warmup_ms)

    def _predict(self, frame: np.ndarray):
        try:
            return self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                agnostic_nms=True,
                verbose=False,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda" not in message:
                raise
            logger.warning("GPU 推理失败，药盒识别切换到 CPU: %s", exc)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.device = "cpu"
            self.imgsz = min(self.imgsz, 320)
            return self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device="cpu",
                agnostic_nms=True,
                verbose=False,
            )

    @staticmethod
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
                detections.append(
                    Detection(polygon_norm, polygon_norm * scale, float(confidence))
                )
        else:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                rectangles = boxes.xyxyn.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                for (x1, y1, x2, y2), confidence in zip(rectangles, confidences):
                    polygon_norm = np.asarray(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    )
                    polygon_norm = np.clip(polygon_norm, 0.0, 1.0)
                    detections.append(
                        Detection(polygon_norm, polygon_norm * scale, float(confidence))
                    )

        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in ordered:
            if all(_polygon_iou(detection, previous) < 0.55 for previous in kept):
                kept.append(detection)
        return kept

    def _stabilize_detections(
        self, detections: list[Detection]
    ) -> list[Detection]:
        frame_index = self._frame_index
        unused = {
            state_id
            for state_id, state in self._states.items()
            if frame_index - state.last_seen <= max(45, self.hold_frames)
        }
        for detection in detections:
            best_state_id = None
            best_score = 0.0
            for state_id in unused:
                state = self._states[state_id]
                iou = _bbox_iou(detection.bbox, state.bbox)
                dx = (
                    detection.bbox[0]
                    + detection.bbox[2]
                    - state.bbox[0]
                    - state.bbox[2]
                ) / 2
                dy = (
                    detection.bbox[1]
                    + detection.bbox[3]
                    - state.bbox[1]
                    - state.bbox[3]
                ) / 2
                state_scale = max(
                    20.0,
                    detection.bbox[2] - detection.bbox[0],
                    detection.bbox[3] - detection.bbox[1],
                    state.bbox[2] - state.bbox[0],
                    state.bbox[3] - state.bbox[1],
                )
                center_score = max(
                    0.0, 1.0 - (dx * dx + dy * dy) ** 0.5 / state_scale
                )
                score = max(iou, center_score * 0.65)
                if score > best_score:
                    best_score = score
                    best_state_id = state_id
            if best_state_id is None or best_score < 0.25:
                best_state_id = self._next_state_id
                self._next_state_id += 1
                self._states[best_state_id] = TrackState(
                    state_id=best_state_id,
                    bbox=detection.bbox,
                    last_seen=frame_index,
                    polygon_norm=detection.polygon_norm.copy(),
                    polygon=detection.polygon.copy(),
                    confidence=detection.confidence,
                    hit_count=1,
                    confirmed=(
                        self.confirm_hits <= 1
                        or detection.confidence >= self.immediate_conf
                    ),
                )
            else:
                unused.remove(best_state_id)
                state = self._states[best_state_id]
                missed_frames = frame_index - state.last_seen
                consecutive = missed_frames <= 2
                state.hit_count = state.hit_count + 1 if consecutive else 1
                state.last_seen = frame_index

                if missed_frames > self.hold_frames:
                    state.polygon_norm = detection.polygon_norm.copy()
                    state.polygon = detection.polygon.copy()
                    state.confidence = detection.confidence
                    if state.status == "known":
                        state.appearance_changed = True
                        state.appearance_change_hits = self.switch_confirm_hits
                    state.confirmed = (
                        self.confirm_hits <= 1
                        or detection.confidence >= self.immediate_conf
                    )
                else:
                    aligned_norm = _align_polygon_vertices(
                        detection.polygon_norm, state.polygon_norm
                    )
                    aligned = _align_polygon_vertices(
                        detection.polygon, state.polygon
                    )
                    alpha = (
                        self.locked_box_smoothing
                        if state.status == "known"
                        else self.box_smoothing
                    )
                    state.polygon_norm = (
                        (1.0 - alpha) * state.polygon_norm + alpha * aligned_norm
                    ).astype(np.float32)
                    state.polygon = (
                        (1.0 - alpha) * state.polygon + alpha * aligned
                    ).astype(np.float32)
                    state.confidence = (
                        (1.0 - alpha) * state.confidence
                        + alpha * detection.confidence
                    )
                state.bbox = (
                    float(state.polygon[:, 0].min()),
                    float(state.polygon[:, 1].min()),
                    float(state.polygon[:, 0].max()),
                    float(state.polygon[:, 1].max()),
                )
                if (
                    state.hit_count >= self.confirm_hits
                    or detection.confidence >= self.immediate_conf
                ):
                    state.confirmed = True
            detection.state_id = best_state_id

        stale_ids = [
            state_id
            for state_id, state in self._states.items()
            if frame_index - state.last_seen > max(60, self.hold_frames * 4)
        ]
        for state_id in stale_ids:
            del self._states[state_id]

        stable: list[Detection] = []
        for state in self._states.values():
            missed_frames = frame_index - state.last_seen
            if not state.confirmed or missed_frames > self.hold_frames:
                continue
            stable.append(
                Detection(
                    polygon_norm=state.polygon_norm.copy(),
                    polygon=state.polygon.copy(),
                    confidence=state.confidence * (0.96 ** missed_frames),
                    state_id=state.state_id,
                    persisted=missed_frames > 0,
                )
            )
        return sorted(stable, key=lambda item: item.confidence, reverse=True)

    def _complete_ocr(self) -> None:
        if self._future is None or not self._future.done():
            return
        try:
            for state_id, recognized, error in self._future.result():
                state = self._states.get(state_id)
                if state is None:
                    continue
                state.last_completed = self._frame_index
                if error or recognized is None:
                    # A transient OCR failure must not make a confirmed result
                    # flicker. Changed packages stay in fast retry mode.
                    if state.status != "known":
                        state.medicine_name = "OCR 失败"
                        state.category_name = "待查询"
                        state.efficacy = ""
                        state.efficacy_source = ""
                        state.status = "error"
                    logger.warning("OCR 识别失败 (track=%s): %s", state_id, error)
                    continue
                self._apply_ocr_result(state, recognized)
        except Exception:
            logger.exception("OCR 后台任务失败")
        finally:
            self._future = None

    def _submit_ocr(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        if self._future is not None or self._executor is None:
            return
        tasks: list[tuple[int, np.ndarray]] = []
        for detection in detections:
            state = self._states[detection.state_id]
            if detection.area < self.min_ocr_area:
                continue

            crop: np.ndarray | None = None
            appearance_just_changed = False
            if (
                state.status == "known"
                and self._frame_index - state.last_appearance_check
                >= self.switch_check_interval
            ):
                crop = self._perspective_crop(
                    frame, detection.polygon_norm.reshape(-1)
                )
                current_signature = _appearance_signature(crop)
                state.last_appearance_check = self._frame_index
                if state.appearance_signature is None:
                    state.appearance_signature = current_signature
                else:
                    distance = _appearance_distance(
                        state.appearance_signature, current_signature
                    )
                    if distance >= self.switch_threshold:
                        state.appearance_change_hits += 1
                    else:
                        state.appearance_change_hits = 0
                        if not state.appearance_changed:
                            state.appearance_signature = _blend_appearance(
                                state.appearance_signature, current_signature
                            )
                    if state.appearance_change_hits >= self.switch_confirm_hits:
                        appearance_just_changed = not state.appearance_changed
                        state.appearance_changed = True

            last_ocr_frame = max(state.last_submitted, state.last_completed)
            interval = (
                self.ocr_interval
                if state.status != "known"
                or state.appearance_changed
                or state.candidate_match_hits > 0
                else self.ocr_known_interval
            )
            if (
                not appearance_just_changed
                and self._frame_index - last_ocr_frame < interval
            ):
                continue
            if crop is None:
                crop = self._perspective_crop(
                    frame, detection.polygon_norm.reshape(-1)
                )
            tasks.append((state.state_id, crop.copy()))
            state.last_submitted = self._frame_index
            if state.status in {"pending", "error"}:
                state.medicine_name = "识别中..."
                state.category_name = "待查询"
                state.efficacy = ""
                state.efficacy_source = ""
                state.status = "pending"
            if len(tasks) >= self.ocr_max_boxes:
                break
        if tasks:
            self._future = self._executor.submit(self._ocr_batch, tasks)

    def _ocr_batch(
        self, tasks: Sequence[tuple[int, np.ndarray]]
    ) -> list[tuple[int, dict[str, Any] | None, str | None]]:
        output = []
        for state_id, crop in tasks:
            started = time.perf_counter()
            try:
                primary = self._recognize_crop(
                    crop, self.ocr_engine, (self.ocr_angles[0],)
                )
                primary_match = self._lookup_recognized(primary)
                selected = primary
                selected_match = primary_match

                # Upright catalog hits finish after one pass. Other orientations
                # are only evaluated when the first pass is uncertain.
                confident_match = (
                    primary_match is not None
                    and (
                        primary_match.match_type == "exact"
                        or primary_match.score >= 0.92
                    )
                )
                if not confident_match and len(self.ocr_angles) > 1:
                    fallback = self._recognize_crop(
                        crop, self.ocr_engine, self.ocr_angles[1:]
                    )
                    fallback_match = self._lookup_recognized(fallback)
                    if fallback_match is not None and (
                        selected_match is None
                        or fallback_match.score > selected_match.score
                    ):
                        selected = fallback
                        selected_match = fallback_match
                    elif selected_match is None and fallback_match is None:
                        if float(fallback.get("quality", 0.0)) > float(
                            selected.get("quality", 0.0)
                        ):
                            selected = fallback

                selected["_medicine_match"] = selected_match
                selected["_appearance_signature"] = _appearance_signature(crop)
                self.last_ocr_ms = (time.perf_counter() - started) * 1000
                output.append((state_id, selected, None))
            except Exception as exc:
                self.last_ocr_ms = (time.perf_counter() - started) * 1000
                output.append((state_id, None, str(exc)))
        return output

    def _lookup_recognized(self, recognized: dict[str, Any]):
        lines = recognized.get("lines") or []
        return self.database.lookup(
            str(recognized.get("text", "")),
            lines=[str(line.get("text", "")) for line in lines],
            fuzzy_threshold=self.fuzzy_threshold,
        )

    def _apply_ocr_result(
        self, state: TrackState, recognized: dict[str, Any]
    ) -> None:
        lines = recognized.get("lines") or []
        if "_medicine_match" in recognized:
            match = recognized["_medicine_match"]
        else:
            match = self._lookup_recognized(recognized)
        signature = recognized.get("_appearance_signature")

        if state.status == "known":
            if match is None:
                # Keep the locked result while glare, motion blur, or a partly
                # inserted replacement package makes OCR temporarily uncertain.
                state.candidate_medicine_id = None
                state.candidate_match_hits = 0
                return

            if match.medicine_id == state.medicine_id:
                self._lock_recognition(state, recognized, match, signature)
                return

            if not state.appearance_changed:
                if state.candidate_medicine_id == match.medicine_id:
                    state.candidate_match_hits += 1
                else:
                    state.candidate_medicine_id = match.medicine_id
                    state.candidate_match_hits = 1
                if state.candidate_match_hits < self.switch_confirm_hits:
                    return

            self._lock_recognition(state, recognized, match, signature)
            return

        state.raw_text = str(recognized.get("text", ""))
        state.ocr_confidence = float(recognized.get("average_confidence", 0.0))
        if match is not None:
            self._lock_recognition(state, recognized, match, signature)
            return

        state.medicine_name = _best_ocr_candidate(lines)
        state.category_name = self.other_category
        state.efficacy = self.other_efficacy
        state.efficacy_source = "catalog:category"
        state.match_score = 0.0
        state.match_type = "none"
        state.status = "unknown"

    @staticmethod
    def _lock_recognition(
        state: TrackState,
        recognized: dict[str, Any],
        match: Any,
        signature: np.ndarray | None,
    ) -> None:
        state.medicine_id = match.medicine_id
        state.medicine_name = match.medicine_name
        state.category_name = match.category_name
        state.efficacy = match.efficacy
        state.efficacy_source = match.efficacy_source
        state.raw_text = str(recognized.get("text", ""))
        state.ocr_confidence = float(recognized.get("average_confidence", 0.0))
        state.match_score = match.score
        state.match_type = match.match_type
        state.status = "known"
        if isinstance(signature, np.ndarray):
            state.appearance_signature = signature.copy()
        state.appearance_change_hits = 0
        state.appearance_changed = False
        state.candidate_medicine_id = None
        state.candidate_match_hits = 0

    def _to_record(self, detection: Detection) -> dict[str, Any]:
        state = self._states[detection.state_id]
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        polygon = np.round(detection.polygon).astype(int).tolist()
        source = "YOLO+OCR+SQLite" if state.status in {"known", "unknown"} else "YOLO-OBB"
        recognition_confidence = _combined_recognition_confidence(
            detection.confidence,
            state.ocr_confidence,
            state.match_score,
            state.status,
        )
        return {
            "name": state.medicine_name,
            "category": state.category_name,
            "confidence": round(detection.confidence, 4),
            "ocr_confidence": round(state.ocr_confidence, 4),
            "match_score": round(state.match_score, 4),
            "match_type": state.match_type,
            "recognition_confidence": round(recognition_confidence, 4),
            "raw_text": state.raw_text,
            "efficacy": state.efficacy,
            "efficacy_source": state.efficacy_source,
            "status": state.status,
            "source": source,
            "persisted": detection.persisted,
            "box": [x1, y1, x2, y2],
            "polygon": polygon,
        }

    def close(self) -> None:
        self._available = False
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self.database is not None:
            try:
                self.database.close()
            except Exception:
                pass
            self.database = None

# -*- coding: utf-8 -*-
"""药盒 OBB 透视矫正、OCR 与功效关键词质检工具。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


MEDICINE_NAMES = (
    "感冒药",
    "止咳化痰",
    "消炎药",
    "止痛药",
    "退烧药",
    "肠胃药",
    "泻药通便",
    "维生素",
    "抗过敏药",
    "外用药",
    "创可贴绷带",
    "慢性病药",
    "心血管药",
    "中成药",
    "妇科用药",
    "肛肠用药",
    "保健品",
    "其他",
)

# 规则只用于筛出人工复核候选，不能自动替代医学分类标签。
MEDICINE_KEYWORDS: dict[int, dict[str, int]] = {
    0: {"感冒": 6, "解表": 3, "疏风": 2, "鼻塞": 2, "流涕": 2, "打喷嚏": 2, "恶寒": 2},
    1: {"止咳": 6, "化痰": 6, "祛痰": 6, "咳嗽": 3, "咳喘": 4, "宣肺": 3},
    2: {
        "消炎": 6,
        "抗菌": 4,
        "阿莫西林": 8,
        "头孢": 8,
        "青霉素": 8,
        "罗红霉素": 8,
        "抗生素": 6,
    },
    3: {"止痛": 6, "镇痛": 6, "疼痛": 2, "布洛芬": 8, "双氯芬酸": 8, "芬必": 8},
    4: {"退热": 8, "退烧": 8, "对乙酰氨基酚": 8, "高热": 3},
    5: {"肠胃": 6, "胃痛": 6, "胃酸": 6, "消化不良": 6, "腹泻": 4, "腹胀": 4},
    6: {"通便": 8, "便秘": 8, "泻药": 8, "乳果糖": 8, "开塞露": 8},
    7: {"维生素": 9, "vitamin": 9, "叶酸": 5, "钙片": 4},
    8: {"过敏": 8, "氯雷他定": 9, "西替利嗪": 9, "扑尔敏": 9},
    9: {"外用": 6, "软膏": 7, "乳膏": 7, "涂抹": 5, "皮肤": 4},
    10: {"创可贴": 9, "绷带": 9, "纱布": 8, "敷料": 8},
    11: {"糖尿病": 8, "降糖": 8, "二甲双胍": 9, "胰岛素": 9, "甲状腺": 5},
    12: {
        "心血管": 9,
        "冠心": 8,
        "降压": 8,
        "高血压": 8,
        "血脂": 6,
        "心绞痛": 8,
        "舒血宁": 7,
    },
    13: {"中成药": 8},
    14: {"月经": 8, "妇科": 8, "益母草": 9, "阴道": 8, "白带": 8, "痛经": 8},
    15: {"痔疮": 9, "痔": 6, "肛": 6, "马应龙": 9},
    16: {"保健食品": 9, "膳食补充": 9, "营养补充": 6, "蓝帽": 7, "增强免疫力": 6},
    17: {},
}


def normalize_text(text: str) -> str:
    """保留中英文和数字，去掉空白及标点，便于容错匹配。"""
    return re.sub(r"[\W_]+", "", str(text).casefold(), flags=re.UNICODE)


def score_text(text: str) -> tuple[dict[int, int], dict[int, list[str]]]:
    normalized = normalize_text(text)
    scores: dict[int, int] = {}
    matches: dict[int, list[str]] = {}
    for class_id, rules in MEDICINE_KEYWORDS.items():
        found = [term for term in rules if normalize_text(term) in normalized]
        matches[class_id] = found
        scores[class_id] = sum(rules[term] for term in found)
    return scores, matches


def suggest_class(text: str, min_score: int = 6, min_margin: int = 2) -> dict[str, Any]:
    scores, matches = score_text(text)
    ordered = sorted(scores, key=lambda class_id: scores[class_id], reverse=True)
    best = ordered[0]
    second_score = scores[ordered[1]] if len(ordered) > 1 else 0
    accepted = scores[best] >= min_score and scores[best] - second_score >= min_margin
    return {
        "class_id": best if accepted else None,
        "class_name": MEDICINE_NAMES[best] if accepted else None,
        "score": scores[best],
        "margin": scores[best] - second_score,
        "matched_keywords": matches[best],
        "scores": scores,
        "matches": matches,
    }


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[2] = points[np.argmax(sums)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def perspective_crop(image: np.ndarray, normalized_points: Sequence[float]) -> np.ndarray:
    """将归一化 OBB 四点透视展开为水平矩形。"""
    height, width = image.shape[:2]
    points = np.asarray(normalized_points, dtype=np.float32).reshape(4, 2)
    points *= np.asarray([width, height], dtype=np.float32)
    ordered = _order_points(points)
    target_width = max(
        np.linalg.norm(ordered[1] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[3]),
    )
    target_height = max(
        np.linalg.norm(ordered[3] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[1]),
    )
    output_width = max(8, int(round(target_width)))
    output_height = max(8, int(round(target_height)))
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (output_width, output_height))


def read_obb_labels(path: Path) -> list[tuple[int, list[float]]]:
    labels: list[tuple[int, list[float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name}:{line_number} 不是合法 OBB 九列标签")
        labels.append((int(values[0]), values[1:]))
    return labels


def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if angle not in rotations:
        raise ValueError(f"仅支持 0/90/180/270 度，收到 {angle}")
    return cv2.rotate(image, rotations[angle])


def _normalize_rapidocr_result(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for item in result or []:
        if len(item) < 3:
            continue
        box, text, confidence = item[0], str(item[1]).strip(), float(item[2])
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "confidence": round(confidence, 5),
                "box": np.asarray(box, dtype=float).round(2).tolist(),
            }
        )
    return lines


def _ocr_quality(lines: Iterable[dict[str, Any]]) -> float:
    return sum(
        float(line["confidence"]) * math.sqrt(max(1, len(normalize_text(line["text"]))))
        for line in lines
    )


def recognize_crop(
    crop: np.ndarray,
    engine: Any,
    angles: Sequence[int] = (0, 180),
) -> dict[str, Any]:
    """Run RapidOCR on one rectified medicine-box crop and select its best orientation."""
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        raise ValueError("OCR crop must be a non-empty numpy array")
    if not angles:
        raise ValueError("angles must contain at least one orientation")

    # Very small camera crops contain too few pixels for the recognizer. Upscaling
    # does not invent detail, but makes the existing strokes easier to segment.
    height, width = crop.shape[:2]
    short_side = min(height, width)
    if short_side < 96:
        scale = min(3.0, 96.0 / max(1, short_side))
        crop = cv2.resize(
            crop,
            (max(8, int(round(width * scale))), max(8, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    candidates: list[tuple[float, int, list[dict[str, Any]]]] = []
    for angle in angles:
        result, _ = engine(_rotate(crop, int(angle)))
        lines = _normalize_rapidocr_result(result)
        candidates.append((_ocr_quality(lines), int(angle), lines))

    quality, selected_angle, lines = max(candidates, key=lambda item: item[0])
    text = "\n".join(line["text"] for line in lines)
    average_confidence = (
        sum(float(line["confidence"]) for line in lines) / len(lines) if lines else 0.0
    )
    return {
        "text": text,
        "line_count": len(lines),
        "average_confidence": round(average_confidence, 5),
        "selected_angle": selected_angle,
        "quality": round(quality, 5),
        "crop_shape": list(crop.shape[:2]),
        "lines": lines,
    }


def run_obb_ocr(
    image_path: Path,
    label_path: Path,
    engine: Any,
    angles: Sequence[int] = (0, 180),
) -> dict[str, Any]:
    """逐目标透视矫正后 OCR，并为每个目标选择质量最高的方向。"""
    image = _read_image(image_path)
    objects: list[dict[str, Any]] = []
    all_lines: list[dict[str, Any]] = []
    for object_index, (class_id, points) in enumerate(read_obb_labels(label_path)):
        crop = perspective_crop(image, points)
        recognized = recognize_crop(crop, engine, angles)
        lines = recognized["lines"]
        for line in lines:
            line["object_index"] = object_index
        all_lines.extend(lines)
        objects.append(
            {
                "object_index": object_index,
                "class_id": class_id,
                "class_name": MEDICINE_NAMES[class_id],
                "selected_angle": recognized["selected_angle"],
                "quality": recognized["quality"],
                "crop_shape": recognized["crop_shape"],
                "lines": lines,
            }
        )

    text = "\n".join(line["text"] for line in all_lines)
    average_confidence = (
        sum(float(line["confidence"]) for line in all_lines) / len(all_lines) if all_lines else 0.0
    )
    return {
        "image": str(image_path.resolve()),
        "label": str(label_path.resolve()),
        "text": text,
        "line_count": len(all_lines),
        "average_confidence": round(average_confidence, 5),
        "objects": objects,
        "lines": all_lines,
        "ocr_method": "rapidocr_obb_rectified",
    }


def load_ocr_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        result = json.load(stream)
    result.setdefault("text", "")
    result.setdefault("lines", [])
    result.setdefault("line_count", len(result["lines"]))
    result.setdefault("average_confidence", 0.0)
    return result

"""药盒OCR与医疗关键信息结构化提取。

功能：
1. 对 data/medicine/images/train 下1～10号图片执行PaddleOCR；
2. 保留OCR文本、坐标和置信度；
3. 过滤网址、拼音、包装说明语等低价值信息；
4. 提取药品名称、批准文号、规格、生产企业、功能主治和药品属性；
5. 输出适合本地大语言模型使用的JSON。

依赖：
    pip install paddlepaddle paddleocr opencv-python numpy

运行：
    python ocr.py
"""

from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Paddle环境配置
# 必须放在import paddle或paddleocr之前
# ============================================================

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")


# 规避Windows上aiohttp加载系统证书库报错
_orig_load_default_certs = ssl.SSLContext.load_default_certs


def _patched_load_default_certs(
    self,
    purpose=ssl.Purpose.SERVER_AUTH,
):
    try:
        _orig_load_default_certs(self, purpose)
    except ssl.SSLError:
        pass


ssl.SSLContext.load_default_certs = _patched_load_default_certs


# ============================================================
# 路径配置
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE

_TRAIN = ROOT / "data" / "medicine" / "images" / "train"

IMG_DIRS = (
    _TRAIN / "images",
    _TRAIN,
)

IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

IDS = range(1, 11)

OUTPUT_DIR = HERE / "ocr_output"


# ============================================================
# 参数配置
# ============================================================

# OCR最低置信度
OCR_SCORE_THRESHOLD = 0.55

# 药名候选最低综合分
MEDICINE_NAME_MIN_SCORE = 4.0

# 是否打印原始OCR明细
PRINT_RAW_OCR_ITEMS = False

# 是否在JSON中保存调试信息
SAVE_DEBUG_INFO = False

# 是否保存JSON
SAVE_JSON = True


# ============================================================
# 数据结构
# ============================================================

@dataclass
class OCRItem:
    """单条OCR识别结果。"""

    text: str
    score: float = 1.0
    box: list[list[float]] = field(default_factory=list)

    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    is_noise: bool = False
    noise_reason: str = ""


@dataclass
class MedicineInfo:
    """提供给医疗助手的核心药品信息。"""

    medicine_name: str = ""
    approval_number: str = ""
    specification: str = ""
    manufacturer: str = ""
    indication: str = ""

    prescription_type: str = ""
    overall_confidence: str = ""

    fields_to_verify: list[str] = field(default_factory=list)

    # 默认不传给大模型，仅用于调试
    debug: dict[str, Any] = field(default_factory=dict)


# ============================================================
# OCR常见纠错
# ============================================================

OCR_REPLACEMENTS = {
    "国码准字": "国药准字",
    "国药淮字": "国药准字",
    "国药准学": "国药准字",
    "国药谁字": "国药准字",
    "国药准宇": "国药准字",
    "国药推字": "国药准字",
    "国药难字": "国药准字",
    "批准文亏": "批准文号",
    "批准义号": "批准文号",
    "功能王治": "功能主治",
    "功效主治": "功能主治",
    "功脑主治": "功能主治",
    "功脑电油": "功能主治",
    "有限公可": "有限公司",
    "有限公同": "有限公司",
    "生产企亚": "生产企业",
    "化痕": "化痰",
    "宣肺止该": "宣肺止咳",
    "毫开": "毫升",
    "毫异": "毫升",
}


DOSAGE_FORMS = [
    "颗粒剂",
    "分散片",
    "缓释片",
    "咀嚼片",
    "泡腾片",
    "肠溶片",
    "薄膜衣片",
    "注射液",
    "口服液",
    "滴眼液",
    "滴鼻液",
    "喷雾剂",
    "混悬液",
    "糖浆",
    "软胶囊",
    "硬胶囊",
    "胶囊",
    "颗粒",
    "片剂",
    "胶丸",
    "丸剂",
    "水丸",
    "蜜丸",
    "浓缩丸",
    "散剂",
    "软膏",
    "乳膏",
    "凝胶",
    "贴膏",
    "贴剂",
    "栓剂",
    "酊剂",
    "合剂",
    "药酒",
    "煎膏",
    "滴丸",
    "冲剂",
    "丸",
    "片",
    "散",
    "膏",
]


MANUFACTURER_SUFFIXES = [
    "集团股份有限公司",
    "股份有限公司",
    "生物制药有限公司",
    "制药有限公司",
    "药业有限公司",
    "医药有限公司",
    "中药有限公司",
    "制药总厂",
    "制药厂",
    "中药厂",
    "有限公司",
]


NOISE_KEYWORDS = [
    "请仔细阅读说明书",
    "按说明书使用",
    "按说明使用",
    "在药师指导下",
    "购买和使用",
    "电子商务",
    "扫码",
    "官方网站",
    "服务热线",
    "客服电话",
    "更多信息",
    "温馨提示",
    "谨遵医嘱",
    "本品不能代替药物",
]


FIELD_KEYWORDS = [
    "国药准字",
    "批准文号",
    "功能主治",
    "适应症",
    "用法用量",
    "不良反应",
    "注意事项",
    "禁忌",
    "有效期",
    "生产企业",
    "生产厂家",
    "执行标准",
    "贮藏",
    "成份",
    "规格",
]


BRAND_WORDS = [
    "白云山",
    "同仁堂",
    "朗致",
    "石药",
    "美罗",
    "金鸡",
    "坤康",
    "仲景",
    "仙景",
    "金戈",
]


INDICATION_ACTION_WORDS = [
    "清热",
    "解毒",
    "止咳",
    "化痰",
    "祛痰",
    "宣肺",
    "疏风",
    "散热",
    "止痛",
    "消炎",
    "抗菌",
    "活血",
    "调经",
    "祛风",
    "除湿",
    "舒筋",
    "通络",
    "健胃",
    "消食",
    "退热",
    "缓解",
    "治疗",
    "用于",
    "祛湿",
    "止泻",
    "通便",
    "抗过敏",
]


# ============================================================
# 正则表达式
# ============================================================

# 严格批准文号：国药准字+字母+8位数字
STRICT_APPROVAL_PATTERN = re.compile(
    r"国药准字[HZSJBTF]\d{8}",
    re.IGNORECASE,
)

# 宽松批准文号，仅用于发现疑似错误结果
LOOSE_APPROVAL_PATTERN = re.compile(
    r"国药准字[A-Z]?\d{7,10}",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"(?:https?://|www\.|[a-zA-Z0-9-]+\.(?:com|cn|net|org))",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?:400[-\s]?\d{3}[-\s]?\d{4})|"
    r"(?:1[3-9]\d{9})|"
    r"(?:0\d{2,3}[-\s]?\d{7,8})"
)

SPEC_PATTERN = re.compile(
    r"""
    (?:
        \d+(?:\.\d+)?
        \s*
        (?:
            mg|g|kg|ml|μg|ug|iu|
            毫克|克|千克|毫升|微克|单位|
            粒|片|袋|支|板|瓶|盒|贴|丸
        )
        (?:
            \s*[×xX*]\s*
            \d+(?:\.\d+)?
            \s*
            (?:粒|片|袋|支|板|瓶|盒|贴|丸)
        )*
        (?:\s*/\s*(?:盒|瓶|袋))?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

ENGLISH_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z\s,\-.'()]{2,}$"
)

NUMBER_ONLY_PATTERN = re.compile(
    r"^[\d\s×xX*/.%-]+$"
)


# ============================================================
# 基础工具函数
# ============================================================

def _find_image(stem: str) -> str | None:
    for directory in IMG_DIRS:
        if not directory.is_dir():
            continue

        for ext in IMAGE_EXTS:
            path = directory / f"{stem}{ext}"

            if path.is_file():
                return str(path)

            upper_path = directory / f"{stem}{ext.upper()}"

            if upper_path.is_file():
                return str(upper_path)

    return None


def normalize_text(text: str) -> str:
    """清理OCR文本中的空白、符号和常见识别错误。"""
    text = str(text).strip()

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = text.replace("（", "(")
    text = text.replace("）", ")")
    text = text.replace("：", ":")
    text = text.replace("Ｘ", "×")

    # 只在数字单位表达式附近统一x/X
    text = re.sub(
        r"(?<=\d)\s*[xX]\s*(?=\d)",
        "×",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    for wrong, correct in OCR_REPLACEMENTS.items():
        text = text.replace(wrong, correct)

    return text.strip()


def compact_text(text: str) -> str:
    """去掉全部空白，用于关键词与正则匹配。"""
    return re.sub(r"\s+", "", text)


def _safe_float(
    value: Any,
    default: float = 1.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    try:
        return list(value)
    except TypeError:
        return [value]


def calculate_box_features(
    box: list[list[float]],
) -> tuple[float, float, float, float]:
    """计算OCR框中心、宽度和高度。"""
    if not box or len(box) < 4:
        return 0.0, 0.0, 0.0, 0.0

    try:
        points = np.asarray(
            box,
            dtype=np.float32,
        )

        center_x = float(points[:, 0].mean())
        center_y = float(points[:, 1].mean())

        width_top = np.linalg.norm(
            points[1] - points[0]
        )
        width_bottom = np.linalg.norm(
            points[2] - points[3]
        )

        height_left = np.linalg.norm(
            points[3] - points[0]
        )
        height_right = np.linalg.norm(
            points[2] - points[1]
        )

        width = float(
            (width_top + width_bottom) / 2
        )
        height = float(
            (height_left + height_right) / 2
        )

        return center_x, center_y, width, height

    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def create_ocr_item(
    text: str,
    score: float = 1.0,
    box: list[list[float]] | None = None,
) -> OCRItem:
    text = normalize_text(text)
    box = box or []

    center_x, center_y, width, height = (
        calculate_box_features(box)
    )

    return OCRItem(
        text=text,
        score=_safe_float(score),
        box=box,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
    )


# ============================================================
# PaddleOCR初始化
# ============================================================

def _init_ocr():
    try:
        from paddleocr import PaddleOCR

    except ImportError as exc:
        raise SystemExit(
            "未安装PaddleOCR，请执行：\n"
            "pip install paddlepaddle paddleocr"
        ) from exc

    candidate_kwargs = (
        {
            "lang": "ch",
            "enable_mkldnn": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": "ch",
            "enable_mkldnn": False,
        },
        {
            "lang": "ch",
        },
    )

    last_error: Exception | None = None

    for kwargs in candidate_kwargs:
        try:
            return PaddleOCR(**kwargs)

        except (TypeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        f"PaddleOCR初始化失败：{last_error}"
    )


# ============================================================
# PaddleOCR 2.x / 3.x输出解析
# ============================================================

def _extract_from_dict(
    record: dict[str, Any],
) -> list[OCRItem]:
    """解析PaddleOCR 3.x字典输出。"""
    items: list[OCRItem] = []

    texts = (
        record.get("rec_texts")
        or record.get("texts")
        or record.get("text")
        or record.get("rec_text")
    )

    scores = (
        record.get("rec_scores")
        or record.get("scores")
        or record.get("score")
        or record.get("rec_score")
    )

    boxes = (
        record.get("rec_polys")
        or record.get("dt_polys")
        or record.get("boxes")
        or record.get("polys")
        or record.get("box")
    )

    if isinstance(texts, str):
        text_list = [texts]
    else:
        text_list = _to_list(texts)

    score_list = _to_list(scores)
    box_list = _to_list(boxes)

    if not text_list:
        return items

    for index, text in enumerate(text_list):
        if text is None:
            continue

        score = (
            score_list[index]
            if index < len(score_list)
            else 1.0
        )

        box = (
            box_list[index]
            if index < len(box_list)
            else []
        )

        if isinstance(box, np.ndarray):
            box = box.tolist()

        if not isinstance(box, list):
            box = []

        item = create_ocr_item(
            text=str(text),
            score=_safe_float(score),
            box=box,
        )

        if item.text:
            items.append(item)

    return items


def _extract_from_object(
    record: Any,
) -> list[OCRItem]:
    """解析PaddleOCR 3.x对象输出。"""
    possible_dict: dict[str, Any] = {}

    attribute_names = (
        "rec_texts",
        "texts",
        "text",
        "rec_text",
        "rec_scores",
        "scores",
        "score",
        "rec_score",
        "rec_polys",
        "dt_polys",
        "boxes",
        "polys",
        "box",
    )

    for name in attribute_names:
        if hasattr(record, name):
            possible_dict[name] = getattr(
                record,
                name,
            )

    if hasattr(record, "get"):
        for name in attribute_names:
            try:
                value = record.get(name)

                if value is not None:
                    possible_dict[name] = value

            except Exception:
                pass

    if possible_dict:
        return _extract_from_dict(possible_dict)

    return []


def _extract_v2_line(
    item: Any,
) -> OCRItem | None:
    """解析PaddleOCR 2.x单行结果。

    典型结构：
    [
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        ("文字", 0.98)
    ]
    """
    if not isinstance(item, (list, tuple)):
        return None

    if len(item) < 2:
        return None

    box = item[0]
    recognition = item[1]

    if not isinstance(
        recognition,
        (list, tuple),
    ):
        return None

    if len(recognition) < 1:
        return None

    text = recognition[0]

    score = (
        recognition[1]
        if len(recognition) > 1
        else 1.0
    )

    if isinstance(box, np.ndarray):
        box = box.tolist()

    if not isinstance(box, list):
        box = []

    result = create_ocr_item(
        text=str(text),
        score=_safe_float(score),
        box=box,
    )

    return result if result.text else None


def parse_ocr_output(
    out: Any,
) -> list[OCRItem]:
    """兼容PaddleOCR 2.x与3.x输出。"""
    if out is None:
        return []

    if isinstance(out, dict):
        return _extract_from_dict(out)

    if not isinstance(
        out,
        (list, tuple),
    ):
        return _extract_from_object(out)

    items: list[OCRItem] = []

    for record in out:
        if record is None:
            continue

        if isinstance(record, dict):
            items.extend(
                _extract_from_dict(record)
            )
            continue

        object_items = _extract_from_object(record)

        if object_items:
            items.extend(object_items)
            continue

        if isinstance(record, list):
            single_item = _extract_v2_line(record)

            if single_item is not None:
                items.append(single_item)
                continue

            for sub_record in record:
                line_item = _extract_v2_line(
                    sub_record
                )

                if line_item is not None:
                    items.append(line_item)
                    continue

                if isinstance(sub_record, dict):
                    items.extend(
                        _extract_from_dict(
                            sub_record
                        )
                    )

    return items


def run_ocr(
    ocr,
    img_bgr: np.ndarray,
) -> list[OCRItem]:
    """运行OCR并返回结构化OCR行。"""
    try:
        if hasattr(ocr, "predict"):
            try:
                output = ocr.predict(
                    img_bgr,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )

            except TypeError:
                output = ocr.predict(img_bgr)

            return parse_ocr_output(output)

        if hasattr(ocr, "ocr"):
            try:
                output = ocr.ocr(img_bgr)

            except TypeError:
                output = ocr.ocr(
                    img_bgr,
                    cls=True,
                )

            return parse_ocr_output(output)

    except Exception as exc:
        print(f"[OCR错误] {exc}")
        return []

    return []


# ============================================================
# 文本过滤
# ============================================================

def is_mostly_english(
    text: str,
) -> bool:
    compact = compact_text(text)

    if not compact:
        return False

    english_count = len(
        re.findall(r"[A-Za-z]", compact)
    )

    chinese_count = len(
        re.findall(
            r"[\u4e00-\u9fff]",
            compact,
        )
    )

    return (
        english_count >= 3
        and chinese_count == 0
        and english_count
        / max(len(compact), 1)
        > 0.75
    )


def is_noise_text(
    text: str,
    score: float = 1.0,
) -> tuple[bool, str]:
    """判断OCR文字是否为低价值噪声。"""
    normalized = normalize_text(text)
    compact = compact_text(normalized)
    lower = compact.lower()

    if not compact:
        return True, "空文本"

    if len(compact) <= 1:
        return True, "文本过短"

    if URL_PATTERN.search(normalized):
        return True, "网址"

    if PHONE_PATTERN.search(normalized):
        return True, "电话号码"

    for keyword in NOISE_KEYWORDS:
        normalized_keyword = compact_text(
            keyword
        ).lower()

        if normalized_keyword in lower:
            return True, f"固定说明语:{keyword}"

    # 即使说明语被OCR识别错，只要命中多个片段也过滤
    instruction_fragments = [
        "阅读",
        "说明",
        "明书",
        "药师",
        "指导",
        "购买",
        "使用",
    ]

    fragment_hits = sum(
        fragment in compact
        for fragment in instruction_fragments
    )

    if (
        len(compact) >= 18
        and fragment_hits >= 2
    ):
        return True, "包装固定说明语"

    # 很长、低置信度且不像适应症的文本
    medical_content_keywords = [
        "功能主治",
        "适应症",
        "用于",
        "治疗",
        "缓解",
        "预防",
    ]

    if (
        len(compact) >= 20
        and score < 0.70
        and not any(
            keyword in compact
            for keyword
            in medical_content_keywords
        )
    ):
        return True, "低置信度长文本"

    if NUMBER_ONLY_PATTERN.fullmatch(compact):
        return True, "纯数字或符号"

    if is_mostly_english(normalized):
        upper = compact.upper()

        if upper not in {"OTC", "RX"}:
            return True, "英文或拼音"

    return False, ""


def deduplicate_items(
    items: list[OCRItem],
) -> list[OCRItem]:
    """相同文本只保留置信度最高的一项。"""
    best_map: dict[str, OCRItem] = {}

    for item in items:
        key = compact_text(
            item.text
        ).lower()

        if not key:
            continue

        old = best_map.get(key)

        if (
            old is None
            or item.score > old.score
        ):
            best_map[key] = item

    return list(best_map.values())


def filter_ocr_items(
    items: list[OCRItem],
) -> tuple[
    list[OCRItem],
    list[OCRItem],
    list[OCRItem],
]:
    """划分有效、低置信度和噪声文本。"""
    valid_items: list[OCRItem] = []
    low_confidence_items: list[OCRItem] = []
    noise_items: list[OCRItem] = []

    for item in items:
        item.text = normalize_text(
            item.text
        )

        noise, reason = is_noise_text(
            item.text,
            item.score,
        )

        if noise:
            item.is_noise = True
            item.noise_reason = reason
            noise_items.append(item)
            continue

        if item.score < OCR_SCORE_THRESHOLD:
            low_confidence_items.append(
                item
            )
            continue

        valid_items.append(item)

    valid_items = deduplicate_items(
        valid_items
    )

    low_confidence_items = (
        deduplicate_items(
            low_confidence_items
        )
    )

    noise_items = deduplicate_items(
        noise_items
    )

    return (
        valid_items,
        low_confidence_items,
        noise_items,
    )


# ============================================================
# 字段提取
# ============================================================

def join_texts(
    items: list[OCRItem],
) -> str:
    return " ".join(
        item.text
        for item in items
        if item.text
    )


def extract_approval_number(
    text: str,
) -> tuple[str, str]:
    """返回可靠批准文号和疑似批准文号。

    严格格式：
        国药准字+字母+8位数字

    OCR缺少字母时不自行猜测。
    """
    compact = compact_text(
        normalize_text(text)
    ).upper()

    strict_match = (
        STRICT_APPROVAL_PATTERN.search(
            compact
        )
    )

    if strict_match:
        return strict_match.group(0), ""

    loose_match = (
        LOOSE_APPROVAL_PATTERN.search(
            compact
        )
    )

    if loose_match:
        return "", loose_match.group(0)

    return "", ""


def extract_specifications(
    text: str,
) -> list[str]:
    normalized = normalize_text(text)

    matches = [
        normalize_text(match.group(0))
        for match
        in SPEC_PATTERN.finditer(
            normalized
        )
    ]

    result: list[str] = []

    for value in matches:
        value = value.replace(" ", "")

        if value not in result:
            result.append(value)

    return result


def manufacturer_score(
    text: str,
) -> float:
    compact = compact_text(text)
    score = 0.0

    for suffix in MANUFACTURER_SUFFIXES:
        if suffix in compact:
            score += 5.0

    if any(
        keyword in compact
        for keyword in [
            "制药",
            "药业",
            "医药",
            "中药厂",
        ]
    ):
        score += 2.0

    if 6 <= len(compact) <= 40:
        score += 1.0

    if any(
        keyword in compact
        for keyword in [
            "功能主治",
            "用于",
            "规格",
        ]
    ):
        score -= 4.0

    return score


def extract_manufacturer(
    items: list[OCRItem],
) -> str:
    candidates: list[
        tuple[float, str]
    ] = []

    for item in items:
        score = manufacturer_score(
            item.text
        )

        if score >= 4.0:
            candidates.append(
                (score, item.text)
            )

    if not candidates:
        return ""

    candidates.sort(
        key=lambda value: (
            value[0],
            len(compact_text(value[1])),
        ),
        reverse=True,
    )

    return candidates[0][1]


def is_field_text(
    text: str,
) -> bool:
    compact = compact_text(text)

    return any(
        keyword in compact
        for keyword in FIELD_KEYWORDS
    )


def medicine_name_score(
    item: OCRItem,
    image_height: int,
) -> float:
    """计算某条OCR文本作为药品名称的可能性。"""
    text = normalize_text(item.text)
    compact = compact_text(text)

    if not compact:
        return -100.0

    score = 0.0
    length = len(compact)

    # 字数
    if 3 <= length <= 18:
        score += 2.0
    elif 19 <= length <= 25:
        score += 0.5
    else:
        score -= 2.0

    # 剂型结尾是药名的强特征
    for dosage_form in DOSAGE_FORMS:
        if compact.endswith(
            dosage_form
        ):
            score += 6.0
            break

    # 中文字符比例
    chinese_count = len(
        re.findall(
            r"[\u4e00-\u9fff]",
            compact,
        )
    )

    if chinese_count >= 3:
        score += 1.5

    if (
        chinese_count
        / max(length, 1)
        > 0.7
    ):
        score += 1.0

    # OCR置信度
    score += (
        min(
            max(item.score, 0.0),
            1.0,
        )
        * 2.0
    )

    # 字体框高度
    if item.height > 0:
        if item.height >= 30:
            score += 2.0

        elif item.height >= 20:
            score += 1.0

    # 药名一般位于包装上部或中部
    if (
        image_height > 0
        and item.center_y > 0
    ):
        relative_y = (
            item.center_y
            / image_height
        )

        if 0.08 <= relative_y <= 0.65:
            score += 1.0

        if relative_y > 0.88:
            score -= 1.5

    # 字段文字排除
    if is_field_text(compact):
        score -= 8.0

    # 厂家排除
    if manufacturer_score(compact) >= 4.0:
        score -= 8.0

    # 规格排除
    if SPEC_PATTERN.fullmatch(compact):
        score -= 6.0

    # 批准文号排除
    if "国药准字" in compact:
        score -= 8.0

    # 品牌名排除
    if compact in BRAND_WORDS:
        score -= 4.0

    # 数字太多
    digit_count = len(
        re.findall(r"\d", compact)
    )

    if digit_count >= 2:
        score -= 2.0

    # 说明句排除
    if any(
        keyword in compact
        for keyword in [
            "用于",
            "功能主治",
            "适应症",
            "请仔细阅读",
            "有效期",
            "治疗",
            "规格",
            "生产",
        ]
    ):
        score -= 4.0

    return score


def extract_medicine_name(
    items: list[OCRItem],
    image_height: int,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    candidates: list[
        dict[str, Any]
    ] = []

    for item in items:
        score = medicine_name_score(
            item=item,
            image_height=image_height,
        )

        if score > 0:
            candidates.append(
                {
                    "text": item.text,
                    "score": round(
                        score,
                        3,
                    ),
                    "ocr_score": round(
                        item.score,
                        4,
                    ),
                    "center_y": round(
                        item.center_y,
                        2,
                    ),
                    "height": round(
                        item.height,
                        2,
                    ),
                }
            )

    candidates.sort(
        key=lambda value: value["score"],
        reverse=True,
    )

    if not candidates:
        return "", []

    best = candidates[0]

    if (
        best["score"]
        < MEDICINE_NAME_MIN_SCORE
    ):
        return "", candidates[:5]

    return best["text"], candidates[:5]


def extract_otc(
    text: str,
) -> tuple[bool | None, str]:
    upper = compact_text(
        text
    ).upper()

    if "OTC" in upper:
        return True, "OTC非处方药"

    if "处方药" in text:
        return False, "处方药"

    if re.search(
        r"\bRX\b",
        text,
        re.IGNORECASE,
    ):
        return False, "处方药"

    return None, ""


def indication_score(
    text: str,
) -> float:
    compact = compact_text(text)
    score = 0.0

    action_hits = sum(
        keyword in compact
        for keyword
        in INDICATION_ACTION_WORDS
    )

    score += action_hits * 2.0

    if 4 <= len(compact) <= 40:
        score += 1.0

    if (
        "功能主治" in compact
        or "适应症" in compact
    ):
        score += 3.0

    if (
        "用于" in compact
        or "治疗" in compact
    ):
        score += 2.0

    if any(
        keyword in compact
        for keyword in [
            "有限公司",
            "国药准字",
            "有效期",
            "规格",
            "批准文号",
        ]
    ):
        score -= 8.0

    return score


def extract_indication(
    items: list[OCRItem],
) -> str:
    candidates: list[
        tuple[float, str]
    ] = []

    for item in items:
        score = indication_score(
            item.text
        )

        if score >= 3.0:
            candidates.append(
                (score, item.text)
            )

    if not candidates:
        return ""

    candidates.sort(
        key=lambda value: (
            value[0],
            len(value[1]),
        ),
        reverse=True,
    )

    best_text = candidates[0][1]

    # 再执行一次常见纠错
    for wrong, correct in (
        ("化痕", "化痰"),
        ("宣肺止该", "宣肺止咳"),
        ("功脑电油", "功能主治"),
    ):
        best_text = best_text.replace(
            wrong,
            correct,
        )

    # 去除字段前缀符号
    best_text = re.sub(
        r"^[【\[\(]?"
        r"(?:功能主治|适应症)"
        r"[】\]\)]?[:：]?",
        "",
        best_text,
    ).strip()

    return best_text


def calculate_overall_confidence(
    medicine_name: str,
    name_candidates: list[
        dict[str, Any]
    ],
    approval_number: str,
    specification: str,
    manufacturer: str,
    indication: str,
) -> str:
    score = 0

    if medicine_name:
        score += 3

    if name_candidates:
        best_ocr_score = (
            name_candidates[0].get(
                "ocr_score",
                0.0,
            )
        )

        if best_ocr_score >= 0.90:
            score += 2

        elif best_ocr_score >= 0.75:
            score += 1

    if approval_number:
        score += 2

    if specification:
        score += 1

    if manufacturer:
        score += 1

    if indication:
        score += 1

    if score >= 8:
        return "高"

    if score >= 5:
        return "较高"

    if score >= 3:
        return "一般"

    return "低"


def build_medicine_info(
    all_items: list[OCRItem],
    image_shape: tuple[int, ...],
) -> MedicineInfo:
    image_height = (
        image_shape[0]
        if image_shape
        else 0
    )

    (
        valid_items,
        low_items,
        noise_items,
    ) = filter_ocr_items(all_items)

    combined_text = join_texts(
        valid_items + low_items
    )

    (
        approval_number,
        suspicious_approval,
    ) = extract_approval_number(
        combined_text
    )

    specifications = (
        extract_specifications(
            combined_text
        )
    )

    specification = (
        specifications[0]
        if specifications
        else ""
    )

    manufacturer = extract_manufacturer(
        valid_items
    )

    (
        medicine_name,
        name_candidates,
    ) = extract_medicine_name(
        valid_items,
        image_height=image_height,
    )

    (
        _,
        prescription_type,
    ) = extract_otc(combined_text)

    indication = extract_indication(
        valid_items
    )

    fields_to_verify: list[str] = []

    if not medicine_name:
        fields_to_verify.append(
            "药品名称"
        )

    if not approval_number:
        fields_to_verify.append(
            "批准文号"
        )

    if not specification:
        fields_to_verify.append(
            "规格"
        )

    if suspicious_approval:
        fields_to_verify.append(
            f"疑似批准文号："
            f"{suspicious_approval}"
        )

    overall_confidence = (
        calculate_overall_confidence(
            medicine_name=medicine_name,
            name_candidates=name_candidates,
            approval_number=approval_number,
            specification=specification,
            manufacturer=manufacturer,
            indication=indication,
        )
    )

    debug_info: dict[str, Any] = {}

    if SAVE_DEBUG_INFO:
        debug_info = {
            "medicine_name_candidates": (
                name_candidates[:5]
            ),
            "cleaned_text": [
                {
                    "text": item.text,
                    "score": round(
                        item.score,
                        4,
                    ),
                }
                for item in valid_items
            ],
            "low_confidence_text": [
                {
                    "text": item.text,
                    "score": round(
                        item.score,
                        4,
                    ),
                }
                for item in low_items
            ],
            "noise_text": [
                {
                    "text": item.text,
                    "score": round(
                        item.score,
                        4,
                    ),
                    "reason": (
                        item.noise_reason
                    ),
                }
                for item in noise_items
            ],
            "raw_ocr_items": [
                {
                    "text": item.text,
                    "score": round(
                        item.score,
                        4,
                    ),
                    "box": item.box,
                    "center_x": round(
                        item.center_x,
                        2,
                    ),
                    "center_y": round(
                        item.center_y,
                        2,
                    ),
                    "width": round(
                        item.width,
                        2,
                    ),
                    "height": round(
                        item.height,
                        2,
                    ),
                }
                for item in all_items
            ],
        }

    return MedicineInfo(
        medicine_name=medicine_name,
        approval_number=approval_number,
        specification=specification,
        manufacturer=manufacturer,
        indication=indication,
        prescription_type=prescription_type,
        overall_confidence=overall_confidence,
        fields_to_verify=fields_to_verify,
        debug=debug_info,
    )


# ============================================================
# 输出函数
# ============================================================

def print_ocr_items(
    items: list[OCRItem],
) -> None:
    if not items:
        print("     没有识别到文字")
        return

    for index, item in enumerate(
        items,
        start=1,
    ):
        print(
            f"     [{index:02d}] "
            f"score={item.score:.3f} "
            f"pos=("
            f"{item.center_x:.1f}, "
            f"{item.center_y:.1f}) "
            f"size=("
            f"{item.width:.1f}, "
            f"{item.height:.1f}) "
            f"text={item.text}"
        )


def print_medicine_info(
    info: MedicineInfo,
) -> None:
    print("     医疗助手关键结果：")

    print(
        f"       药品名称："
        f"{info.medicine_name or '未可靠识别'}"
    )

    print(
        f"       批准文号："
        f"{info.approval_number or '未可靠识别'}"
    )

    print(
        f"       规格："
        f"{info.specification or '未识别'}"
    )

    print(
        f"       生产企业："
        f"{info.manufacturer or '未识别'}"
    )

    print(
        f"       功能主治："
        f"{info.indication or '未可靠识别'}"
    )

    print(
        f"       药品属性："
        f"{info.prescription_type or '未确定'}"
    )

    print(
        f"       识别可信度："
        f"{info.overall_confidence}"
    )

    if info.fields_to_verify:
        print(
            "       需要人工核对："
            + "；".join(
                info.fields_to_verify
            )
        )


def save_json_result(
    image_path: str,
    info: MedicineInfo,
) -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_name = Path(
        image_path
    ).stem

    output_path = (
        OUTPUT_DIR
        / f"{image_name}.json"
    )

    medicine_data = asdict(info)

    if not SAVE_DEBUG_INFO:
        medicine_data.pop(
            "debug",
            None,
        )

    payload = {
        "image": Path(
            image_path
        ).name,
        "medicine": medicine_data,
        "safety_notice": (
            "识别结果来自药盒图像OCR。"
            "药品名称、批准文号和规格应由用户核对，"
            "不得仅依据本结果自行决定用药。"
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def build_llm_input(
    info: MedicineInfo,
) -> dict[str, Any]:
    """构造后续本地大语言模型的输入。"""
    return {
        "medicine_name": (
            info.medicine_name
        ),
        "approval_number": (
            info.approval_number
        ),
        "specification": (
            info.specification
        ),
        "manufacturer": (
            info.manufacturer
        ),
        "indication": (
            info.indication
        ),
        "prescription_type": (
            info.prescription_type
        ),
        "overall_confidence": (
            info.overall_confidence
        ),
        "fields_to_verify": (
            info.fields_to_verify
        ),
    }


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("药盒OCR与医疗关键信息提取")
    print("=" * 70)

    print(f"查找目录：{IMG_DIRS[0]}")
    print(
        f"OCR最低置信度："
        f"{OCR_SCORE_THRESHOLD}"
    )
    print("初始化PaddleOCR……")

    ocr = _init_ocr()

    print("初始化完成，开始识别。\n")

    success_count = 0
    missing_count = 0

    for image_id in IDS:
        stem = str(image_id)
        image_path = _find_image(stem)

        if image_path is None:
            print(
                f"[{image_id:>2}] "
                f"未找到图片，已尝试"
                f"{stem}.jpg、"
                f"{stem}.png等格式"
            )

            missing_count += 1
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(
                f"[{image_id:>2}] "
                f"图片读取失败："
                f"{image_path}"
            )

            missing_count += 1
            continue

        print("-" * 70)

        print(
            f"[{image_id:>2}] "
            f"{os.path.basename(image_path)}"
        )

        ocr_items = run_ocr(
            ocr=ocr,
            img_bgr=image,
        )

        if PRINT_RAW_OCR_ITEMS:
            print(
                f"     OCR共识别到"
                f"{len(ocr_items)}条文本"
            )

            print("     OCR明细：")
            print_ocr_items(ocr_items)

        medicine_info = (
            build_medicine_info(
                all_items=ocr_items,
                image_shape=image.shape,
            )
        )

        print_medicine_info(
            medicine_info
        )

        # 后续连接本地大模型时，
        # 只需要使用这个llm_input
        llm_input = build_llm_input(
            medicine_info
        )

        if SAVE_JSON:
            json_path = save_json_result(
                image_path=image_path,
                info=medicine_info,
            )

            print(
                f"     JSON已保存："
                f"{json_path}"
            )

        if SAVE_DEBUG_INFO:
            print(
                "     LLM输入："
                + json.dumps(
                    llm_input,
                    ensure_ascii=False,
                )
            )

        print()
        success_count += 1

    print("=" * 70)

    print(
        f"完成：成功识别"
        f"{success_count}张，"
        f"缺失或失败"
        f"{missing_count}张"
    )

    if SAVE_JSON:
        print(
            f"JSON输出目录："
            f"{OUTPUT_DIR}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
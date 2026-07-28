"""YOLO + OCR + LLM 融合模块

流程：
  YOLO检测 → 置信度 < 阈值？
    ├── 是 → 裁剪区域 → PaddleOCR → 结构化提取(药名/批号/规格/厂家/主治)
    │         → LLM校验纠错 → JSON保存 → 返回修正结果
    └── 否 → 直接返回YOLO结果

OCR提取的完整结构化信息（药名、批准文号、规格、厂家、功能主治）会：
  1. 实时显示在视频标注中（LLM纠错后的药名）
  2. 保存为 JSON 到 ocr_output/ 目录
  3. 存入内存缓存，供健康问答 LLM 引用
"""

import json
import logging
import os
import re
import ssl
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OCR_OUTPUT_DIR = PROJECT_ROOT / "dashboard" / "ocr_output"
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 模块级路径准备（无副作用） ──
_YOLO_DIR = str(PROJECT_ROOT / "training" / "yolo")
if _YOLO_DIR not in sys.path:
    sys.path.insert(0, _YOLO_DIR)

_ocr_module = None
_env_setup_done = False

def _setup_env():
    """延迟设置 PaddleOCR 环境变量 & SSL patch（仅在首次使用时执行）"""
    global _env_setup_done
    if _env_setup_done:
        return
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    _orig_load = ssl.SSLContext.load_default_certs
    if not getattr(_orig_load, "_fusion_patched", False):

        def _patched_ssl(self, purpose=ssl.Purpose.SERVER_AUTH):
            try:
                _orig_load(self, purpose)
            except ssl.SSLError:
                pass

        _patched_ssl._fusion_patched = True  # type: ignore[attr-defined]
        ssl.SSLContext.load_default_certs = _patched_ssl

    _env_setup_done = True


def _get_ocr_module():
    global _ocr_module
    if _ocr_module is None:
        _setup_env()
        from ocr import build_medicine_info, build_llm_input, parse_ocr_output, MedicineInfo
        _ocr_module = {
            "build_medicine_info": build_medicine_info,
            "build_llm_input": build_llm_input,
            "parse_ocr_output": parse_ocr_output,
            "MedicineInfo": MedicineInfo,
        }
    return _ocr_module

# ── 融合参数 ──
FUSION_CONF_THRESHOLD = 0.55
OCR_CACHE_TTL = 3.0
MAX_CONCURRENT_OCR = 3       # OCR线程上限
MAX_CACHE_SIZE = 50          # 缓存条目上限

# ── 全局状态 ──
_ocr_engine = None
_ocr_lock = threading.Lock()
_fusion_results = {}
_fusion_lock = threading.Lock()
_pending_jobs = set()
_ocr_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_OCR)  # 控制并发

# 最近的完整结构化检测结果（供问答LLM引用）
_recent_medicine_infos = []  # [(timestamp, medicine_info_dict), ...]
MAX_RECENT = 20


def _get_paddleocr():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        _setup_env()
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(lang="ch")
            logger.info("PaddleOCR ready")
        except Exception as e:
            logger.warning("PaddleOCR failed: %s", e)
            _ocr_engine = False
    return _ocr_engine


def _crop_hash(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return None
    small = cv2.resize(roi, (48, 48))
    return hash(small.tobytes())


def _run_ocr_full(frame, box):
    """对裁剪区域运行完整OCR+结构化提取管线

    Returns:
        dict: 结构化药品信息（来自 ocr.py 的 build_llm_input）
        list: 原始OCR文本列表
    """
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, []

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, []

    ocr = _get_paddleocr()
    if ocr is False or ocr is None:
        return None, []

    try:
        # 运行 OCR
        if hasattr(ocr, "predict"):
            output = ocr.predict(roi)
        else:
            output = ocr.ocr(roi)

        # 解析 OCR 输出
        mod = _get_ocr_module()
        ocr_items = mod["parse_ocr_output"](output)
        if not ocr_items:
            return None, []

        # 收集原始文本
        raw_texts = [item.text for item in ocr_items if item.text]

        # 结构化提取
        medicine_info = mod["build_medicine_info"](
            all_items=ocr_items,
            image_shape=roi.shape,
        )
        llm_input = mod["build_llm_input"](medicine_info)

        return llm_input, raw_texts

    except Exception as e:
        logger.warning("OCR full pipeline error: %s", e)
        return None, []


def _ask_llm_to_correct(yolo_name, yolo_conf, ocr_texts, structured_info):
    """调用本地LLM校验YOLO结果，结合OCR文字和结构化信息"""
    try:
        import json as _json
        import urllib.request
        from models.llm_client import _ollama_pick_model, OLLAMA_URL

        model = _ollama_pick_model()
        if not model:
            return None

        # 构建提示词（含结构化信息）
        struc_block = ""
        if structured_info:
            si = structured_info
            parts = []
            if si.get("medicine_name"): parts.append(f"OCR识别的药名: {si['medicine_name']}")
            if si.get("approval_number"): parts.append(f"批准文号: {si['approval_number']}")
            if si.get("specification"): parts.append(f"规格: {si['specification']}")
            if si.get("manufacturer"): parts.append(f"厂家: {si['manufacturer']}")
            if si.get("indication"): parts.append(f"功能主治: {si['indication']}")
            if parts:
                struc_block = "\n".join(parts)

        ocr_block = "\n".join(f"  - {t}" for t in (ocr_texts or [])[:15])
        prompt = (
            "你是药品识别专家。摄像头检测到一个药盒，请根据OCR文字判断这是什么药。\n\n"
            f"YOLO视觉检测: {yolo_name}（置信度 {yolo_conf:.0%}）\n\n"
            f"OCR提取的结构化信息:\n{struc_block or '(未提取到)'}\n\n"
            f"药盒上所有OCR文字:\n{ocr_block or '(未识别到文字)'}\n\n"
            "请回答:\n"
            "1. 这是什么药？（只给药名，如\"布洛芬缓释胶囊\"）\n"
            "2. 药品类别（感冒用药/止咳化痰/消炎药/止痛药/退烧药/肠胃药/维生素-保健/抗过敏药/外用药/慢性病药/中成药/其他）\n"
            "3. 可信度评分（0-100）\n\n"
            '严格按JSON格式回复:\n{"name":"药品名","category":"类别","confidence":85}'
        )

        req = urllib.request.Request(
            OLLAMA_URL,
            data=_json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = _json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "").strip()

        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            result = _json.loads(json_match.group())
            return {
                "name": result.get("name", yolo_name),
                "category": result.get("category", "其他"),
                "confidence": min(result.get("confidence", 80), 100) / 100.0,
                "source": "YOLO+OCR+LLM"
            }
    except Exception as e:
        logger.warning("LLM correct failed: %s", e)
    return None


# ═══════════════════════════════════════════════════════════════
#  对外接口
# ═══════════════════════════════════════════════════════════════

def request_fusion(frame, detection):
    """请求OCR+LLM融合纠错（后台异步）

    Args:
        frame: 完整帧
        detection: {"name", "confidence", "box", "category"}

    Returns:
        dict: 当前最佳结果
    """
    name = detection["name"]
    conf = detection["confidence"]
    box = tuple(detection["box"])

    # 高置信度→直接用YOLO
    if conf >= FUSION_CONF_THRESHOLD:
        return {**detection, "source": "YOLO"}

    # 检查缓存
    ch = _crop_hash(frame, detection["box"])
    cache_key = (box, ch)
    with _fusion_lock:
        if cache_key in _fusion_results:
            cached = _fusion_results[cache_key]
            if time.time() - cached.get("_time", 0) < OCR_CACHE_TTL:
                return {**detection, **cached, "source": cached.get("source", "YOLO+OCR")}

    # 避免重复提交
    if cache_key in _pending_jobs:
        return {**detection, "source": "YOLO"}
    _pending_jobs.add(cache_key)

    # ── 后台执行完整管线（信号量控制并发数） ──
    def _job():
        acquired = _ocr_semaphore.acquire(blocking=False)
        if not acquired:
            _pending_jobs.discard(cache_key)
            return  # 超过并发上限，丢弃本次请求
        try:
            structured_info, ocr_texts = _run_ocr_full(frame, detection["box"])

            # LLM 纠错
            llm_result = _ask_llm_to_correct(name, conf, ocr_texts, structured_info)

            result = {"_time": time.time(), "ocr_texts": (ocr_texts or [])[:10]}

            if llm_result:
                result["name"] = llm_result["name"]
                result["category"] = llm_result["category"]
                result["confidence"] = llm_result["confidence"]
                result["source"] = "YOLO+OCR+LLM"
            elif structured_info and structured_info.get("medicine_name"):
                # LLM不可用，用OCR结构化提取的药名
                result["name"] = structured_info["medicine_name"]
                result["category"] = structured_info.get("prescription_type", "其他")
                result["source"] = "YOLO+OCR"
            else:
                result["source"] = "YOLO+OCR"

            # ── 保存完整结构化结果到JSON ──
            if structured_info:
                result["structured"] = structured_info
                _save_detection_json(detection, structured_info, ocr_texts)

            # ── 存入近期记录（供问答引用） ──
            global _recent_medicine_infos
            record = {
                "time": time.strftime("%H:%M:%S"),
                "yolo_name": name,
                "yolo_conf": conf,
                "corrected_name": result.get("name", name),
                "source": result.get("source", "YOLO+OCR"),
                "structured": structured_info,
            }
            _recent_medicine_infos.append((time.time(), record))
            if len(_recent_medicine_infos) > MAX_RECENT:
                _recent_medicine_infos = _recent_medicine_infos[-MAX_RECENT:]

            with _fusion_lock:
                # 缓存大小限制
                if len(_fusion_results) >= MAX_CACHE_SIZE:
                    # 删除最旧的条目
                    oldest = min(_fusion_results.items(),
                                 key=lambda x: x[1].get("_time", 0))
                    del _fusion_results[oldest[0]]
                _fusion_results[cache_key] = result

        except Exception as e:
            logger.warning("job error: %s", e)
        finally:
            _pending_jobs.discard(cache_key)
            _ocr_semaphore.release()

    t = threading.Thread(target=_job, daemon=True)
    t.start()

    return {**detection, "source": "YOLO"}


def _save_detection_json(detection, structured_info, ocr_texts):
    """保存检测结果到JSON文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detection_{timestamp}_{detection['name']}.json"
    filepath = OCR_OUTPUT_DIR / filename

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "yolo_detection": {
            "name": detection["name"],
            "confidence": round(detection["confidence"], 4),
            "category": detection.get("category", ""),
        },
        "ocr_structured": structured_info,
        "ocr_texts": ocr_texts[:20] if ocr_texts else [],
        "safety_notice": "识别结果来自药盒图像OCR+LLM融合。"
                         "药品信息应由用户核对，不得仅依据本结果自行决定用药。"
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("JSON save error: %s", e)


def get_fusion_status():
    """返回融合模块状态"""
    return {
        "ocr_available": _get_paddleocr() is not False and _get_paddleocr() is not None,
        "threshold": FUSION_CONF_THRESHOLD,
        "cache_size": len(_fusion_results),
        "pending": len(_pending_jobs),
        "json_output_dir": str(OCR_OUTPUT_DIR),
        "recent_detections": len(_recent_medicine_infos),
    }


def get_recent_detections(limit=5):
    """获取最近的检测结果（供问答LLM引用）"""
    items = sorted(_recent_medicine_infos, key=lambda x: x[0], reverse=True)[:limit]
    return [item[1] for item in items]


def get_medicine_context_for_chat():
    """获取当前检测到的药品上下文（注入问答LLM的system prompt）"""
    items = get_recent_detections(3)
    if not items:
        return ""

    lines = ["【当前检测到的药品信息】"]
    for i, item in enumerate(items, 1):
        si = item.get("structured") or {}
        lines.append(f"{i}. {item['corrected_name']}")
        if si.get("specification"):
            lines.append(f"   规格: {si['specification']}")
        if si.get("indication"):
            lines.append(f"   主治: {si['indication']}")
        if si.get("manufacturer"):
            lines.append(f"   厂家: {si['manufacturer']}")
        if si.get("approval_number"):
            lines.append(f"   批号: {si['approval_number']}")
    return "\n".join(lines)

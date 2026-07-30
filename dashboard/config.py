"""应用配置文件"""

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def setup_logging():
    """配置全局日志格式和级别"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 抑制第三方库的 DEBUG 日志
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


class Config:
    """基础配置"""
    SECRET_KEY = os.environ.get("SECRET_KEY", "medpro-dashboard-secret-key")
    DATABASE = str(BASE_DIR / "database" / "medpro.db")

    # 摄像头配置
    CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

    # ── YOLO 药品检测配置 ──
    YOLO_WEIGHTS = os.environ.get(
        "YOLO_WEIGHTS",
        str(PROJECT_ROOT / "models" / "yolo" / "best.pt")
    )
    YOLO_CONF = float(os.environ.get("YOLO_CONF", 0.15))
    YOLO_IOU = float(os.environ.get("YOLO_IOU", 0.5))
    YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", 416))

    # ── 情绪识别配置 ──
    EMOTION_MODEL = os.environ.get(
        "EMOTION_MODEL",
        str(PROJECT_ROOT / "models" / "emotion" / "emotion_torch.pt")
    )

    # ── 本地大模型配置 ──
    # Qwen + LoRA (本地，优先)
    LLM_BASE_MODEL = os.environ.get(
        "LLM_BASE_MODEL",
        str(PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    )
    LLM_LORA_ADAPTER = os.environ.get(
        "LLM_LORA_ADAPTER",
        str(PROJECT_ROOT / "models" / "lora_adapter")
    )
    # RAG 知识库
    RAG_CHROMA_DIR = os.environ.get(
        "RAG_CHROMA_DIR",
        str(PROJECT_ROOT / "models" / "rag")
    )
    RAG_BGE_MODEL = os.environ.get(
        "RAG_BGE_MODEL",
        str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5")
    )
    # Ollama (降级方案)
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    # 检测入库间隔（秒）
    SAVE_INTERVAL = int(os.environ.get("SAVE_INTERVAL", 5))

    # Flask 配置
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", 5000))
    # 静态文件缓存（生产环境建议设为 3600 或更高）
    SEND_FILE_MAX_AGE_DEFAULT = int(os.environ.get("STATIC_CACHE_SECONDS", 3600))

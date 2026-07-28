"""药品检测模型封装 — YOLO11-OBB 真实模型 + 模拟降级"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── 项目根目录（medpro_dashboard 与 YOLO/output 平级） ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "best.pt"
YOLO_YAML = PROJECT_ROOT / "training" / "yolo" / "data" / "medicine" / "medicine_obb.yaml"

# 18 类药品名称（与 medicine_obb.yaml 一致，硬编码避免 yaml 依赖）
MEDICINE_CLASSES = [
    "感冒药", "止咳化痰", "消炎药", "止痛药", "退烧药", "肠胃药",
    "泻药通便", "维生素", "抗过敏药", "外用药", "创可贴绷带",
    "慢性病药", "心血管药", "中成药", "妇科用药", "肛肠用药", "保健品", "其他"
]

# 类别→大类映射（供饼图统计使用）
CATEGORY_MAP = {
    "感冒药": "感冒用药", "止咳化痰": "感冒用药", "退烧药": "感冒用药",
    "消炎药": "抗生素", "止痛药": "解热镇痛药",
    "肠胃药": "肠胃用药", "泻药通便": "肠胃用药",
    "维生素": "维生素/保健", "保健品": "维生素/保健",
    "抗过敏药": "抗过敏药",
    "外用药": "外用药", "创可贴绷带": "外用药",
    "慢性病药": "慢性病药", "心血管药": "慢性病药",
    "中成药": "中成药",
    "妇科用药": "妇科用药", "肛肠用药": "肛肠用药",
    "其他": "其他",
}


def _pick_device():
    """自动选择推理设备：优先 CUDA GPU，否则 CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            name = torch.cuda.get_device_name(0)
            logger.info("使用 GPU: %s", name)
            return 0
    except Exception:
        pass
    logger.info("使用 CPU")
    return "cpu"


class MedicineDetector:
    """药品目标检测器 — YOLO11-OBB

    自动加载训练好的 YOLO 模型；模型不存在时降级为模拟数据。
    """

    def __init__(self, model_path=None, conf=0.15, iou=0.5, imgsz=416):
        self.model_path = model_path or str(YOLO_WEIGHTS)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = None
        self.device = "cpu"
        self._loaded = False
        self._available = False

        self._try_load()

    def _try_load(self):
        """尝试加载 YOLO 模型"""
        if not os.path.isfile(self.model_path):
            logger.warning("模型文件不存在: %s", self.model_path)
            logger.warning("降级为模拟检测数据")
            return

        try:
            from ultralytics import YOLO

            self.device = _pick_device()
            logger.info("加载模型: %s  (imgsz=%s)", self.model_path, self.imgsz)
            self.model = YOLO(self.model_path)
            self._available = True
            self._loaded = True
            logger.info("模型加载成功，%d 类药品", len(MEDICINE_CLASSES))

        except ImportError:
            logger.warning("ultralytics 未安装，降级为模拟检测数据")
        except Exception as e:
            logger.warning("模型加载失败: %s", e)
            logger.warning("降级为模拟检测数据")

    def detect(self, frame):
        """对单帧图像进行药品检测

        Args:
            frame: BGR 格式的 numpy 图像数组

        Returns:
            list[dict]: 检测结果列表 [{name, confidence, box, category}, ...]
        """
        if self._available and self.model is not None:
            return self._detect_real(frame)
        return self._detect_mock(frame)

    def _detect_real(self, frame):
        """使用真实 YOLO 模型检测"""
        try:
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                logger.warning("OOM，降级为 CPU 推理")
                self.device = "cpu"
                self.model.to("cpu")
                try:
                    results = self.model.predict(
                        frame, conf=self.conf, iou=self.iou,
                        imgsz=320, device="cpu", verbose=False,
                    )
                except Exception:
                    return self._detect_mock(frame)
            else:
                return self._detect_mock(frame)

        r = results[0]
        detections = []

        # OBB 模型输出在 r.obb，标准模型输出在 r.boxes
        boxes = None
        if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
            boxes = r.obb
        elif r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = MEDICINE_CLASSES[cls_id] if cls_id < len(MEDICINE_CLASSES) else "未知"

                detections.append({
                    "name": name,
                    "confidence": round(conf, 4),
                    "box": [x1, y1, x2, y2],
                    "category": CATEGORY_MAP.get(name, "其他"),
                })

        return detections

    def _detect_mock(self, frame):
        """模拟检测数据（降级方案）"""
        h, w = frame.shape[:2]
        return [
            {
                "name": "布洛芬",
                "confidence": 0.92,
                "box": [int(w * 0.25), int(h * 0.25), int(w * 0.45), int(h * 0.75)],
                "category": "解热镇痛药"
            },
            {
                "name": "阿莫西林",
                "confidence": 0.89,
                "box": [int(w * 0.52), int(h * 0.28), int(w * 0.72), int(h * 0.75)],
                "category": "抗生素"
            },
        ]

    @property
    def is_real_model(self):
        """是否使用真实模型"""
        return self._available

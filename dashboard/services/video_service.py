"""视频流服务 — 异步推理版（后台线程检测 + 主线程只负责推流）"""

import logging
import os
import time
import threading
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

from models.detector import MedicineDetector
from models.emotion import EmotionRecognizer
from models.fusion import request_fusion, FUSION_CONF_THRESHOLD

# ── 全局模型实例 ──
detector = MedicineDetector()
emotion_model = EmotionRecognizer()

_DETECTOR_TAG = "YOLO-OBB" if detector.is_real_model else "MOCK"
_EMOTION_TAG = "CNN" if emotion_model.is_real_model else "MOCK"

# ── 检测模式 ──
_detect_mode = "medicine"  # medicine / emotion / both（与前端默认按钮一致）

# ── 入库参数 ──
_SAVE_INTERVAL = 5  # 秒（与 config.py 的 SAVE_INTERVAL 一致）
_last_save_time = 0


def get_detect_mode():
    return _detect_mode


def set_detect_mode(mode):
    global _detect_mode
    _detect_mode = mode
    logger.info("检测模式切换为: %s", mode)

# ── 性能参数 ──
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 78
YOLO_IMGSZ = 416         # YOLO 推理尺寸（后台异步运行，不影响帧率）

# ── 中文字体 ──
_FONT_PATH = None
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
]


def _get_font(size=20):
    global _FONT_PATH
    if _FONT_PATH is None:
        for path in _FONT_CANDIDATES:
            if os.path.isfile(path):
                _FONT_PATH = path
                break
        if _FONT_PATH is None:
            _FONT_PATH = ""
    try:
        return ImageFont.truetype(_FONT_PATH, size) if _FONT_PATH else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _draw_all_cn_labels(frame, labels):
    """批量绘制中文标注（一次 PIL 转换）"""
    if not labels:
        return
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for text, x, y, font_size, color, bg_color in labels:
        font = _get_font(font_size)
        pil_color = (color[2], color[1], color[0])
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bg_color is not None:
            pil_bg = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle([x, y, x + tw + 6, y + th + 4], fill=pil_bg)
        draw.text((x + 3, y + 1), text, font=font, fill=pil_color)

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_cn_text(img, text, pos, font_size=20, color=(255, 255, 255), bg_color=None):
    _draw_all_cn_labels(img, [(text, pos[0], pos[1], font_size, color, bg_color)])


# ═══════════════════════════════════════════════════════════════
#  异步推理线程
# ═══════════════════════════════════════════════════════════════

class AsyncDetector:
    """后台线程异步推理：不阻塞主视频流循环"""

    def __init__(self, detector_obj, emotion_obj):
        self.detector = detector_obj
        self.emotion = emotion_obj
        self.lock = threading.Lock()

        # 共享结果
        self.latest_frame = None
        self.detections = []
        self.emotion_result = {"emotion": "检测中...", "confidence": 0.0, "face_box": None}

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """后台循环：取最新帧 → 推理 → OCR+LLM融合(低置信度) → 写入结果"""
        while self.running:
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None

            if frame is not None:
                mode = get_detect_mode()

                # ── YOLO 检测（medicine / both 模式） ──
                if mode in ("medicine", "both"):
                    # 不预缩放，让 YOLO 内部 letterbox 保持宽高比
                    try:
                        raw_detections = self.detector.detect(frame)
                    except Exception:
                        raw_detections = []

                    fused = []
                    for d in raw_detections:
                        if d["confidence"] < FUSION_CONF_THRESHOLD:
                            enhanced = request_fusion(frame, d)
                            fused.append(enhanced)
                        else:
                            d["source"] = "YOLO"
                            fused.append(d)
                    self.detections = fused
                else:
                    self.detections = []

                # ── 情绪识别（emotion / both 模式） ──
                if mode in ("emotion", "both"):
                    try:
                        self.emotion_result = self.emotion.predict(frame)
                    except Exception:
                        pass
                else:
                    self.emotion_result = {"emotion": "已关闭", "confidence": 0.0, "face_box": None}
            else:
                time.sleep(0.01)

    def push_frame(self, frame):
        """主线程放入新帧（非阻塞，如果上一帧还没处理完就跳过）"""
        with self.lock:
            if self.latest_frame is None:  # 上一帧已消费，放入新帧
                self.latest_frame = frame.copy()

    def get_results(self):
        """主线程获取最新推理结果（非阻塞）"""
        with self.lock:
            return list(self.detections), dict(self.emotion_result)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
#  视频流主循环
# ═══════════════════════════════════════════════════════════════

def generate_video_stream():
    """生成 MJPEG 视频流 — 异步推理版

    主线程：读摄像头 → 请求推理（非阻塞）→ 叠加标注 → 编码输出
    后台线程：取帧 → YOLO 检测 + 情绪识别 → 写入结果
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield from _generate_placeholder_stream()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 启动后台推理线程
    async_det = AsyncDetector(detector, emotion_model)

    prev_time = time.time()
    detections = []
    emotion_result = {"emotion": "检测中...", "confidence": 0.0, "face_box": None}

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # ── 请求异步推理（非阻塞：放入帧即返回） ──
            async_det.push_frame(frame)

            # ── 读取最新推理结果（非阻塞） ──
            detections, emotion_result = async_det.get_results()

            # ── FPS ──
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            # ── 定期入库 ──
            _save_detections_to_db(detections, emotion_result)

            # ── 叠加标注 ──
            cn_labels = []

            for item in detections:
                x1, y1, x2, y2 = item["box"]
                name = item["name"]
                conf = item["confidence"]
                source = item.get("source", "YOLO")

                # 框颜色：LLM纠错后用绿色，纯YOLO用橙蓝
                if "LLM" in source:
                    box_color = (0, 200, 100)   # 绿色 = 已纠错
                elif "OCR" in source:
                    box_color = (0, 180, 220)   # 黄色 = OCR辅助
                else:
                    box_color = (0, 80, 255)    # 橙蓝 = YOLO

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # 标签：药名 + 置信度 + 来源标记
                src_tag = {"YOLO": "", "YOLO+OCR": " [OCR]", "YOLO+OCR+LLM": " [AI]"}.get(source, "")
                label = f"{name} {conf:.2f}{src_tag}"
                cn_labels.append((label, x1, max(y1 - 26, 0), 16,
                                  (255, 255, 255), box_color))

            face_box = emotion_result.get("face_box")
            if face_box:
                fx1, fy1, fx2, fy2 = face_box
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)

            emotion = emotion_result["emotion"]
            emotion_conf = emotion_result["confidence"]
            cn_labels.append((f"情绪: {emotion}  ({emotion_conf:.2f})",
                              20, frame.shape[0] - 28, 18,
                              (0, 255, 255), (0, 0, 0)))

            # 批量中文
            _draw_all_cn_labels(frame, cn_labels)

            # 英文 HUD
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{_DETECTOR_TAG} | {_EMOTION_TAG}",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # ── 编码 ──
            ret, buf = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() +
                   b"\r\n")

    except GeneratorExit:
        pass
    except Exception as e:
        logger.error("Stream error: %s", e)
    finally:
        async_det.stop()
        cap.release()


def _save_detections_to_db(detections, emotion_result, app=None):
    """每隔 SAVE_INTERVAL 秒将检测结果写入 SQLite"""
    global _last_save_time
    now = time.time()
    if now - _last_save_time < _SAVE_INTERVAL:
        return
    _last_save_time = now

    try:
        if app is None:
            from flask import current_app
            app = current_app

        from database.db import get_db
        db = get_db(app)

        if detections:
            # 写入检测记录（每个检测结果一条）
            for d in detections:
                db.execute(
                    "INSERT INTO detection_record (medicine_name, confidence, emotion, emotion_confidence) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        d.get("name", "未知"),
                        d.get("confidence", 0),
                        emotion_result.get("emotion", ""),
                        emotion_result.get("confidence", 0),
                    )
                )

        # 如果没有任何检测，也记录一条（用于统计活跃时间）
        if not detections:
            db.execute(
                "INSERT INTO detection_record (medicine_name, confidence, emotion, emotion_confidence) "
                "VALUES (?, ?, ?, ?)",
                ("", 0, emotion_result.get("emotion", ""), emotion_result.get("confidence", 0))
            )

        db.commit()
    except Exception as e:
        logger.warning("检测记录入库失败: %s", e)


def _generate_placeholder_stream():
    while True:
        frame = np.ones((360, 480, 3), dtype=np.uint8) * 50
        put_cn_text(frame, "摄像头未连接", (120, 140), font_size=26, color=(200, 200, 200))
        put_cn_text(frame, "请检查摄像头设备", (130, 190), font_size=16, color=(150, 150, 150))
        ret, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() +
               b"\r\n")
        time.sleep(0.1)

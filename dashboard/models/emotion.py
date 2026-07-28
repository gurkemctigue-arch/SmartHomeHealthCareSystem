"""情绪识别模型封装 — ResNet-18 + OpenCV 人脸检测 + 模拟降级"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch 可选导入（无 GPU 时也能跑）
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None
    TORCH_AVAILABLE = False

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMOTION_MODEL_PATH = PROJECT_ROOT / "models" / "emotion" / "emotion_torch.pt"

# 7 类情绪（与模型训练时一致）
EMOTION_CLASSES = ["愤怒", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "平静"]

# 情绪→中文映射
EMOTION_CN = {
    "angry": "愤怒", "disgust": "厌恶", "fear": "恐惧",
    "happy": "开心", "sad": "悲伤", "surprise": "惊讶", "neutral": "平静",
}


# ═══════════════════════════════════════════════════════════════
#  ResNet-18 架构（与 emotion_torch.pt 权重一致）
# ═══════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class BasicBlock(nn.Module):
        """ResNet BasicBlock — 两个 3×3 卷积 + 残差连接"""
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.downsample = None
            if stride != 1 or in_planes != planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)


    class ResNet18Emotion(nn.Module):
        """ResNet-18 情绪识别（灰度单通道输入，7 分类）

        与 emotion_torch.pt 权重结构精确匹配：
          conv1 → bn1 → relu → maxpool → layer1~4 → avgpool → fc
        """

        def __init__(self, num_classes=7):
            super().__init__()
            self.in_planes = 64

            # 输入：1×48×48 灰度图
            self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)  # 48→24→12

            self.layer1 = self._make_layer(64, 2, stride=1)   # 12×12
            self.layer2 = self._make_layer(128, 2, stride=2)  # 6×6
            self.layer3 = self._make_layer(256, 2, stride=2)  # 3×3
            self.layer4 = self._make_layer(512, 2, stride=2)  # 2×2

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        def _make_layer(self, planes, num_blocks, stride):
            layers = [BasicBlock(self.in_planes, planes, stride)]
            self.in_planes = planes
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(planes, planes, 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

else:
    BasicBlock = None  # type: ignore
    ResNet18Emotion = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
#  EmotionRecognizer
# ═══════════════════════════════════════════════════════════════

class EmotionRecognizer:
    """用户情绪识别器 — ResNet-18 + Haar Cascade 人脸检测

    自动加载训练好的模型；模型不存在时降级为模拟数据。
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or str(EMOTION_MODEL_PATH)
        self.model = None
        self.device = "cpu"
        self.classes = EMOTION_CLASSES
        self._available = False
        self.face_cascade = None

        self._try_load()

    def _try_load(self):
        """尝试加载情绪识别模型和人脸检测器"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch 未安装，降级为模拟情绪数据")
            return

        # 加载人脸检测器
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("人脸检测器加载失败，将使用全图识别")
                self.face_cascade = None
        except Exception:
            pass

        # 加载情绪模型
        if not os.path.isfile(self.model_path):
            logger.warning("模型文件不存在: %s", self.model_path)
            logger.warning("降级为模拟情绪数据")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.classes = checkpoint.get("classes", EMOTION_CLASSES)

            self.model = ResNet18Emotion(num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.model.to(self.device)
            self.model.eval()
            self._available = True
            logger.info("模型加载成功，%d 类情绪  device=%s", len(self.classes), self.device)

        except Exception as e:
            logger.warning("模型加载失败: %s", e)
            logger.warning("降级为模拟情绪数据")

    def predict(self, frame):
        """对单帧图像进行情绪识别

        Args:
            frame: BGR 格式的 numpy 图像数组

        Returns:
            dict: {"emotion": str, "confidence": float, "face_box": list | None}
        """
        if self._available and self.model is not None:
            return self._predict_real(frame)
        return self._predict_mock(frame)

    def _preprocess_face(self, face_roi):
        """将人脸 ROI 预处理为模型输入格式"""
        import cv2

        # 灰度 + 缩放至 48x48
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        # 归一化到 [-1, 1]
        img = gray.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=(0, 1))  # (1, 1, 48, 48)

        return img

    def _predict_real(self, frame):
        """使用真实模型进行情绪识别"""
        import cv2

        face_box = None

        # 人脸检测
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

            if len(faces) > 0:
                # 取最大人脸
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_box = [int(x), int(y), int(x + w), int(y + h)]
                face_roi = frame[y:y + h, x:x + w]
            else:
                # 无人脸时使用画面中央区域
                h, w = frame.shape[:2]
                face_roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        else:
            h, w = frame.shape[:2]
            face_roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

        try:
            img = self._preprocess_face(face_roi)
            img_tensor = torch.from_numpy(img).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

            emotion = self.classes[pred.item()]
            confidence = round(conf.item(), 4)

        except Exception:
            emotion = "平静"
            confidence = 0.5

        return {
            "emotion": emotion,
            "confidence": confidence,
            "face_box": face_box
        }

    def _predict_mock(self, frame):
        """模拟情绪数据（降级方案）"""
        return {
            "emotion": "开心",
            "confidence": 0.91,
            "face_box": None
        }

    @property
    def is_real_model(self):
        """是否使用真实模型"""
        return self._available

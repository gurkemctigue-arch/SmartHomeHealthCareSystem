"""OpenCV 读写图 + YOLO 中文可视化：兼容 Windows 中文路径。

cv2.imread / imwrite 在含中文路径时会静默失败；本模块用 fromfile/imdecode、
imencode/tofile 绕过。plot_yolo_cn 解决 ultralytics 默认字体不支持中文的问题。

供 ch07/05_yolo_camera.py 等调用。
"""
import os  # 导入 os：字体路径、建父目录

import cv2  # 导入 OpenCV：编解码、色彩转换
import numpy as np  # 导入 NumPy：fromfile 读字节、数组互转
from PIL import Image, ImageDraw, ImageFont  # 导入 PIL：中文文字绘制

# Windows 常见中文字体候选（按优先级）
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
)


def _load_font(size):
    """按候选列表加载 TrueType 字体；都失败则用默认字体。"""
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)  # size：字号（像素）
    return ImageFont.load_default()


def imread_color(path):
    """读彩色 BGR 图；失败时抛出 FileNotFoundError。"""
    buf = np.fromfile(path, dtype=np.uint8)  # 按字节读文件（支持中文路径）
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # flags：解码为 BGR 三通道
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def imwrite_color(path, img):
    """写 BGR 图像；失败时抛出 RuntimeError。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"  # 编码格式由后缀决定
    ok, buf = cv2.imencode(ext, img)  # 编码到内存缓冲区
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(path)  # 再写盘 → 支持中文路径


def put_chinese(img_bgr, text, pos, size=22, color=(0, 255, 0)):
    """在 BGR 图像上绘制中文（cv2.putText 不支持中文），返回新 BGR 图。

    Args:
        img_bgr: 输入图（BGR）
        text: 要写的中文/字符串
        pos: 左上角 (x, y)
        size: 字号
        color: BGR 颜色，默认绿
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])  # BGR → RGB（PIL 用 RGB）
    draw.text(
        pos,  # xy：文字左上角
        text,  # text
        font=_load_font(size),  # font
        fill=rgb,  # fill：RGB 颜色
    )
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def plot_yolo_cn(result, show_conf=True, font_size=18):
    """YOLO Results 可视化（中文类别名）。

    支持普通检测 boxes，也支持 OBB（result.obb）。
    ultralytics 的 plot() 缺中文字体时会乱码；此处只画框再用系统字体写标签。
    """
    annotated = result.plot(labels=False)
    names = result.names

    # ---------- OBB：旋转框 ----------
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = _load_font(font_size)
        for i in range(len(obb)):
            # xyxyxyxy: 四点；用于文字锚点取最上边中心附近
            try:
                pts = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            except Exception:
                # 回退：用外接水平框左上角
                xyxy = obb.xyxy[i].cpu().tolist()
                pts = np.array(
                    [[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]]
                )
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            cls_id = int(obb.cls[i])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if show_conf:
                label = f"{label} {float(obb.conf[i]):.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ty = max(y1 - th - 4, 0)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=(0, 180, 0))
            draw.text((x1 + 2, ty + 1), label, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ---------- 水平框 detect ----------
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)

    for box in boxes:
        x1, y1 = (int(v) for v in box.xyxy[0][:2].tolist())
        cls_id = int(box.cls[0])
        label = names[cls_id]
        if show_conf:
            label = f"{label} {float(box.conf[0]):.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(y1 - th - 4, 0)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=(0, 180, 0))
        draw.text((x1 + 2, ty + 1), label, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

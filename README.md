# 多模态智能医疗家庭助手

基于 YOLO-OBB + RapidOCR + SQLite 药品库的实时药盒识别与家庭健康工作台。

## 快速启动

```bash
# 安装 Web 与识别依赖
pip install -r dashboard/requirements.txt

# 从项目根目录启动
cd dashboard
python app.py
# 浏览器打开 http://127.0.0.1:5000
```

## 项目结构

```
Project/
├── Makefile                          # make run / install / clean
├── 多模态智能医疗家庭助手_前端大屏设计文档.md
│
├── dashboard/                 # Flask 大屏应用
│   ├── app.py                        # Flask 主程序与 API 路由
│   ├── config.py                     # 全局配置
│   ├── requirements.txt
│   ├── models/                       # 模型封装层
│   │   ├── medicine_recognizer.py    # YOLO 药盒定位 + OCR + 药名库匹配
│   │   ├── detector.py               # 旧检测器（兼容保留）
│   │   ├── emotion.py                # ResNet-18 情绪识别 (7类)
│   │   ├── llm_client.py             # 健康问答 (Qwen+LoRA > Ollama)
│   │   └── fusion.py                 # 旧融合链路（兼容保留）
│   ├── services/                     # 业务逻辑层
│   │   ├── video_service.py          # 异步视频流 + 推理 + 入库
│   │   ├── stats_service.py          # 概览/图表数据 (DB实时查询)
│   │   ├── alert_service.py          # 告警查询 + 规则引擎
│   │   └── medicine_service.py       # 药品管理 (CRUD)
│   ├── database/                     # SQLite 数据库 (含索引)
│   ├── templates/index.html          # 大屏页面
│   ├── static/css/dashboard.css      # 样式
│   ├── static/js/dashboard.js        # ECharts + 轮询
│   └── ocr_output/                   # OCR 检测 JSON 输出
│
├── yolo-ocr/                         # 已嵌入的药盒识别、训练与数据资产
│   ├── models/best_obb.pt            # Web 实际使用的 YOLO11-OBB 权重
│   ├── db/medicine.db                # 3130 种药品与 3619 个药名/别名
│   ├── day4/                         # 训练、数据准备、OCR 和药品库工具
│   ├── data/                         # OBB 训练/验证/测试数据集
│   ├── runs/                         # 训练指标与权重归档
│   └── output/                       # OCR、评估与数据处理输出
│
├── models/                           # 其他模型权重与向量库（可选）
│   ├── emotion/emotion_torch.pt      # 情绪识别 ResNet-18 (1.4MB, 7类)
│   ├── lora/                         # LLM 医疗微调 LoRA 适配器 (4.3MB)
│   ├── rag/                          # 医疗百科向量库 ChromaDB (9.5MB, 500条)
│   ├── Qwen2.5-0.5B-Instruct/        # LLM 基座 (988MB)
│   └── bge-small-zh-v1.5/            # RAG 中文嵌入模型
│
└── training/yolo/                    # 早期训练与摄像头脚本（兼容保留）
    ├── ocr.py                        # OCR 结构化提取管线
    ├── train.py                      # YOLO OBB 训练脚本
    ├── val.py / test.py              # 验证/测试
    └── camera.py                     # 独立摄像头检测 demo
```

## 技术架构

```
摄像头 → YOLO11-OBB 药盒定位 → 透视矫正 → RapidOCR 文字提取
                                              ↓
                              SQLite 药名/商品名/OCR别名匹配
                                              ↓
                         稳定框 + 药名/类别/功效/综合置信度

情绪识别: 人脸检测 → ResNet-18 情绪分类 → 视频叠加

健康问答: 用户问题 → RAG 医疗知识检索 → Qwen+LoRA (优先) / Ollama (降级)
```

## 数据流

```
检测 → detection_record 表 (SQLite, 每5秒入库)
告警 → alert_record 表
问答 → chat_record 表 (持久化)
OCR 与训练输出 → yolo-ocr/output/
```

## 依赖

- Python 3.9+ (cv conda 环境)
- Flask, OpenCV, NumPy, Pillow
- PyTorch + Ultralytics (YOLO)
- RapidOCR + ONNX Runtime (OCR)
- SQLite + PyYAML（药品、类别、功效与 OCR 别名）
- Transformers + PEFT (本地 LLM)
- ChromaDB + Sentence-Transformers (RAG)
- Ollama (LLM 降级方案)

## 模型路径配置

所有外部路径通过 `config.py` 中的环境变量配置，默认值自动推导。也可通过环境变量覆盖：

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `YOLO_WEIGHTS` | `yolo-ocr/models/best_obb.pt` | 药盒 OBB 定位权重 |
| `MEDICINE_DATABASE` | `yolo-ocr/db/medicine.db` | 药名、类别、功效与别名库 |
| `MEDICINE_CATALOG` | `yolo-ocr/day4/medicine_catalog.yaml` | 基础药品目录 |
| `OCR_INTERVAL` | `8` | 未确认药品的 OCR 检查间隔 |
| `OCR_KNOWN_INTERVAL` | `90` | 已确认药品的安全复核间隔 |
| `EMOTION_MODEL` | `models/emotion/emotion_torch.pt` | 情绪识别模型 |
| `LLM_BASE_MODEL` | `models/Qwen2.5-0.5B-Instruct` | LLM 基座 |
| `LLM_LORA_ADAPTER` | `models/lora` | LoRA 适配器 |
| `RAG_CHROMA_DIR` | `models/rag` | 知识库 |
| `RAG_BGE_MODEL` | `models/bge-small-zh-v1.5` | 嵌入模型 |

## 清理说明

本项目已清理教学实验产物（MNIST/EMNIST/Contest 等），训练代码归档至 `training/`。
详细说明见设计文档。

# 多模态智能医疗家庭助手

基于 YOLO + OCR + LLM 融合的实时药品检测与健康科普大屏。

## 快速启动

```bash
# 前置条件：cv conda 环境，Ollama 运行中
make run
# 浏览器打开 http://127.0.0.1:5000
```

## 项目结构

```
Project/
├── Makefile                          # make run / install / clean
├── 多模态智能医疗家庭助手_前端大屏设计文档.md
│
├── dashboard/                 # Flask 大屏应用
│   ├── app.py                        # 主程序，9 条路由
│   ├── config.py                     # 全局配置
│   ├── requirements.txt
│   ├── models/                       # 模型封装层
│   │   ├── detector.py               # YOLO11-OBB 药品检测 (18类)
│   │   ├── emotion.py                # ResNet-18 情绪识别 (7类)
│   │   ├── llm_client.py             # 健康问答 (Qwen+LoRA > Ollama)
│   │   └── fusion.py                 # YOLO+OCR+LLM 融合纠错
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
├── models/                           # 模型权重 & 向量库 (需自行下载)
│   ├── yolo/best.pt                  # YOLO 药品检测权重 (19MB, 18类)
│   ├── emotion/emotion_torch.pt      # 情绪识别 ResNet-18 (1.4MB, 7类)
│   ├── lora/                         # LLM 医疗微调 LoRA 适配器 (4.3MB)
│   ├── rag/                          # 医疗百科向量库 ChromaDB (9.5MB, 500条)
│   ├── Qwen2.5-0.5B-Instruct/        # LLM 基座 (988MB)
│   └── bge-small-zh-v1.5/            # RAG 中文嵌入模型
│
└── training/                         # 训练代码（归档，非运行时依赖）
    ├── yolo/
    │   ├── ocr.py                    # PaddleOCR 结构化提取管线
    │   ├── train.py                  # YOLO OBB 训练脚本
    │   ├── val.py / test.py          # 验证/测试
    │   ├── camera.py                 # 独立摄像头检测 demo
    │   └── data/medicine/            # 训练数据集
    └── nlp/
        ├── lora/                     # LoRA 微调脚本
        ├── rag/                      # RAG 检索构建脚本
        ├── ollama/                   # Ollama 客户端入门
        └── nlp_basics/               # NLP 基础实验
```

## 技术架构

```
摄像头 → YOLO 药品检测 → 低置信度? → PaddleOCR 文字提取 → LLM 纠错
                      ↓ 高置信度
                 视频标注输出 (药名 + 边框)

情绪识别: 人脸检测 → ResNet-18 情绪分类 → 视频叠加

健康问答: 用户问题 → RAG 医疗知识检索 → Qwen+LoRA (优先) / Ollama (降级)
```

## 数据流

```
检测 → detection_record 表 (SQLite, 每5秒入库)
告警 → alert_record 表
问答 → chat_record 表 (持久化)
OCR 结构化输出 → ocr_output/*.json
```

## 依赖

- Python 3.9+ (cv conda 环境)
- Flask, OpenCV, NumPy, Pillow
- PyTorch + Ultralytics (YOLO)
- PaddleOCR (OCR)
- Transformers + PEFT (本地 LLM)
- ChromaDB + Sentence-Transformers (RAG)
- Ollama (LLM 降级方案)

## 模型路径配置

所有外部路径通过 `config.py` 中的环境变量配置，默认值自动推导。也可通过环境变量覆盖：

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `YOLO_WEIGHTS` | `models/yolo/best.pt` | 药品检测权重 |
| `EMOTION_MODEL` | `models/emotion/emotion_torch.pt` | 情绪识别模型 |
| `LLM_BASE_MODEL` | `models/Qwen2.5-0.5B-Instruct` | LLM 基座 |
| `LLM_LORA_ADAPTER` | `models/lora` | LoRA 适配器 |
| `RAG_CHROMA_DIR` | `models/rag` | 知识库 |
| `RAG_BGE_MODEL` | `models/bge-small-zh-v1.5` | 嵌入模型 |

## 清理说明

本项目已清理教学实验产物（MNIST/EMNIST/Contest 等），训练代码归档至 `training/`。
详细说明见设计文档。

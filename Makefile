.PHONY: run install clean tree

# 多模态智能医疗家庭助手·成果大屏

# 启动 Flask 开发服务器
run:
	cd dashboard && python app.py

# 安装依赖（在 cv conda 环境中）
install:
	cd dashboard && pip install -r requirements.txt

# 清理 pycache 和数据库
clean:
	python -c "import shutil, os; [shutil.rmtree(os.path.join(r,d)) for r,ds,f in os.walk('.') for d in ds if d=='__pycache__']"
	python -c "import os; p='dashboard/database/medpro.db'; os.remove(p) if os.path.exists(p) else None"
	@echo Done.

# 显示项目结构
tree:
	@echo "Project/"
	@echo "├── Makefile"
	@echo "├── README.md"
	@echo "├── 多模态智能医疗家庭助手_前端大屏设计文档.md"
	@echo "│"
	@echo "├── dashboard/          Flask 大屏应用"
	@echo "│   ├── app.py                 主程序"
	@echo "│   ├── config.py              全局配置"
	@echo "│   ├── models/                模型封装 (detector/emotion/llm/fusion)"
	@echo "│   ├── services/              业务逻辑 (video/stats/alert/medicine)"
	@echo "│   ├── database/              SQLite 数据库"
	@echo "│   ├── templates/             HTML 页面"
	@echo "│   ├── static/                CSS / JS"
	@echo "│   └── ocr_output/            OCR 检测 JSON"
	@echo "│"
	@echo "├── models/                    所有模型权重"
	@echo "│   ├── yolo/                  YOLO 药品检测 (19MB)"
	@echo "│   ├── emotion/               表情识别 CNN (1.4MB)"
	@echo "│   ├── lora/                  LLM 医疗 LoRA (4.3MB)"
	@echo "│   ├── rag/                   医疗知识库 ChromaDB (9.5MB)"
	@echo "│   ├── Qwen2.5-0.5B-Instruct/ LLM 基座 (988MB)"
	@echo "│   └── bge-small-zh-v1.5/     RAG 嵌入模型"
	@echo "│"
	@echo "└── training/                  训练代码 (归档)"
	@echo "    ├── yolo/                  ocr / train / val / test / data"
	@echo "    └── nlp/                   (已删除，运行时不需要)"

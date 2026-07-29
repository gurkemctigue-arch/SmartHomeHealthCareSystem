# YOLO 药盒检测 + OCR 药名识别

该目录是当前药盒项目的独立归档。原文件仍保留在原位置，本目录中的默认路径已改为优先使用归档内的数据、模型和数据库。

## 目录

- `day4/`：YOLO OBB 训练、数据增强、摄像头检测、RapidOCR 和 SQLite 脚本。
- `models/best_obb.pt`：当前最佳模型，来自 `medicine_stage3_copypaste`。
- `models/yolo11s-obb.pt`：继续训练使用的 YOLO11s-OBB 预训练权重。
- `db/medicine.db`：药名、别名与 18 类类别数据库。
- `data/yolo/medicine/`：原始药盒 OBB 数据集。
- `data/training/medicine_copypaste/`：当前最终训练集，包含纠错、困难负样本和 Copy-Paste 增强结果。
- `runs/obb/`：各阶段的参数与指标；最终阶段另保留 `best.pt` 和 `last.pt`。
- `output/`：OCR 审核结果、数据库扩充结果和训练验证指标。
- `docs/类别(2).pdf`：18 类药品分类依据。

## 环境

```powershell
conda activate ai-vision-nmgdx
cd C:\Users\王鑫\OneDrive\桌面\ai-vision\软通动力实训-2026-07\软通动力实训-2026-07\ai-training\yolo-ocr
```

当前归档环境为 Python 3.10.20，主要依赖版本见 `requirements.txt`。

## 检查与摄像头识别

```powershell
python day4/10_yolo_camera_obb.py --check
python day4/10_yolo_camera_obb.py
```

测试单张图片：

```powershell
python day4/10_yolo_camera_obb.py --image data/yolo/medicine/images/train/images/1.jpg --output output/test_result.jpg
```

## 继续训练

先做数据预检，不启动训练：

```powershell
python day4/08_yolo11_train_obb.py --dry-run
```

以当前最佳模型为起点继续训练：

```powershell
python day4/08_yolo11_train_obb.py --data data/training/medicine_copypaste/medicine_copypaste.yaml --weights models/best_obb.pt --artifact-root . --name medicine_next --epochs 80 --batch 12 --device 0
```

训练脚本会在 `runs/obb/medicine_next/` 保存运行记录，并将选出的最佳权重写入 `models/best_obb.pt`。正式训练前建议先备份现有最佳权重，或把 `--artifact-root` 指向新的产物目录。

## 数据迭代

构建基础纠错数据集：

```powershell
python day4/11_prepare_medicine_map50_dataset.py
```

从不含药盒的图片目录挖掘困难负样本：

```powershell
python day4/12_prepare_hard_negative_dataset.py --candidate-dir D:\你的负样本图片目录
```

在困难负样本数据集上生成下一版 OBB Copy-Paste 数据：

```powershell
python day4/13_prepare_obb_copypaste_dataset.py
```

## 归档说明

未复制早期运行中数 GB 的 `epoch*.pt` 重复检查点、下载缓存和无关模型。它们不影响当前模型调用或下一轮训练；各阶段的 `args.yaml`、`results.csv`、最终权重和验证结果均已保留。

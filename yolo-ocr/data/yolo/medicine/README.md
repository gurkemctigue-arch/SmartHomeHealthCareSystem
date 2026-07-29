# medicine 药盒数据（Labelme 自标注）

## 发给学生：`待标记/`

12 组平均分图（**复制**自 `images/`，原图保留）：

```text
待标记/
├── train/          ← 第1组 … 第9组（来自 images/train）
│   ├── 第1组/
│   └── …
└── val/            ← 第10组 … 第12组（来自 images/valid）
    ├── 第10组/
    └── …
```

学生用 Labelme 打开**本组文件夹**标注即可。类名与标注规则见同目录 **[类别.md](./待标记/类别.md)**（18 类功效大类）。

简要：使用 **Create Rectangle** 框住**整个药盒**；类名须与 `类别.md` / `ch07/yolo_dataset.py` 一致。

当前各组约 **302** 张（已按数量拉平；文件名为连续数字 `1.jpg`…）。

## 训练用目录：`images/`

与 Roboflow 风格同构，供 `ch07/01` → `02` →（改 YAML 后）`03` 使用。

```text
images/
├── train/images/   ← 训练图（数字命名）
├── train/labels/   ← 同名 YOLO txt（回收各组 json 后由 01 生成）
├── valid/images/
├── valid/labels/
├── test/images/
└── test/labels/
```

## 标注回收后转换

1. 把各组标注好的 `*.json`（及对应图若需）收回，放回 `images/train/images` 或 `images/valid/images`（与图片同名）。
2. 在 ai-training 根目录：

```bash
python ch07/01_labelme2yolo.py
python ch07/02_split_dataset_yaml.py
```

`02` **不再重新划分** train/valid，只检查配对并生成 `medicine.yaml`。

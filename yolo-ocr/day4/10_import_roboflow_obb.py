# -*- coding: utf-8 -*-
"""受控导入 Roboflow 数据并合并为药盒 18 类 OBB 数据集。

Roboflow 的类别不能自动猜测。第一次只提供 ``--source`` 会生成映射模板；人工填写
映射、来源 URL 和许可证后再次运行，脚本才会转换标签并按图片哈希去重。
支持 YOLO 水平框、YOLO OBB 四点和 YOLO 分割多边形。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

from medicine_ocr import MEDICINE_NAMES


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_DATA = ROOT / "data" / "yolo" / "medicine" / "medicine_obb.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _default_destination() -> Path:
    configured = os.environ.get("YOLO_OBB_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser() / "datasets" / "medicine_roboflow_combined"
    return ROOT / "data" / "training" / "medicine_roboflow_combined"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导入 Roboflow YOLO 数据并转换为药盒 OBB 18 类。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Roboflow 解压目录或 ZIP")
    source.add_argument("--download-url", help="登录 Roboflow 后获得的导出 ZIP 下载 URL")
    parser.add_argument("--mapping", type=Path, default=None, help="人工确认的类别映射 YAML")
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA, help="当前药盒 OBB YAML")
    parser.add_argument("--destination", type=Path, default=_default_destination())
    parser.add_argument("--source-url", default=None, help="Roboflow Universe 数据集页面 URL")
    parser.add_argument("--license", dest="license_name", default=None, help="数据集许可证名称")
    return parser.parse_args(argv)


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _safe_extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"ZIP 包含越界路径：{member.filename}")
        archive.extractall(destination)
    return destination


def _obtain_source(args: argparse.Namespace, staging: Path) -> Path:
    if args.source is not None:
        source = _absolute(args.source)
        if source.is_dir():
            return source
        if not source.is_file() or not zipfile.is_zipfile(source):
            raise ValueError(f"--source 必须是目录或 ZIP：{source}")
        return _safe_extract(source, staging / "extracted")

    download = staging / "roboflow_export.zip"
    staging.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(str(args.download_url), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, download.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if not zipfile.is_zipfile(download):
        raise ValueError("下载结果不是 ZIP；请确认 Roboflow 导出 URL 仍有效")
    return _safe_extract(download, staging / "extracted")


def _find_yaml(source: Path) -> Path:
    preferred = list(source.rglob("data.yaml"))
    candidates = preferred or list(source.rglob("*.yaml")) + list(source.rglob("*.yml"))
    if not candidates:
        raise FileNotFoundError(f"Roboflow 导出中没有 data.yaml：{source}")
    return min(candidates, key=lambda path: len(path.parts))


def _read_yaml(path: Path) -> tuple[dict[str, Any], dict[int, str], Path]:
    with path.open("r", encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream) or {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError(f"YAML 缺少 names：{path}")
    raw_root = Path(str(data.get("path", path.parent)))
    dataset_root = raw_root if raw_root.is_absolute() else path.parent / raw_root
    return data, names, dataset_root.resolve()


def _split_image_dir(data: dict[str, Any], root: Path, split: str) -> Path | None:
    value = data.get(split)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"暂不支持多个 {split} 路径")
        value = value[0]
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if path.name.lower() != "images" and (path / "images").is_dir():
        path /= "images"
    return path


def _label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"图像路径中缺少 images：{image_path}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _write_mapping_template(path: Path, names: dict[int, str], source_yaml: Path) -> None:
    content = {
        "source_yaml": str(source_yaml),
        "target_names": list(MEDICINE_NAMES),
        "mapping": {name: None for _, name in sorted(names.items())},
        "notes": "把每个外部类别填成目标中文类名、0..17，或 null（丢弃）。",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(content, stream, allow_unicode=True, sort_keys=False)


def _load_mapping(path: Path, source_names: dict[int, str]) -> dict[int, int | None]:
    with path.open("r", encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream) or {}
    values = raw.get("mapping", raw)
    target_by_name = {name: index for index, name in enumerate(MEDICINE_NAMES)}
    target_by_name["其它"] = target_by_name["其他"]
    output: dict[int, int | None] = {}
    for source_id, source_name in source_names.items():
        value = values.get(source_name)
        if value is None or str(value).strip().lower() in {"", "null", "none", "skip"}:
            output[source_id] = None
        elif isinstance(value, int) or str(value).strip().isdigit():
            target = int(value)
            if not 0 <= target < len(MEDICINE_NAMES):
                raise ValueError(f"映射目标超出 0..17：{source_name} -> {target}")
            output[source_id] = target
        else:
            target_name = str(value).strip()
            if target_name not in target_by_name:
                raise ValueError(f"未知目标类别：{source_name} -> {target_name}")
            output[source_id] = target_by_name[target_name]
    return output


def _polygon_to_obb(coords: Sequence[float]) -> list[float]:
    points = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    box = cv2.boxPoints(cv2.minAreaRect(points)).reshape(-1)
    return np.clip(box, 0.0, 1.0).astype(float).tolist()


def _convert_line(line: str, mapping: dict[int, int | None]) -> tuple[int, list[float]] | None:
    values = [float(value) for value in line.split()]
    if len(values) < 5:
        raise ValueError(f"YOLO 标签列数不足：{line[:80]}")
    source_id = int(values[0])
    target_id = mapping.get(source_id)
    if target_id is None:
        return None
    coords = values[1:]
    if len(values) == 5:
        x, y, width, height = coords
        x1, x2 = x - width / 2, x + width / 2
        y1, y2 = y - height / 2, y + height / 2
        obb = [x1, y1, x2, y1, x2, y2, x1, y2]
    elif len(values) == 9:
        obb = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        obb = _polygon_to_obb(coords)
    else:
        raise ValueError(f"无法识别的 YOLO 标签格式（{len(values)} 列）")
    if not all(np.isfinite(obb)):
        raise ValueError("标签包含 NaN/Inf")
    return target_id, np.clip(np.asarray(obb), 0.0, 1.0).astype(float).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_files(directory: Path | None) -> Iterable[Path]:
    if directory is None or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _write_obb_label(path: Path, labels: Sequence[tuple[int, Sequence[float]]]) -> None:
    lines = [f"{class_id} " + " ".join(f"{value:.6f}" for value in coords) for class_id, coords in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    destination = _absolute(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"目标目录非空，为避免覆盖请换一个 --destination：{destination}")
    staging = destination.parent / f".{destination.name}_staging"
    source_root = _obtain_source(args, staging)
    source_yaml = _find_yaml(source_root)
    source_data, source_names, source_dataset_root = _read_yaml(source_yaml)

    if args.mapping is None:
        template = destination.parent / "roboflow_class_mapping.yaml"
        _write_mapping_template(template, source_names, source_yaml)
        print(f"已生成类别映射模板：{template}")
        print("请人工填写 mapping、确认 Roboflow 页面 URL 和许可证后重新运行。")
        return 0
    if not args.source_url or not args.license_name:
        raise ValueError("正式导入必须同时提供 --source-url 和 --license，确保数据可追溯")

    mapping = _load_mapping(_absolute(args.mapping), source_names)
    base_data, _, base_root = _read_yaml(_absolute(args.base_data))
    output_dirs = {
        split: {
            "images": destination / split / "images",
            "labels": destination / split / "labels",
        }
        for split in ("train", "valid", "test")
    }
    for dirs in output_dirs.values():
        dirs["images"].mkdir(parents=True, exist_ok=True)
        dirs["labels"].mkdir(parents=True, exist_ok=True)

    hashes: set[str] = set()
    provenance: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()

    def add_image(
        image: Path,
        labels: Sequence[tuple[int, Sequence[float]]],
        output_split: str,
        source_kind: str,
        source_split: str,
    ) -> bool:
        digest = _sha256(image)
        if digest in hashes:
            return False
        hashes.add(digest)
        name = f"{source_kind}_{digest[:16]}{image.suffix.lower()}"
        target_image = output_dirs[output_split]["images"] / name
        target_label = output_dirs[output_split]["labels"] / f"{Path(name).stem}.txt"
        shutil.copy2(image, target_image)
        _write_obb_label(target_label, labels)
        class_counts.update(class_id for class_id, _ in labels)
        provenance.append(
            {
                "output_image": str(target_image),
                "sha256": digest,
                "source_image": str(image),
                "source_kind": source_kind,
                "source_split": source_split,
                "source_url": args.source_url if source_kind == "rf" else "local",
                "license": args.license_name if source_kind == "rf" else "local",
            }
        )
        return True

    # 保持当前验证集和测试集不变，Roboflow 的所有划分只扩充训练集。
    for split, output_split in (
        ("train", "train"),
        ("val", "valid"),
        ("test", "test"),
    ):
        image_dir = _split_image_dir(base_data, base_root, split)
        for image in _image_files(image_dir):
            label = _label_for_image(image)
            if not label.is_file():
                continue
            labels = []
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    values = [float(value) for value in line.split()]
                    labels.append((int(values[0]), values[1:9]))
            add_image(image, labels, output_split, "base", split)

    imported = 0
    skipped_unmapped = 0
    for split in ("train", "val", "valid", "test"):
        image_dir = _split_image_dir(source_data, source_dataset_root, split)
        for image in _image_files(image_dir):
            label = _label_for_image(image)
            if not label.is_file():
                continue
            converted: list[tuple[int, list[float]]] = []
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    item = _convert_line(line, mapping)
                    if item is not None:
                        converted.append(item)
            if not converted:
                skipped_unmapped += 1
                continue
            if add_image(image, converted, "train", "rf", split):
                imported += 1

    output_yaml = destination / "medicine_obb_roboflow.yaml"
    config = {
        "path": destination.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(MEDICINE_NAMES),
        "names": list(MEDICINE_NAMES),
    }
    with output_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    provenance_path = destination / "provenance.csv"
    with provenance_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)

    print("\n========== Roboflow 导入完成 ==========")
    print(f"  外部新增图像 : {imported}")
    print(f"  未映射跳过   : {skipped_unmapped}")
    print(f"  类别目标数   : {dict(sorted(class_counts.items()))}")
    print(f"  训练 YAML    : {output_yaml}")
    print(f"  来源记录     : {provenance_path}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Prepare and apply a human review queue for unmatched medicine OCR samples."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from medicine_db import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DB_PATH,
    MedicineDatabase,
    add_medicine,
    normalize_name,
)
from medicine_ocr import read_obb_labels, suggest_class


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OCR_DIR = ROOT / "output/ocr_results/json"
DEFAULT_OUTPUT_DIR = ROOT / "output/unmatched_medicine_review"
DEFAULT_CORRECTIONS = Path(__file__).resolve().parent / "medicine_label_corrections.yaml"
DEFAULT_DECISIONS = Path(__file__).resolve().parent / "medicine_ocr_database_decisions.yaml"
APPROVE_VALUES = {"approve", "approved", "yes", "y", "通过", "确认"}
PRODUCT_FORM_PATTERN = re.compile(
    r"(?:缓释片|肠溶片|分散片|咀嚼片|泡腾片|含片|片剂|片|软胶囊|胶囊|"
    r"颗粒剂|颗粒|口服液|口服溶液|注射液|混悬滴剂|混悬液|糖浆|滴剂|"
    r"喷雾剂|气雾剂|滴眼液|滴耳液|洗液|搽剂|乳膏|软膏|贴膏|眼膏|"
    r"凝胶|敷料|创可贴|绷带|栓剂|栓|丸|散剂|散|膏|酊|茶|饮)$"
)
NOISE_PATTERN = re.compile(
    r"有限公司|制药厂|集团|批准文号|国药准字|功能主治|适应症|说明书|"
    r"购买和使用|药师指导|生产日期|有效期|专利号|汉语拼音|规格|成份|"
    r"用法用量|禁忌|不良反应|使用图示|OTC|医院专用|Pharmaceuticals",
    re.IGNORECASE,
)
NAME_PREFIX_PATTERN = re.compile(r"^(?:通用名称|药品名称)\s*[:：]\s*")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class NameCandidate:
    name: str
    score: float
    confidence: float
    has_product_form: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review OCR samples not found in medicine.db")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.78)
    parser.add_argument(
        "--apply",
        type=Path,
        metavar="CSV",
        help="Import rows whose decision column is approve/通过/确认",
    )
    args = parser.parse_args()
    if not 0.5 <= args.fuzzy_threshold <= 1.0:
        parser.error("--fuzzy-threshold must be in [0.5, 1.0]")
    return args


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def _label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _extract_name_candidates(lines: Iterable[dict[str, Any]]) -> list[NameCandidate]:
    by_name: dict[str, NameCandidate] = {}
    for line in lines:
        raw_value = str(line.get("text", "")).strip()
        had_name_prefix = bool(NAME_PREFIX_PATTERN.match(raw_value))
        value = NAME_PREFIX_PATTERN.sub("", raw_value)
        value = re.sub(r"\s+", "", value).strip("【】[]()（）:：,，。;；")
        normalized = normalize_name(value)
        if not 3 <= len(normalized) <= 28 or not CHINESE_PATTERN.search(value):
            continue
        if NOISE_PATTERN.search(value) or re.search(r"\d+(?:片|粒|袋|支|盒|克|毫克|ml)", value, re.I):
            continue
        confidence = float(line.get("confidence", 0.0))
        has_product_form = bool(PRODUCT_FORM_PATTERN.search(value) or value.startswith("注射用"))
        score = confidence * 4.0
        if has_product_form:
            score += 6.0
        if had_name_prefix:
            score += 3.0
        if 4 <= len(normalized) <= 16:
            score += 2.0
        if re.search(r"[A-Za-z]{5,}", value):
            score -= 2.0
        candidate = NameCandidate(value, round(score, 3), confidence, has_product_form)
        previous = by_name.get(normalized)
        if previous is None or candidate.score > previous.score:
            by_name[normalized] = candidate
    return sorted(by_name.values(), key=lambda item: (item.score, item.confidence), reverse=True)


def _load_manual_decisions(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    corrections = {
        (str(item.get("split", "train")), str(item["image"])): item
        for item in document.get("corrections", [])
    }
    reviewed_keep = {
        (str(item.get("split", "train")), str(item["image"])): item
        for item in document.get("reviewed_keep", [])
    }
    return corrections, reviewed_keep


def _load_existing_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {str(row.get("sample_id", "")): row for row in csv.DictReader(stream)}


def _load_database_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    return {str(item["sample_id"]): item for item in document.get("decisions", [])}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "decision",
        "confirmed_name",
        "confirmed_category_id",
        "confirmed_category_name",
        "sample_id",
        "candidate_name",
        "alternative_names",
        "name_score",
        "name_ocr_confidence",
        "dataset_label_ids",
        "dataset_label_names",
        "ocr_suggested_id",
        "ocr_suggested_name",
        "ocr_suggested_score",
        "review_status",
        "evidence",
        "image",
        "review_image",
        "ocr_json",
        "ocr_text",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _apply_review(args: argparse.Namespace) -> None:
    review_path = args.apply.expanduser().resolve()
    with review_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    approved = [row for row in rows if str(row.get("decision", "")).strip().casefold() in APPROVE_VALUES]
    approved_by_name: dict[str, int] = {}
    for row in approved:
        normalized = normalize_name(str(row.get("confirmed_name", "")))
        try:
            category_id = int(str(row.get("confirmed_category_id", "")).strip())
        except ValueError as exc:
            raise ValueError(f"{row.get('sample_id')}: invalid confirmed_category_id") from exc
        previous_category = approved_by_name.get(normalized)
        if previous_category is not None and previous_category != category_id:
            raise ValueError(
                f"approved medicine has conflicting categories: {row.get('confirmed_name')} "
                f"({previous_category} vs {category_id})"
            )
        approved_by_name[normalized] = category_id

    database_path = args.database.expanduser().resolve()
    backup_path = review_path.parent / "medicine_before_local_ocr_review.db"
    if database_path.is_file() and not backup_path.exists():
        source_connection = sqlite3.connect(database_path)
        backup_connection = sqlite3.connect(backup_path)
        try:
            source_connection.backup(backup_connection)
        finally:
            backup_connection.close()
            source_connection.close()
        print(f"Database backup: {backup_path}")

    imported = 0
    already_present = 0
    processed_names: set[str] = set()
    with MedicineDatabase(args.database, args.catalog, initialize=True) as database:
        category_names = {category_id: str(database.category(category_id)["name"]) for category_id in range(18)}
        for row in approved:
            name = str(row.get("confirmed_name", "")).strip()
            if len(normalize_name(name)) < 3:
                raise ValueError(f"{row.get('sample_id')}: confirmed_name is missing or too short")
            try:
                category_id = int(str(row.get("confirmed_category_id", "")).strip())
            except ValueError as exc:
                raise ValueError(f"{row.get('sample_id')}: invalid confirmed_category_id") from exc
            if category_id not in category_names:
                raise ValueError(f"{row.get('sample_id')}: category ID must be 0..17")
            normalized_name = normalize_name(name)
            if normalized_name in processed_names:
                continue
            processed_names.add(normalized_name)
            existing = database.lookup(name, lines=[name], fuzzy_threshold=1.0)
            if existing is not None and existing.score == 1.0:
                already_present += 1
                continue
            candidate_name = str(row.get("candidate_name", "")).strip()
            candidate_match = (
                database.lookup(candidate_name, lines=[candidate_name], fuzzy_threshold=1.0)
                if candidate_name
                else None
            )
            aliases = (
                [candidate_name]
                if candidate_name
                and normalize_name(candidate_name) != normalized_name
                and candidate_match is None
                else []
            )
            notes = json.dumps(
                {
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "sample_id": row.get("sample_id"),
                    "image": row.get("image"),
                    "evidence": row.get("evidence"),
                },
                ensure_ascii=False,
            )
            add_medicine(
                args.database,
                name,
                category_id,
                aliases=aliases,
                notes=notes,
                source="review:local_ocr",
            )
            imported += 1
    print(f"Approved rows: {len(approved)}, imported: {imported}, already present: {already_present}")


def main() -> None:
    args = parse_args()
    if args.apply:
        _apply_review(args)
        return

    ocr_dir = args.ocr_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    review_image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_image_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "unmatched_medicine_review.csv"
    existing_review = _load_existing_review(report_path)
    correction_map, keep_map = _load_manual_decisions(args.corrections.expanduser().resolve())
    database_decisions = _load_database_decisions(args.decisions.expanduser().resolve())

    rows: list[dict[str, Any]] = []
    total_ocr = 0
    with MedicineDatabase(args.database, args.catalog, initialize=True) as database:
        category_names = {category_id: str(database.category(category_id)["name"]) for category_id in range(18)}
        for json_path in sorted(ocr_dir.glob("*.json"), key=_numeric_sort_key):
            total_ocr += 1
            result = json.loads(json_path.read_text(encoding="utf-8-sig"))
            lines = result.get("lines") or []
            line_texts = [str(line.get("text", "")) for line in lines]
            if database.lookup(
                str(result.get("text", "")),
                lines=line_texts,
                fuzzy_threshold=args.fuzzy_threshold,
            ) is not None:
                continue

            image_path = Path(str(result["image"])).expanduser().resolve()
            sample_id = image_path.stem
            split = image_path.parent.parent.name
            label_path = _label_path(image_path)
            label_ids = sorted({class_id for class_id, _ in read_obb_labels(label_path)}) if label_path.is_file() else []
            candidates = _extract_name_candidates(lines)
            candidate = candidates[0] if candidates else NameCandidate("", 0.0, 0.0, False)
            suggestion = suggest_class(str(result.get("text", "")))
            manual_key = (split, image_path.name)
            correction = correction_map.get(manual_key)
            reviewed_keep = keep_map.get(manual_key)
            final_label: int | None = None
            evidence = ""
            manual_confirmed = False
            if correction is not None:
                final_label = int(correction["to"])
                evidence = f"已有人工修正：{correction.get('evidence', '')}"
                manual_confirmed = True
            elif reviewed_keep is not None:
                final_label = int(reviewed_keep["label"])
                evidence = f"已有人工确认保留：{reviewed_keep.get('reason', '')}"
                manual_confirmed = True
            elif len(label_ids) == 1:
                final_label = label_ids[0]
                evidence = "数据集只有一个类别标签，仍需核对药名和类别"
            else:
                evidence = "数据集含多个类别或缺少标签，必须人工选择"

            strong_name = candidate.has_product_form and candidate.score >= 10.0
            ocr_suggested_id = suggestion["class_id"]
            if manual_confirmed and strong_name:
                review_status = "manual_decision_ready"
                default_decision = "approve"
            elif strong_name and final_label is not None and ocr_suggested_id == final_label:
                review_status = "label_ocr_agree"
                default_decision = ""
                evidence += "；OCR 功效关键词与数据集类别一致"
            elif not candidate.name:
                review_status = "medicine_name_missing"
                default_decision = ""
            elif ocr_suggested_id is not None and final_label is not None and ocr_suggested_id != final_label:
                review_status = "category_conflict"
                default_decision = ""
                evidence += "；OCR 功效关键词与数据集类别冲突"
            else:
                review_status = "needs_review"
                default_decision = ""

            review_image = review_image_dir / image_path.name
            if image_path.is_file() and not review_image.exists():
                shutil.copy2(image_path, review_image)
            previous = existing_review.get(sample_id, {})
            curated_decision = database_decisions.get(sample_id)
            if curated_decision is not None:
                decision_value = str(curated_decision.get("decision", "")).strip()
                confirmed_name = str(curated_decision.get("name", "")).strip()
                confirmed_category_id = str(curated_decision.get("category_id", "")).strip()
                review_status = (
                    "curated_approved" if decision_value.casefold() in APPROVE_VALUES else "curated_rejected"
                )
                evidence = f"人工数据库复核：{curated_decision.get('reason', '')}"
            else:
                decision_value = previous.get("decision") or default_decision
                confirmed_name = previous.get("confirmed_name") or candidate.name
                confirmed_category_id = previous.get("confirmed_category_id") or (
                    str(final_label) if final_label is not None else ""
                )
            confirmed_category_name = (
                category_names.get(int(confirmed_category_id), "")
                if str(confirmed_category_id).isdigit()
                else ""
            )
            rows.append(
                {
                    "decision": decision_value,
                    "confirmed_name": confirmed_name,
                    "confirmed_category_id": confirmed_category_id,
                    "confirmed_category_name": confirmed_category_name,
                    "sample_id": sample_id,
                    "candidate_name": candidate.name,
                    "alternative_names": " | ".join(item.name for item in candidates[1:3]),
                    "name_score": candidate.score,
                    "name_ocr_confidence": round(candidate.confidence, 5),
                    "dataset_label_ids": "|".join(map(str, label_ids)),
                    "dataset_label_names": "|".join(category_names[item] for item in label_ids),
                    "ocr_suggested_id": "" if ocr_suggested_id is None else ocr_suggested_id,
                    "ocr_suggested_name": "" if ocr_suggested_id is None else category_names[ocr_suggested_id],
                    "ocr_suggested_score": suggestion["score"],
                    "review_status": review_status,
                    "evidence": evidence,
                    "image": str(image_path),
                    "review_image": str(review_image.resolve()),
                    "ocr_json": str(json_path),
                    "ocr_text": str(result.get("text", "")).replace("\r", " ").replace("\n", " | "),
                }
            )

    _write_csv(report_path, rows)
    statuses = Counter(str(row["review_status"]) for row in rows)
    decisions = Counter(str(row["decision"]) or "blank" for row in rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_ocr_samples": total_ocr,
        "unmatched_samples": len(rows),
        "statuses": dict(statuses),
        "decisions": dict(decisions),
        "report": str(report_path),
        "review_image_dir": str(review_image_dir),
        "curated_decisions": len(database_decisions),
        "apply_command": f"python day4/review_unmatched_medicines.py --apply \"{report_path}\"",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

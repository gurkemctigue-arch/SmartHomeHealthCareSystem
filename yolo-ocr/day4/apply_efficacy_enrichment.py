# -*- coding: utf-8 -*-
"""Apply reviewed per-medicine efficacy text to remaining generic catalog rows."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medicine_db import DEFAULT_DB_PATH


ENTRIES: dict[str, tuple[str, str]] = {
    "头孢氨苄注射液": (
        "用于敏感菌所致感染；本条药名来自本地包装复核，具体适应症须以该注射制剂说明书为准。",
        "curated:local-ocr-review",
    ),
    "奈玛特韦片/利托那韦片组合包装": (
        "用于治疗成人伴有进展为重症高风险因素的轻至中度新型冠状病毒感染，须严格遵医嘱使用。",
        "curated:local-ocr-review",
    ),
    "吗丁啉混悬液": (
        "用于胃排空延缓、胃食管反流、慢性胃炎或食管炎引起的消化不良，以及多种原因引起的恶心、呕吐。",
        "https://ypk.39.net/500780/manual/",
    ),
    "枯草杆菌、肠球菌二联活菌多维颗粒剂": (
        "用于消化不良、食欲不振、营养不良，以及肠道菌群紊乱引起的腹泻、便秘、腹胀等症状。",
        "https://ypk.39.net/2002421/manual/",
    ),
    "美沙拉嗪肠溶片": (
        "用于溃疡性结肠炎和节段性回肠炎（克罗恩病）。",
        "https://ypk.39.net/504018/manual/",
    ),
    "L-赖氨酸盐酸盐颗粒剂": (
        "用于补充赖氨酸，改善因赖氨酸缺乏导致的营养不足；具体适用人群和用量以包装说明书为准。",
        "curated:local-ocr-review",
    ),
    "福乃得维铁缓释片": (
        "用于明确原因的缺铁性贫血。",
        "https://ypk.39.net/501535/manual/",
    ),
    "丁苯羟酸乳膏": (
        "用于湿疹和神经性皮炎。",
        "https://ypk.39.net/500590/manual/",
    ),
    "复方曲安奈德乳膏": (
        "用于过敏性皮炎、湿疹、神经性皮炎、脂溢性皮炎和接触性皮炎等，也可用于部分念珠菌感染性皮肤病。",
        "https://ypk.39.net/501661/manual/",
    ),
    "复方硫酸软骨素滴眼液": (
        "用于视疲劳和干眼症。",
        "https://ypk.39.net/563855/manual/",
    ),
    "氧化锌硫软膏": (
        "用于皮炎和湿疹。",
        "https://ypk.39.net/741541/manual/",
    ),
    "高锰酸钾": (
        "配制成低浓度水溶液后用于急性皮炎、湿疹或创面的清洗湿敷；浓度和用法须严格按说明书执行。",
        "curated:catalog-review",
    ),
    "医用几丁糖液体敷料": (
        "适用于烧烫伤、褥疮、溃疡、皮肤湿疹、手术及外伤创口等皮肤创面的辅助护理。",
        "https://ypk.39.net/2296028/manual/",
    ),
    "医用硅酮凝胶敷料": (
        "适用于预防和辅助治疗烧伤、创伤或外科手术引起的增生性瘢痕。",
        "https://ypk.39.net/2309216/manual/",
    ),
    "医用重组人源胶原蛋白功能敷料": (
        "用于修复受损皮肤屏障，并辅助改善皮炎、湿疹和痤疮相关的皮肤症状。",
        "https://ypk.39.net/2266818/manual/",
    ),
    "弹性创可贴": (
        "用于小创口、擦伤等浅表创面的覆盖、保护和辅助固定。",
        "curated:catalog-review",
    ),
    "永益医用纱布块": (
        "用于创面清洁、吸收渗液、覆盖保护或辅助包扎，具体用途以产品说明书为准。",
        "curated:catalog-review",
    ),
    "防水创可贴": (
        "用于小创口、擦伤等浅表创面的防水覆盖和保护。",
        "curated:catalog-review",
    ),
    "甲状腺片": (
        "用于各种原因引起的甲状腺功能减退症。",
        "https://ypk.39.net/568492/manual/",
    ),
    "痔根断片": (
        "用于痔疮及其相关症状，如瘙痒、灼痛。",
        "https://ypk.39.net/510182/manual/",
    ),
    "天然维生素E软胶囊": (
        "用于补充维生素E；作为保健食品时不用于替代药物治疗，适宜人群和用量以产品标签为准。",
        "curated:catalog-review",
    ),
    "新成长快乐牌复合维生素咀嚼片": (
        "用于补充多种维生素和矿物质；属于保健食品，不用于替代药物治疗。",
        "curated:catalog-review",
    ),
    "百合康牌复合氨基酸维生素B片": (
        "用于补充氨基酸和B族维生素；属于保健食品，不用于替代药物治疗。",
        "curated:catalog-review",
    ),
    "维生素B族片": (
        "用于补充B族维生素；属于保健食品，不用于替代药物治疗。",
        "curated:catalog-review",
    ),
    "硫酸沙丁胺醇气雾剂": (
        "用于缓解哮喘或慢性阻塞性肺疾病患者的支气管痉挛，也可预防运动或过敏原诱发的支气管痉挛。",
        "https://ypk.39.net/762530/manual/",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply reviewed medicine efficacy enrichment")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "efficacy_enrichment",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _backup_database(database_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"medicine_before_efficacy_{timestamp}.db"
    source = sqlite3.connect(database_path)
    target = sqlite3.connect(backup_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return backup_path


def main() -> None:
    args = parse_args()
    database_path = args.database.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    backup_path = None if args.dry_run else _backup_database(database_path, output_dir)
    reviewed_at = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    report: list[dict[str, Any]] = []
    try:
        for name, (efficacy, source) in ENTRIES.items():
            row = connection.execute(
                "SELECT id, name, notes, efficacy_source FROM medicine WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                report.append({"name": name, "status": "missing", "source": source})
                continue
            if str(row["efficacy_source"]) != "catalog:category":
                report.append(
                    {
                        "name": name,
                        "status": "kept_specific",
                        "source": str(row["efficacy_source"]),
                    }
                )
                continue
            notes_text = str(row["notes"] or "")
            try:
                notes = json.loads(notes_text) if notes_text else {}
            except json.JSONDecodeError:
                notes = {"previous_notes": notes_text}
            notes["efficacy_enrichment"] = {"source": source, "reviewed_at": reviewed_at}
            status = "would_update" if args.dry_run else "updated"
            if not args.dry_run:
                connection.execute(
                    "UPDATE medicine SET efficacy = ?, efficacy_source = ?, notes = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (efficacy, source, json.dumps(notes, ensure_ascii=False), int(row["id"])),
                )
            report.append({"name": name, "status": status, "source": source})
        if not args.dry_run:
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": reviewed_at,
        "database": str(database_path),
        "backup": str(backup_path) if backup_path else None,
        "dry_run": args.dry_run,
        "statuses": {
            status: sum(item["status"] == status for item in report)
            for status in sorted({item["status"] for item in report})
        },
        "rows": report,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

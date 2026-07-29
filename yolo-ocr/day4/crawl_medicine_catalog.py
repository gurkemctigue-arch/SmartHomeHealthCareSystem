# -*- coding: utf-8 -*-
"""Crawl public medicine category pages and conservatively enrich SQLite.

The source site's robots.txt is checked before crawling. Raw pages are cached,
all imported rows retain provenance, and uncertain/conflicting items are written
to CSV for manual review instead of being silently classified.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from medicine_db import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DB_PATH,
    MedicineDatabase,
    add_medicine,
    normalize_name,
)


BASE_URL = "https://ypk.39.net"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
SOURCE_ID = "web:ypk.39.net"
USER_AGENT = "medicine-catalog-research/1.0 (+local AI training project)"
DETAIL_URL_PATTERN = re.compile(r"^https://ypk\.39\.net/\d+/?$")
TRAILING_BRAND_PATTERN = re.compile(r"(?:\([^()]*\)|（[^（）]*）)+$")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class SourceRule:
    path: str
    source_label: str
    category_id: int
    mode: str = "direct"
    confidence: float = 0.92

    @property
    def url(self) -> str:
        return urljoin(BASE_URL, self.path)


# The source site's categories are mapped to the 18 project categories. Broad
# or ambiguous pages are intentionally excluded from this automatic import.
SOURCE_RULES = (
    SourceRule("/ganmao/", "感冒发热", 0, "cold"),
    SourceRule("/bingduxingganmao/", "病毒性感冒", 0, "cold"),
    SourceRule("/zhikequtan/", "止咳祛痰", 1, "cough"),
    SourceRule("/xiaoerkesou/", "小儿止咳化痰", 1, "cough"),
    SourceRule("/biantaotiyan/", "扁桃体炎", 2, "antibiotic", 0.88),
    SourceRule("/jierezhentong/", "解热镇痛", 3, "pain_or_fever", 0.88),
    SourceRule("/toutongtouyun/", "头痛头晕", 3, "pain"),
    SourceRule("/yatong/", "牙痛", 3, "pain"),
    SourceRule("/jirouguanjietong/", "肌肉关节痛", 3, "pain"),
    SourceRule("/fengshi/", "风湿疼痛", 3, "pain"),
    SourceRule("/piantoutong/", "偏头痛", 3, "pain"),
    SourceRule("/kuiyang/", "胃肠溃疡", 5),
    SourceRule("/zhuxiaohua/", "助消化", 5),
    SourceRule("/zhixie/", "止泻", 5),
    SourceRule("/weiwantengtong/", "胃脘疼痛", 5),
    SourceRule("/xiaoerweichang/", "小儿胃肠用药", 5),
    SourceRule("/zhichuang/", "痔疮便秘便血", 15, "laxative_or_hemorrhoid", 0.90),
    SourceRule("/bugaibuxin/c1", "补钙补锌-药品", 7),
    SourceRule("/butie/c1", "补铁补硒-药品", 7),
    SourceRule("/weishengsu/c1", "维生素-药品", 7),
    SourceRule("/yesuan/c1", "叶酸-药品", 7),
    SourceRule("/weiliangyuansu/c1", "微量元素-药品", 7),
    SourceRule("/guomin/", "抗过敏", 8),
    SourceRule("/piyan/", "皮炎湿疹", 9, "external"),
    SourceRule("/yanke/", "眼科用药", 9, "external"),
    SourceRule("/huizhijia/", "灰指甲", 9, "external"),
    SourceRule("/search/%E5%88%9B%E5%8F%AF%E8%B4%B4", "搜索-创可贴", 10, "first_aid"),
    SourceRule("/search/%E5%8C%BB%E7%94%A8%E7%BA%B1%E5%B8%83", "搜索-医用纱布", 10, "first_aid"),
    SourceRule("/tangniaobing/", "糖尿病", 11),
    SourceRule("/jzxkj/", "甲状腺功能亢进", 11),
    SourceRule("/jzxjt/", "甲状腺功能减退", 11),
    SourceRule("/gaoxueya/", "高血压", 12),
    SourceRule("/gaoxuezhi/", "高血脂", 12),
    SourceRule("/zhouweixueguanjibing/", "周围血管疾病", 12),
    SourceRule("/guanxinbing/", "心脏及冠脉疾病", 12),
    SourceRule("/zhongfeng/", "中风偏瘫", 12),
    SourceRule("/yindaoyan/", "阴道炎", 14, "women"),
    SourceRule("/yuejingbutiao/", "月经不调", 14, "women"),
    SourceRule("/tongjing/", "痛经", 14, "women"),
    SourceRule("/baidaiyichang/", "白带异常", 14, "women"),
    SourceRule("/ruxianzengsheng/", "乳腺增生", 14, "women"),
    SourceRule("/weishengsu/c3", "维生素-保健品", 16),
    SourceRule("/bugaibuxin/c3", "补钙补锌-保健品", 16),
)


COLD_PATTERN = re.compile(r"感冒|清瘟|氨酚黄那敏|氨酚烷胺|酚麻美敏|小柴胡")
COUGH_PATTERN = re.compile(r"止咳|咳喘|咳嗽|化痰|祛痰|氨溴|川贝|枇杷|肺热")
FEVER_PATTERN = re.compile(r"退热|退烧|布洛芬混悬|对乙酰氨基酚(?:颗粒|混悬|口服液)")
PAIN_PATTERN = re.compile(
    r"痛|疼|布洛芬|对乙酰氨基酚|双氯芬酸|酚咖|尼美舒利|天麻|天舒|羊角|"
    r"新癀|独一味|氟桂利嗪|蠲痹|鸿茅|正天丸|通天口服液"
)
ANTIBIOTIC_PATTERN = re.compile(
    r"阿莫西林|头孢|青霉素|霉素|沙星|克拉维酸|硝唑|磺胺|抗菌|消炎"
)
EXTERNAL_PATTERN = re.compile(
    r"软膏|乳膏|滴眼液|眼膏|滴耳液|搽剂|喷雾剂|贴膏|洗液|酊|凝胶|外用|涂剂"
)
FIRST_AID_PATTERN = re.compile(r"创可贴|纱布|绷带|敷料")
LAXATIVE_PATTERN = re.compile(r"开塞露|乳果糖|通便|便秘|麻仁|芦荟胶囊")
HEMORRHOID_PATTERN = re.compile(r"痔|肛泰|马应龙")

# These corrections come from manual review of the first import against the PDF
# boundaries. They are applied only to rows sourced by this crawler.
CURATED_CATEGORY_OVERRIDES = {
    normalize_name(name): category_id
    for name, category_id in {
        "一清颗粒": 13,
        "云南白药胶囊": 13,
        "口腔溃疡散": 9,
        "安神胶囊": 13,
        "对乙酰氨基酚混悬滴剂": 4,
        "开喉剑喷雾剂": 9,
        "氨酚伪麻那敏片": 0,
        "牛黄上清丸": 13,
        "牛黄上清片": 13,
        "牛黄解毒片": 13,
        "眩晕宁片": 13,
        "麝香壮骨膏": 9,
        "黄连上清丸": 13,
        "黄连上清片": 13,
        "归脾丸": 13,
        "补中益气丸": 13,
        "异维A酸软胶囊": 17,
        "复方曲安奈德乳膏": 9,
        "氟轻松维B6乳膏": 9,
        "注射用还原型谷胱甘肽": 17,
        "酒石酸美托洛尔胶囊": 12,
        "了哥王片": 13,
        "替硝唑片": 2,
        "高锰酸钾": 9,
    }.items()
}
CURATED_REJECTIONS = {
    normalize_name(name)
    for name in (
        "吗丁啉片剂10mgOTC",
        "散结乳癖膏(散结乳癖贴膏",
        "VC90粒装",
        "牌维生素C咀嚼片",
    )
}


@dataclass
class RawCandidate:
    raw_name: str
    canonical_name: str
    category_id: int | None
    confidence: float
    source_label: str
    source_page: str
    detail_url: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl and enrich the medicine SQLite catalog")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output/medicine_catalog_import",
    )
    parser.add_argument("--max-items-per-source", type=int, default=10)
    parser.add_argument("--limit-sources", type=int, help="Only process the first N source pages")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--delay", type=float, help="Request delay; never lower than robots.txt")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached HTML")
    parser.add_argument("--dry-run", action="store_true", help="Crawl and report without database writes")
    args = parser.parse_args()
    if args.max_items_per_source < 1:
        parser.error("--max-items-per-source must be positive")
    if args.limit_sources is not None and args.limit_sources < 1:
        parser.error("--limit-sources must be positive")
    if not 0.5 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be in [0.5, 1.0]")
    return args


def canonicalize_name(raw_name: str) -> str:
    value = re.sub(r"\s+", "", raw_name).strip("-—|,，。")
    while True:
        cleaned = TRAILING_BRAND_PATTERN.sub("", value).strip()
        if cleaned == value:
            break
        value = cleaned
    return value


def classify(rule: SourceRule, name: str) -> tuple[int | None, float, str]:
    if rule.mode == "cold":
        if COUGH_PATTERN.search(name):
            return 1, rule.confidence, "咳嗽/化痰特征优先于感冒类"
        if not COLD_PATTERN.search(name):
            return None, 0.0, "感冒页药名缺少明确感冒特征"
        return 0, rule.confidence, "感冒页且药名含感冒特征"
    if rule.mode == "cough":
        if COLD_PATTERN.search(name) and not COUGH_PATTERN.search(name):
            return 0, rule.confidence, "止咳页中出现明确感冒药名，改归感冒药"
        if not COUGH_PATTERN.search(name):
            return None, 0.0, "止咳页药名缺少止咳/化痰特征"
        return 1, rule.confidence, "止咳页且药名含止咳/化痰特征"
    if rule.mode == "antibiotic":
        if not ANTIBIOTIC_PATTERN.search(name):
            return None, 0.0, "炎症相关页面但药名不能确认是抗感染药"
        return 2, rule.confidence, "药名含明确抗感染成分"
    if rule.mode == "pain_or_fever":
        if COLD_PATTERN.search(name):
            return 0, rule.confidence, "解热镇痛页中出现明确感冒药名，改归感冒药"
        if FEVER_PATTERN.search(name):
            return 4, rule.confidence, "药名明确强调退烧/退热剂型"
        if PAIN_PATTERN.search(name):
            return 3, rule.confidence, "解热镇痛页且药名含镇痛成分/特征"
        return None, 0.0, "解热镇痛页药名不能确认止痛或退烧边界"
    if rule.mode == "pain":
        if not PAIN_PATTERN.search(name):
            return None, 0.0, "疼痛相关页面但药名缺少镇痛成分/特征"
        return 3, rule.confidence, "疼痛页且药名含镇痛成分/特征"
    if rule.mode == "external":
        if not EXTERNAL_PATTERN.search(name):
            return None, 0.0, "外用相关页面但剂型不能确认是外用药"
        return 9, rule.confidence, "药名含明确外用剂型"
    if rule.mode == "first_aid":
        if not FIRST_AID_PATTERN.search(name):
            return None, 0.0, "搜索结果不含创可贴/纱布/绷带/敷料"
        return 10, rule.confidence, "药名明确是创可贴绷带类"
    if rule.mode == "laxative_or_hemorrhoid":
        if LAXATIVE_PATTERN.search(name):
            return 6, rule.confidence, "药名明确是通便药"
        if HEMORRHOID_PATTERN.search(name):
            return 15, rule.confidence, "药名明确是肛肠用药"
        return None, 0.0, "痔疮/便秘混合页面，药名无法确定边界"
    if rule.mode == "women":
        if ANTIBIOTIC_PATTERN.search(name):
            return 2, rule.confidence, "妇科页中的通用抗感染药，改归消炎药"
        if FEVER_PATTERN.search(name):
            return 4, rule.confidence, "妇科页中的明确退烧剂型，改归退烧药"
        if PAIN_PATTERN.search(name):
            return 3, rule.confidence, "妇科页中的通用镇痛药，改归止痛药"
        return 14, rule.confidence, "妇科功效分类直接映射"
    return rule.category_id, rule.confidence, "来源功效分类直接映射"


def apply_curated_database_corrections(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    moved = 0
    deleted = 0
    try:
        with connection:
            for normalized_name, category_id in CURATED_CATEGORY_OVERRIDES.items():
                cursor = connection.execute(
                    """
                    UPDATE medicine SET category_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE normalized_name = ? AND source = ?
                    """,
                    (category_id, normalized_name, SOURCE_ID),
                )
                moved += cursor.rowcount
            for normalized_name in CURATED_REJECTIONS:
                cursor = connection.execute(
                    "DELETE FROM medicine WHERE normalized_name = ? AND source = ?",
                    (normalized_name, SOURCE_ID),
                )
                deleted += cursor.rowcount
        return {"moved": moved, "deleted": deleted}
    finally:
        connection.close()


class CachedCrawler:
    def __init__(self, cache_dir: Path, requested_delay: float | None, refresh: bool) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"})
        robots_response = self.session.get(ROBOTS_URL, timeout=30)
        robots_response.raise_for_status()
        self.robots = RobotFileParser()
        self.robots.set_url(ROBOTS_URL)
        self.robots.parse(robots_response.text.splitlines())
        delay_values = re.findall(
            r"(?im)^\s*crawl-delay\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            robots_response.text,
        )
        robots_delay = max(
            [float(self.robots.crawl_delay("*") or 0.0), *(float(value) for value in delay_values)]
        )
        self.delay = max(float(robots_delay), float(requested_delay or 0.0))
        self.last_request = time.monotonic()
        print(f"robots.txt: allowed, crawl delay={self.delay:.1f}s")

    @staticmethod
    def _cache_name(url: str) -> str:
        parsed = urlparse(url)
        label = re.sub(r"[^a-zA-Z0-9._-]+", "_", parsed.path.strip("/") or "index")
        return label[:120] + ".html"

    def fetch(self, url: str) -> str:
        if not self.robots.can_fetch("*", url):
            raise PermissionError(f"robots.txt does not allow: {url}")
        cache_path = self.cache_dir / self._cache_name(url)
        if cache_path.is_file() and not self.refresh:
            return cache_path.read_text(encoding="utf-8")
        wait_seconds = self.delay - (time.monotonic() - self.last_request)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        response = self.session.get(url, timeout=30)
        self.last_request = time.monotonic()
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        cache_path.write_text(response.text, encoding="utf-8")
        return response.text


def parse_candidates(html: str, rule: SourceRule, limit: int) -> list[RawCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    output: list[RawCandidate] = []
    seen: set[tuple[str, str]] = set()
    for anchor in soup.select("a[href]"):
        detail_url = urljoin(BASE_URL, str(anchor.get("href", "")))
        if not DETAIL_URL_PATTERN.fullmatch(detail_url):
            continue
        raw_name = anchor.get_text(" ", strip=True).split("参考价", 1)[0]
        raw_name = re.sub(r"\s+", "", raw_name).strip()
        canonical_name = canonicalize_name(raw_name)
        key = (normalize_name(canonical_name), detail_url)
        if key in seen:
            continue
        seen.add(key)
        if len(normalize_name(canonical_name)) < 3 or not CHINESE_PATTERN.search(canonical_name):
            continue
        category_id, confidence, reason = classify(rule, canonical_name)
        output.append(
            RawCandidate(
                raw_name=raw_name,
                canonical_name=canonical_name,
                category_id=category_id,
                confidence=confidence,
                source_label=rule.source_label,
                source_page=rule.url,
                detail_url=detail_url,
                reason=reason,
            )
        )
        if len([item for item in output if item.category_id is not None]) >= limit:
            break
    return output


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = list(SOURCE_RULES[: args.limit_sources]) if args.limit_sources else list(SOURCE_RULES)
    crawler = CachedCrawler(output_dir / "cache", args.delay, args.refresh)

    raw_candidates: list[RawCandidate] = []
    failures: list[dict[str, str]] = []
    for index, rule in enumerate(rules, start=1):
        print(f"[{index:02d}/{len(rules):02d}] {rule.source_label}: {rule.url}", flush=True)
        try:
            html = crawler.fetch(rule.url)
            candidates = parse_candidates(html, rule, args.max_items_per_source)
            raw_candidates.extend(candidates)
            accepted = sum(item.category_id is not None for item in candidates)
            print(f"  parsed={len(candidates)}, classifiable={accepted}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_label": rule.source_label, "url": rule.url, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)

    grouped: dict[str, list[RawCandidate]] = defaultdict(list)
    for candidate in raw_candidates:
        grouped[normalize_name(candidate.canonical_name)].append(candidate)

    report_rows: list[dict[str, Any]] = []
    importable: list[tuple[str, int, list[str], list[RawCandidate]]] = []
    for normalized_name, items in grouped.items():
        curated_category_id = CURATED_CATEGORY_OVERRIDES.get(normalized_name)
        classifiable = [
            item
            for item in items
            if item.category_id is not None and item.confidence >= args.min_confidence
        ]
        category_ids = {int(item.category_id) for item in classifiable}
        if normalized_name in CURATED_REJECTIONS:
            status = "review_curated_rejection"
            reason = "人工审计判定名称含包装规格、残缺括号或缺少有效商品名"
            category_ids = set()
        elif curated_category_id is not None:
            status = "candidate"
            reason = "人工审计按 PDF 类别边界修正"
            category_ids = {curated_category_id}
            representative = max(items, key=lambda item: item.confidence)
            aliases = list(dict.fromkeys(item.raw_name for item in items))
            importable.append((representative.canonical_name, curated_category_id, aliases, items))
        elif not classifiable:
            status = "review_unclassified"
            reason = "; ".join(dict.fromkeys(item.reason for item in items))
        elif len(category_ids) > 1:
            status = "review_category_conflict"
            reason = f"来源给出多个类别：{sorted(category_ids)}"
        else:
            status = "candidate"
            reason = "; ".join(dict.fromkeys(item.reason for item in classifiable))
            representative = max(classifiable, key=lambda item: item.confidence)
            aliases = list(dict.fromkeys(item.raw_name for item in classifiable))
            importable.append(
                (representative.canonical_name, int(representative.category_id), aliases, classifiable)
            )
        representative = max(items, key=lambda item: item.confidence)
        report_rows.append(
            {
                "status": status,
                "canonical_name": representative.canonical_name,
                "raw_names": " | ".join(dict.fromkeys(item.raw_name for item in items)),
                "proposed_category_id": " | ".join(map(str, sorted(category_ids))),
                "confidence": max(item.confidence for item in items),
                "source_categories": " | ".join(dict.fromkeys(item.source_label for item in items)),
                "source_pages": " | ".join(dict.fromkeys(item.source_page for item in items)),
                "detail_urls": " | ".join(dict.fromkeys(item.detail_url for item in items)),
                "reason": reason,
            }
        )

    database_path = args.database.expanduser().resolve()
    backup_path: Path | None = None
    if database_path.is_file() and not args.dry_run:
        backup_path = output_dir / "medicine_before_web_import.db"
        if not backup_path.exists():
            source_connection = sqlite3.connect(database_path)
            backup_connection = sqlite3.connect(backup_path)
            try:
                source_connection.backup(backup_connection)
            finally:
                backup_connection.close()
                source_connection.close()
            print(f"Database backup: {backup_path}")
    correction_stats = {"moved": 0, "deleted": 0}
    if database_path.is_file() and not args.dry_run:
        correction_stats = apply_curated_database_corrections(database_path)
        print(f"Curated corrections: {correction_stats}")
    imported = 0
    already_present = 0
    database_conflicts = 0
    imported_by_category: Counter[int] = Counter()
    with MedicineDatabase(database_path, args.catalog, initialize=True) as database:
        existing_aliases = {
            str(row["normalized_alias"]): (int(row["medicine_id"]), int(row["category_id"]))
            for row in database.aliases
        }
        category_names = {category_id: database.category(category_id)["name"] for category_id in range(18)}
        for name, category_id, aliases, evidence in importable:
            normalized = normalize_name(name)
            existing = existing_aliases.get(normalized)
            row = next(item for item in report_rows if normalize_name(item["canonical_name"]) == normalized)
            if existing is not None:
                if existing[1] == category_id:
                    row["status"] = "already_present"
                    already_present += 1
                else:
                    row["status"] = "review_database_conflict"
                    row["reason"] = f"数据库类别 {existing[1]} 与抓取类别 {category_id} 冲突"
                    database_conflicts += 1
                continue
            safe_aliases = []
            for alias in aliases:
                alias_owner = existing_aliases.get(normalize_name(alias))
                if alias_owner is None:
                    safe_aliases.append(alias)
            if not args.dry_run:
                source_pages = list(dict.fromkeys(item.source_page for item in evidence))
                detail_urls = list(dict.fromkeys(item.detail_url for item in evidence))
                notes = json.dumps(
                    {
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "source_pages": source_pages,
                        "detail_urls": detail_urls[:5],
                    },
                    ensure_ascii=False,
                )
                medicine_id = add_medicine(
                    database_path,
                    name,
                    category_id,
                    aliases=safe_aliases,
                    notes=notes,
                    source=SOURCE_ID,
                )
                row["medicine_id"] = medicine_id
            row["status"] = "dry_run" if args.dry_run else "imported"
            row["category_name"] = category_names[category_id]
            imported += 1
            imported_by_category[category_id] += 1
            existing_aliases[normalized] = (-1, category_id)
            for alias in safe_aliases:
                existing_aliases[normalize_name(alias)] = (-1, category_id)

    report_fields = [
        "status",
        "medicine_id",
        "canonical_name",
        "raw_names",
        "proposed_category_id",
        "category_name",
        "confidence",
        "source_categories",
        "source_pages",
        "detail_urls",
        "reason",
    ]
    _write_csv(output_dir / "medicine_pairs.csv", report_rows, report_fields)
    _write_csv(output_dir / "crawl_failures.csv", failures, ["source_label", "url", "error"])
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": BASE_URL,
        "robots_url": ROBOTS_URL,
        "request_delay_seconds": crawler.delay,
        "source_pages": len(rules),
        "raw_candidates": len(raw_candidates),
        "unique_candidates": len(grouped),
        "imported_or_dry_run": imported,
        "already_present": already_present,
        "database_conflicts": database_conflicts,
        "failures": len(failures),
        "imported_by_category": dict(sorted(imported_by_category.items())),
        "dry_run": args.dry_run,
        "database_backup": str(backup_path) if backup_path else None,
        "curated_corrections": correction_stats,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'medicine_pairs.csv'}")


if __name__ == "__main__":
    main()

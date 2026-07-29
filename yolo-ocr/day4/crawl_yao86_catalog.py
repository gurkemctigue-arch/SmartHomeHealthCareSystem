# -*- coding: utf-8 -*-
"""Crawl public yao86 medicine pages and enrich the local SQLite catalog.

The crawler checks robots.txt, throttles network requests, caches source HTML,
keeps source provenance, and sends uncertain classifications to a review CSV.
Existing medicine categories and specific efficacy text are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from medicine_db import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DB_PATH,
    initialize_database,
    normalize_name,
)


BASE_URL = "https://www.yao86.com"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
SOURCE_ID = "web:yao86.com"
BRAND_ALIAS_SOURCE_ID = "web:yao86.com:brand"
USER_AGENT = "medicine-catalog-research/1.0 (+local medicine database enrichment)"
DEFAULT_DELAY_SECONDS = 0.25
MIN_DELAY_SECONDS = 0.20

SOURCE_LISTS = (
    ("western", "western", "西药", f"{BASE_URL}/a/全部西药"),
    ("tcm", "tcm", "中成药", f"{BASE_URL}/a/全部中成药"),
)
TARGETED_LISTS = (
    (
        "allergy",
        "western",
        "西药",
        f"{BASE_URL}/西药-药理/抗变态反应药",
    ),
    (
        "laxative_western",
        "western",
        "西药",
        f"{BASE_URL}/西药-药理/泻药及止泻药",
    ),
    (
        "laxative_tcm",
        "tcm",
        "中成药",
        f"{BASE_URL}/中成药-分类/泻下剂",
    ),
    (
        "hemorrhoid_tcm",
        "tcm",
        "中成药",
        f"{BASE_URL}/中成药-ATC/治疗痔疮和肛裂的药物",
    ),
)

FORMULATION_PATTERN = re.compile(
    r"(?:"
    r"片|胶囊|软胶囊|颗粒|丸|滴丸|散|粉|锭|膜|栓|贴|贴剂|贴膏|"
    r"乳膏|软膏|眼膏|凝胶|霜|酊|搽剂|洗剂|涂剂|糊剂|"
    r"滴眼液|滴鼻液|滴耳液|眼用制剂|鼻用制剂|"
    r"喷雾剂|鼻喷剂|气雾剂|粉雾剂|吸入剂|"
    r"注射液|注射剂|冻干粉针|植入剂|缓释制剂|"
    r"口服液|口服溶液|混悬液|乳剂|合剂|糖浆|露|酒|茶|"
    r"煎膏|膏|油|水|液|剂"
    r")(?:\([^)]*\)|（[^）]*）)?$"
)
REJECT_NAME_PATTERN = re.compile(r"(?:成方|原料药|中间体)$")
CLINICAL_ONLY_PATTERN = re.compile(
    r"注射|输液|粉针|冻干|植入|透析|疫苗|造影|麻醉|冲洗液"
)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")

FIRST_AID_PATTERN = re.compile(r"创可贴|绷带|纱布|敷料|创口贴")
EXTERNAL_NAME_PATTERN = re.compile(
    r"(?:乳膏|软膏|眼膏|凝胶|霜|酊|搽剂|洗剂|涂剂|涂膜|涂膜剂|糊剂|"
    r"滴眼液|滴鼻液|滴耳液|眼用制剂|鼻用制剂|"
    r"鼻喷剂|贴剂|贴膏)$"
)
MANUAL_CATEGORY_OVERRIDES = {
    normalize_name(name): 17
    for name in (
        "阿达木单抗注射液",
        "艾曲泊帕乙醇胺片",
        "马来酸阿伐曲泊帕片",
        "氨肽素片",
        "阿普米司特片",
    )
}


@dataclass(frozen=True)
class ListingCandidate:
    name: str
    url: str
    kind: str
    source_page: str


@dataclass(frozen=True)
class ParsedMedicine:
    name: str
    kind: str
    url: str
    efficacy: str
    pharmacology: tuple[str, ...]
    atc: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Classification:
    category_id: int | None
    confidence: float
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl yao86.com and enrich the medicine SQLite catalog"
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "output"
        / "yao86_catalog_import",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=500,
        help="Maximum number of formulation detail pages to inspect",
    )
    parser.add_argument(
        "--max-pages-per-kind",
        type=int,
        default=500,
        help="Safety limit for listing pages from each medicine kind",
    )
    parser.add_argument(
        "--page-order",
        choices=("balanced", "sequential"),
        default="balanced",
        help="Use uniformly distributed listing pages or crawl from page one",
    )
    parser.add_argument(
        "--targeted-items",
        type=int,
        default=300,
        help="Reserve detail candidates for currently underrepresented categories",
    )
    parser.add_argument(
        "--kind",
        choices=("all", "western", "tcm"),
        default="all",
        help="Limit crawling to western medicine or traditional Chinese medicine",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.86,
        help="Minimum classification confidence for new rows",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Network request delay in seconds (minimum {MIN_DELAY_SECONDS})",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached HTML")
    parser.add_argument(
        "--include-clinical",
        action="store_true",
        help="Include injections, vaccines and other primarily clinical formulations",
    )
    parser.add_argument(
        "--skip-existing-enrichment",
        action="store_true",
        help="Do not look up existing rows that only have category-level efficacy",
    )
    parser.add_argument(
        "--max-existing-enrichment",
        type=int,
        help="Limit exact-name lookups for existing generic-efficacy rows",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Crawl and report without database writes"
    )
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("--max-items must be positive")
    if args.max_pages_per_kind < 1:
        parser.error("--max-pages-per-kind must be positive")
    if not 0 <= args.targeted_items <= args.max_items:
        parser.error("--targeted-items must be between zero and --max-items")
    if args.max_existing_enrichment is not None and args.max_existing_enrichment < 1:
        parser.error("--max-existing-enrichment must be positive")
    if not 0.5 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be in [0.5, 1.0]")
    if args.delay < MIN_DELAY_SECONDS:
        parser.error(f"--delay must be at least {MIN_DELAY_SECONDS}")
    return args


def clean_text(value: str) -> str:
    value = " ".join(str(value).replace("\xa0", " ").split())
    return re.sub(r"\s*([，。；：、])\s*", r"\1", value).strip()


def is_likely_formulation(name: str) -> bool:
    normalized = clean_text(name)
    if REJECT_NAME_PATTERN.search(normalized):
        return False
    if not CHINESE_PATTERN.search(normalized):
        return False
    if not 2 <= len(normalize_name(normalized)) <= 80:
        return False
    return FORMULATION_PATTERN.search(normalized) is not None


def _decoded_path(url: str) -> str:
    return unquote(urlparse(url).path)


def parse_listing(
    html: str,
    kind: str,
    source_label: str,
    source_page: str,
    include_clinical: bool = False,
) -> list[ListingCandidate]:
    """Extract only dosage-form detail links from a yao86 listing page."""
    expected_prefix = f"/{source_label}/"
    soup = BeautifulSoup(html, "html.parser")
    output: list[ListingCandidate] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        raw_url = urljoin(BASE_URL, str(anchor.get("href", "")))
        parsed_url = urlparse(raw_url)
        decoded_path = _decoded_path(raw_url)
        if parsed_url.netloc not in {"www.yao86.com", "yao86.com"}:
            continue
        if not decoded_path.startswith(expected_prefix):
            continue
        if decoded_path.count("/") != 2 or "%%" in raw_url:
            continue
        name = clean_text(anchor.get_text(" ", strip=True))
        if not is_likely_formulation(name):
            continue
        if not include_clinical and CLINICAL_ONLY_PATTERN.search(name):
            continue
        canonical_url = f"{BASE_URL}{parsed_url.path}"
        key = normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            ListingCandidate(
                name=name,
                url=canonical_url,
                kind=kind,
                source_page=source_page,
            )
        )
    return output


def _metadata_values(soup: BeautifulSoup, label: str) -> tuple[str, ...]:
    for item in soup.select("li.item.category"):
        name_node = item.select_one(".item-name")
        if name_node is None or clean_text(name_node.get_text(" ", strip=True)).rstrip(":：") != label:
            continue
        values = [clean_text(anchor.get_text(" ", strip=True)) for anchor in item.select("a")]
        return tuple(dict.fromkeys(value for value in values if value))
    return ()


def _efficacy_text(soup: BeautifulSoup, kind: str) -> str:
    target = "【功能主治】" if kind == "tcm" else "【适应症】"
    for item in soup.select("dl.item"):
        heading = item.select_one("dt.item-name h3")
        if heading is None or clean_text(heading.get_text(" ", strip=True)) != target:
            continue
        content = item.select_one("dd.item-text")
        if content is None:
            return ""
        # Inline disease links belong inside Chinese words; a separator would
        # turn terms such as "高甘油三酯" into "高 甘油 三酯".
        return clean_text(content.get_text("", strip=True))
    return ""


def _brand_aliases(soup: BeautifulSoup, medicine_name: str) -> tuple[str, ...]:
    aliases: list[str] = []
    for item in soup.select("dl.item"):
        heading = item.select_one("dt.item-name h3")
        if heading is None or clean_text(heading.get_text("", strip=True)) != "【药品名称】":
            continue
        content = item.select_one("dd.item-text")
        if content is None:
            break
        fragment = BeautifulSoup(str(content), "html.parser")
        for line_break in fragment.select("br"):
            line_break.replace_with("\n")
        raw_text = fragment.get_text("", strip=False).replace("\xa0", " ")
        for match in re.finditer(r"商品名称[ \t]*[：:][ \t]*([^\r\n]*)", raw_text):
            value = clean_text(match.group(1))
            for alias in re.split(r"[、,，/;；]+", value):
                alias = clean_text(alias).strip("-—· ")
                normalized = normalize_name(alias)
                if alias in {"无", "暂无", "未标注", "不适用"}:
                    continue
                if re.search(r"通用名称|英文名称|汉语拼音|注册商标", alias):
                    continue
                if not CHINESE_PATTERN.search(alias):
                    continue
                if not 3 <= len(normalized) <= 20:
                    continue
                if normalized == normalize_name(medicine_name):
                    continue
                aliases.append(alias)
        break
    return tuple(dict.fromkeys(aliases))


def parse_detail(html: str, kind: str, url: str) -> ParsedMedicine:
    """Parse one detail page without crossing instruction-section boundaries."""
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.select_one("h1")
    name = clean_text(heading.get_text(" ", strip=True)) if heading else ""
    return ParsedMedicine(
        name=name,
        kind=kind,
        url=url,
        efficacy=_efficacy_text(soup, kind),
        pharmacology=_metadata_values(soup, "药理分类"),
        atc=_metadata_values(soup, "ATC分类"),
        aliases=_brand_aliases(soup, name),
    )


def efficacy_is_useful(value: str) -> bool:
    normalized = normalize_name(value)
    if len(normalized) < 8:
        return False
    weak_values = ("尚不明确", "详见说明书", "请仔细阅读说明书")
    return not any(normalized == normalize_name(item) for item in weak_values)


def _classification_text(medicine: ParsedMedicine) -> tuple[str, str]:
    metadata = " ".join((*medicine.pharmacology, *medicine.atc))
    full_text = " ".join((medicine.name, metadata, medicine.efficacy))
    return metadata, full_text


def classify_medicine(medicine: ParsedMedicine) -> Classification:
    """Map yao86 metadata into the project's fixed 18-category taxonomy."""
    metadata, full_text = _classification_text(medicine)
    name = medicine.name

    manual_category = MANUAL_CATEGORY_OVERRIDES.get(normalize_name(name))
    if manual_category is not None:
        return Classification(manual_category, 0.99, "人工审阅：不属于项目现有专用类别")
    if FIRST_AID_PATTERN.search(name):
        return Classification(10, 0.99, "药名明确为创口覆盖或包扎用品")
    if re.search(r"轻泻药|缓泻|泻下剂|通便药|便秘治疗药", metadata) or re.search(
        r"通便|泻下|便秘", name
    ):
        return Classification(6, 0.94, "站点分类或适应症明确为泻药通便")
    if re.search(r"止泻药|肠道吸附药|抗胃肠动力药", metadata):
        return Classification(5, 0.96, "站点分类明确为止泻或肠道吸附药")
    if re.search(r"治疗痔疮和肛裂|肛肠", metadata) or re.search(r"痔疮|消痔", name):
        return Classification(15, 0.95, "站点分类明确为痔疮或肛肠用药")
    if EXTERNAL_NAME_PATTERN.search(name) and re.search(r"皮肤病用", metadata):
        return Classification(9, 0.96, "外用剂型且站点 ATC 分类明确为皮肤局部用药")
    if re.search(r"妇产科用药|妇科用药|调经药物|避孕药|子宫收缩", metadata) or (
        EXTERNAL_NAME_PATTERN.search(name)
        and re.search(r"外阴|阴道|宫颈", medicine.efficacy)
    ):
        return Classification(14, 0.96, "站点药理或 ATC 分类明确为妇科用药")
    if re.search(r"镇咳|止咳|祛痰|化痰|粘液溶解", metadata) and re.search(
        r"吸入|喷雾剂|气雾剂", name
    ):
        return Classification(1, 0.95, "吸入剂型且站点分类明确为止咳或祛痰药")
    if EXTERNAL_NAME_PATTERN.search(name) or (
        re.search(r"喷雾剂|气雾剂", name)
        and not re.search(r"呼吸系统用药|平喘药|祛痰药", metadata)
    ) or (
        "皮肤科用药" in metadata
        and "止痒药" in metadata
        and re.search(r"溶液|液|水|剂$", name)
    ):
        return Classification(9, 0.94, "剂型或站点分类明确为局部外用药")
    if re.search(
        r"维生素、矿物质类药|维生素类|维生素制剂|矿物质补充剂|"
        r"微量元素补充剂|钙剂",
        metadata,
    ):
        return Classification(7, 0.96, "站点药理或 ATC 分类明确为维生素/矿物质")
    if re.search(r"抗组胺|抗过敏|变态反应", metadata):
        return Classification(8, 0.96, "站点药理分类明确为抗过敏药")
    if re.search(r"糖尿病|降血糖|胰岛素|甲状腺|血脂调节|调血脂", metadata):
        return Classification(11, 0.95, "站点药理或 ATC 分类明确为慢性病长期用药")
    if re.search(
        r"心血管|抗心绞痛|抗心律失常|抗心力衰竭|抗高血压|降压药|"
        r"冠心病|抗血栓|抗凝血|脑血管",
        metadata,
    ):
        return Classification(12, 0.95, "站点药理或 ATC 分类明确为心脑血管用药")
    if re.search(
        r"消化系统|胃病|胃肠|止吐|止泻|抗酸|消化不良|肝、胆疾病",
        metadata,
    ):
        return Classification(5, 0.93, "站点药理或 ATC 分类明确为胃肠消化用药")
    if re.search(r"镇咳|止咳|祛痰|化痰|粘液溶解", metadata):
        return Classification(1, 0.95, "站点药理分类明确为止咳或祛痰药")
    if re.search(r"平喘|支气管扩张", metadata):
        if medicine.kind == "tcm":
            return Classification(1, 0.93, "中成药分类明确为止咳平喘剂")
        return Classification(17, 0.96, "人工分类边界：西药支气管扩张剂归其它")
    if re.search(r"感冒用药|抗感冒|解表剂", metadata):
        return Classification(0, 0.94, "站点药理分类明确为感冒或解表用药")
    if re.search(
        r"抗微生物|抗菌药|抗生素|抗病毒|抗真菌|青霉素|头孢|"
        r"大环内酯|喹诺酮|磺胺类|抗结核",
        metadata,
    ):
        return Classification(2, 0.96, "站点药理或 ATC 分类明确为抗感染药")
    if re.search(r"镇痛药|非甾体抗炎|抗风湿|抗痛风|麻醉性镇痛", metadata):
        return Classification(3, 0.94, "站点药理或 ATC 分类明确为镇痛/抗炎药")
    if re.search(r"解热药|退热药|退烧药", metadata):
        return Classification(4, 0.92, "站点药理或 ATC 分类明确为解热药")

    if medicine.kind == "tcm":
        return Classification(13, 0.90, "站点明确收录为中成药，未命中更具体项目类别")
    return Classification(None, 0.0, "站点分类无法可靠映射到项目现有类别")


class CachedCrawler:
    def __init__(self, cache_dir: Path, requested_delay: float, refresh: bool) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"}
        )
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

        robots_response = self.session.get(ROBOTS_URL, timeout=(10, 30))
        robots_response.raise_for_status()
        self.robots = RobotFileParser()
        self.robots.set_url(ROBOTS_URL)
        self.robots.parse(robots_response.text.splitlines())
        robots_delay = float(self.robots.crawl_delay("*") or 0.0)
        self.delay = max(MIN_DELAY_SECONDS, requested_delay, robots_delay)
        self.last_request = time.monotonic()
        self.cache_hits = 0
        self.network_requests = 1
        print(f"robots.txt: allowed; request delay={self.delay:.2f}s")

    @staticmethod
    def _cache_name(url: str) -> str:
        parsed = urlparse(url)
        readable = clean_text(unquote(parsed.path)).strip("/") or "index"
        readable = re.sub(r"[^\w.-]+", "_", readable, flags=re.UNICODE)[:80]
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"{readable}_{digest}.html"

    def fetch(self, url: str) -> str:
        if not self.robots.can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt does not allow: {url}")
        cache_path = self.cache_dir / self._cache_name(url)
        if cache_path.is_file() and not self.refresh:
            self.cache_hits += 1
            return cache_path.read_text(encoding="utf-8")

        wait_seconds = self.delay - (time.monotonic() - self.last_request)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        response = self.session.get(url, timeout=(10, 45))
        self.last_request = time.monotonic()
        self.network_requests += 1
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        cache_path.write_text(response.text, encoding="utf-8")
        return response.text


def _listing_page_count(html: str, base_list_url: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    expected_path = unquote(urlparse(base_list_url).path)
    pages = [1]
    for anchor in soup.select("a[href]"):
        parsed = urlparse(urljoin(BASE_URL, str(anchor.get("href", ""))))
        if unquote(parsed.path) != expected_path:
            continue
        values = parse_qs(parsed.query).get("page", ())
        for value in values:
            if value.isdigit():
                pages.append(int(value))
    return max(pages)


def balanced_page_numbers(total_pages: int) -> list[int]:
    """Return a deterministic order whose every prefix spans the full range."""
    if total_pages <= 1:
        return [1]
    output = [1]
    seen = {1}
    denominator = 2
    while len(output) < total_pages:
        for numerator in range(1, denominator, 2):
            fraction = numerator / denominator
            page = int(fraction * (total_pages - 1) + 0.5) + 1
            if page not in seen:
                output.append(page)
                seen.add(page)
        denominator *= 2
        if denominator > total_pages * 4:
            break
    output.extend(page for page in range(1, total_pages + 1) if page not in seen)
    return output


def _discover_from_sources(
    crawler: CachedCrawler,
    sources: Sequence[tuple[str, str, str, str]],
    max_items: int,
    max_pages_per_source: int,
    page_order: str,
    include_clinical: bool,
    failures: list[dict[str, str]],
    seen: set[str],
) -> tuple[list[ListingCandidate], Counter[str]]:
    output: list[ListingCandidate] = []
    parsed_pages: Counter[str] = Counter()
    states: list[dict[str, Any]] = []
    for source_key, kind, source_label, base_list_url in sources:
        first_page_url = f"{base_list_url}?page=1"
        try:
            first_html = crawler.fetch(first_page_url)
            total_pages = _listing_page_count(first_html, base_list_url)
            ordered_pages = (
                balanced_page_numbers(total_pages)
                if page_order == "balanced"
                else list(range(1, total_pages + 1))
            )
            states.append(
                {
                    "key": source_key,
                    "kind": kind,
                    "label": source_label,
                    "url": base_list_url,
                    "pages": ordered_pages[:max_pages_per_source],
                    "index": 0,
                    "first_html": first_html,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "stage": "listing",
                    "name": "",
                    "url": first_page_url,
                    "error": str(exc),
                }
            )
            print(f"Listing failed: {first_page_url}: {exc}", flush=True)

    active = True
    while active and len(output) < max_items:
        active = False
        for state in states:
            if len(output) >= max_items:
                break
            index = int(state["index"])
            pages = state["pages"]
            if index >= len(pages):
                continue
            active = True
            page_number = int(pages[index])
            state["index"] = index + 1
            page_url = f"{state['url']}?page={page_number}"
            try:
                html = state["first_html"] if page_number == 1 else crawler.fetch(page_url)
                candidates = parse_listing(
                    html,
                    str(state["kind"]),
                    str(state["label"]),
                    page_url,
                    include_clinical=include_clinical,
                )
                parsed_pages[str(state["key"])] += 1
                for candidate in candidates:
                    key = normalize_name(candidate.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(candidate)
                    if len(output) >= max_items:
                        break
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"stage": "listing", "name": "", "url": page_url, "error": str(exc)}
                )
                print(f"Listing failed: {page_url}: {exc}", flush=True)
    return output, parsed_pages


def discover_candidates(
    crawler: CachedCrawler,
    max_items: int,
    max_pages_per_kind: int,
    selected_kind: str,
    page_order: str,
    targeted_items: int,
    include_clinical: bool,
    failures: list[dict[str, str]],
) -> tuple[list[ListingCandidate], dict[str, int]]:
    seen: set[str] = set()
    output: list[ListingCandidate] = []
    parsed_pages: Counter[str] = Counter()

    targeted_sources = [
        item for item in TARGETED_LISTS if selected_kind in {"all", item[1]}
    ]
    if targeted_items and targeted_sources:
        target_limit = min(targeted_items, max_items)
        base_quota, remainder = divmod(target_limit, len(targeted_sources))
        for index, source in enumerate(targeted_sources):
            quota = base_quota + (1 if index < remainder else 0)
            if quota <= 0:
                continue
            targeted, page_counts = _discover_from_sources(
                crawler,
                (source,),
                max_items=quota,
                max_pages_per_source=max_pages_per_kind,
                page_order="sequential",
                include_clinical=include_clinical,
                failures=failures,
                seen=seen,
            )
            output.extend(targeted)
            parsed_pages.update(page_counts)

    general_sources = [item for item in SOURCE_LISTS if selected_kind in {"all", item[1]}]
    remaining = max_items - len(output)
    if remaining > 0:
        general, page_counts = _discover_from_sources(
            crawler,
            general_sources,
            max_items=remaining,
            max_pages_per_source=max_pages_per_kind,
            page_order=page_order,
            include_clinical=include_clinical,
            failures=failures,
            seen=seen,
        )
        output.extend(general)
        parsed_pages.update(page_counts)
    return output, dict(parsed_pages)


def discover_existing_enrichment(
    crawler: CachedCrawler,
    database_path: Path,
    already_parsed: set[str],
    max_items: int | None,
) -> tuple[list[ParsedMedicine], list[dict[str, str]]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = list(
            connection.execute(
                """
                SELECT name, normalized_name, category_id
                FROM medicine
                WHERE efficacy_source = 'catalog:category'
                ORDER BY CASE WHEN source = 'catalog' THEN 0 ELSE 1 END, name
                """
            )
        )
    finally:
        connection.close()
    if max_items is not None:
        rows = rows[:max_items]

    parsed_items: list[ParsedMedicine] = []
    misses: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        normalized = str(row["normalized_name"])
        if normalized in already_parsed:
            continue
        name = str(row["name"])
        kind_order = ("tcm", "western") if int(row["category_id"]) == 13 else ("western", "tcm")
        attempted_urls: list[str] = []
        found: ParsedMedicine | None = None
        for kind in kind_order:
            source_label = "中成药" if kind == "tcm" else "西药"
            url = f"{BASE_URL}/{quote(source_label, safe='')}/{quote(name, safe='')}"
            attempted_urls.append(url)
            try:
                parsed = parse_detail(crawler.fetch(url), kind, url)
            except Exception:  # noqa: BLE001
                continue
            if normalize_name(parsed.name) != normalized:
                continue
            if efficacy_is_useful(parsed.efficacy):
                found = parsed
                break
        if found is not None:
            parsed_items.append(found)
            already_parsed.add(normalized)
        else:
            misses.append(
                {
                    "name": name,
                    "category_id": str(row["category_id"]),
                    "attempted_urls": " | ".join(attempted_urls),
                }
            )
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(
                f"Existing efficacy: {index}/{len(rows)}; found={len(parsed_items)}; "
                f"misses={len(misses)}",
                flush=True,
            )
    return parsed_items, misses


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _backup_database(database_path: Path, output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"medicine_before_yao86_{timestamp}.db"
    source = sqlite3.connect(database_path)
    target = sqlite3.connect(backup_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return backup_path


def _existing_maps(connection: sqlite3.Connection) -> tuple[dict[str, sqlite3.Row], dict[str, int]]:
    medicines = {
        str(row["normalized_name"]): row
        for row in connection.execute(
            "SELECT id, name, normalized_name, category_id, source, efficacy, efficacy_source "
            "FROM medicine"
        )
    }
    aliases = {
        str(row["normalized_alias"]): int(row["medicine_id"])
        for row in connection.execute("SELECT medicine_id, normalized_alias FROM medicine_alias")
    }
    return medicines, aliases


def import_medicines(
    database_path: Path,
    parsed_items: Sequence[ParsedMedicine],
    min_confidence: float,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[int], Counter[str]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    report: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    imported_categories: Counter[int] = Counter()
    alias_stats: Counter[str] = Counter()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        invalid_brand_alias_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM medicine_alias
                WHERE source = ? AND (
                    LENGTH(normalized_alias) < 3
                    OR alias LIKE '%英文名称%'
                    OR alias LIKE '%汉语拼音%'
                    OR alias LIKE '%通用名称%'
                )
                """,
                (BRAND_ALIAS_SOURCE_ID,),
            ).fetchone()[0]
        )
        if invalid_brand_alias_count:
            alias_stats["would_remove_invalid" if dry_run else "removed_invalid"] = (
                invalid_brand_alias_count
            )
            if not dry_run:
                connection.execute(
                    """
                    DELETE FROM medicine_alias
                    WHERE source = ? AND (
                        LENGTH(normalized_alias) < 3
                        OR alias LIKE '%英文名称%'
                        OR alias LIKE '%汉语拼音%'
                        OR alias LIKE '%通用名称%'
                    )
                    """,
                    (BRAND_ALIAS_SOURCE_ID,),
                )
        medicines, aliases = _existing_maps(connection)
        parsed_alias_owners: dict[str, set[str]] = defaultdict(set)
        for item in parsed_items:
            owner = normalize_name(item.name)
            for alias in item.aliases:
                normalized_alias = normalize_name(alias)
                if normalized_alias:
                    parsed_alias_owners[normalized_alias].add(owner)
        ambiguous_aliases = {
            alias for alias, owners in parsed_alias_owners.items() if len(owners) > 1
        }
        blocked_aliases = set(ambiguous_aliases)
        for item in parsed_items:
            normalized = normalize_name(item.name)
            classification = classify_medicine(item)
            medicine_id_for_aliases: int | None = None
            row: dict[str, Any] = {
                "status": "",
                "medicine_id": "",
                "name": item.name,
                "kind": item.kind,
                "existing_category_id": "",
                "category_id": classification.category_id
                if classification.category_id is not None
                else "",
                "confidence": classification.confidence,
                "efficacy": item.efficacy,
                "pharmacology": " | ".join(item.pharmacology),
                "atc": " | ".join(item.atc),
                "aliases": " | ".join(item.aliases),
                "aliases_added": "",
                "alias_conflicts": "",
                "detail_url": item.url,
                "reason": classification.reason,
            }
            if not normalized or not efficacy_is_useful(item.efficacy):
                row["status"] = "review_missing_efficacy"
            elif normalized in medicines:
                existing = medicines[normalized]
                row["medicine_id"] = int(existing["id"])
                medicine_id_for_aliases = int(existing["id"])
                row["existing_category_id"] = int(existing["category_id"])
                if str(existing["source"]) == SOURCE_ID:
                    if (
                        classification.category_id is None
                        or classification.confidence < min_confidence
                    ):
                        row["status"] = "review_existing_unclassified"
                        row["reason"] = "爬虫规则更新后无法可靠分类；数据库原记录暂未删除"
                    else:
                        desired_category = int(classification.category_id)
                        category_changed = desired_category != int(existing["category_id"])
                        row["status"] = (
                            "would_reclassify_existing"
                            if dry_run and category_changed
                            else "reclassified_existing"
                            if category_changed
                            else "already_present"
                        )
                        if not dry_run:
                            connection.execute(
                                "UPDATE medicine SET category_id = ?, efficacy = ?, "
                                "efficacy_source = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                                (desired_category, item.efficacy, item.url, int(existing["id"])),
                            )
                        row["category_id"] = desired_category
                else:
                    existing_source = str(existing["efficacy_source"] or "")
                    can_enrich = (
                        not str(existing["efficacy"] or "").strip()
                        or existing_source == "catalog:category"
                    )
                    if can_enrich:
                        row["status"] = "would_enrich_existing" if dry_run else "enriched_existing"
                        if not dry_run:
                            connection.execute(
                                "UPDATE medicine SET efficacy = ?, efficacy_source = ?, "
                                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                                (item.efficacy, item.url, int(existing["id"])),
                            )
                    else:
                        row["status"] = "already_present"
                    row["category_id"] = int(existing["category_id"])
                    row["reason"] = "保留数据库现有分类；" + row["reason"]
            elif normalized in aliases:
                row["status"] = "review_existing_alias"
                row["medicine_id"] = aliases[normalized]
                row["reason"] = "该名称已是另一个药品的别名，未自动改写归属"
            elif classification.category_id is None or classification.confidence < min_confidence:
                row["status"] = "review_unclassified"
            else:
                category_id = int(classification.category_id)
                row["status"] = "dry_run_new" if dry_run else "imported"
                medicine_id_for_aliases = -(len(report) + 1)
                if not dry_run:
                    notes = json.dumps(
                        {
                            "retrieved_at": retrieved_at,
                            "detail_url": item.url,
                            "pharmacology": item.pharmacology,
                            "atc": item.atc,
                            "classification_reason": classification.reason,
                            "classification_confidence": classification.confidence,
                        },
                        ensure_ascii=False,
                    )
                    cursor = connection.execute(
                        """
                        INSERT INTO medicine(
                            name, normalized_name, category_id, source, notes,
                            efficacy, efficacy_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item.name,
                            normalized,
                            category_id,
                            SOURCE_ID,
                            notes,
                            item.efficacy,
                            item.url,
                        ),
                    )
                    medicine_id = int(cursor.lastrowid)
                    medicine_id_for_aliases = medicine_id
                    connection.execute(
                        """
                        INSERT INTO medicine_alias(
                            medicine_id, alias, normalized_alias, source
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (medicine_id, item.name, normalized, SOURCE_ID),
                    )
                    row["medicine_id"] = medicine_id
                    medicines[normalized] = connection.execute(
                        "SELECT id, name, normalized_name, category_id, source, efficacy, efficacy_source "
                        "FROM medicine WHERE id = ?",
                        (medicine_id,),
                    ).fetchone()
                    aliases[normalized] = medicine_id
                else:
                    aliases[normalized] = medicine_id_for_aliases
                imported_categories[category_id] += 1

            added_aliases: list[str] = []
            alias_conflicts: list[str] = []
            if medicine_id_for_aliases is not None and not str(row["status"]).startswith("review_"):
                for alias in item.aliases:
                    normalized_alias = normalize_name(alias)
                    if len(normalized_alias) < 2 or normalized_alias == normalized:
                        continue
                    if normalized_alias in blocked_aliases:
                        alias_conflicts.append(f"{alias}（多药共用）")
                        alias_stats["ambiguous"] += 1
                        continue
                    owner = aliases.get(normalized_alias)
                    if owner is not None:
                        if owner != medicine_id_for_aliases:
                            alias_conflicts.append(alias)
                            alias_stats["conflicts"] += 1
                            blocked_aliases.add(normalized_alias)
                            alias_row = connection.execute(
                                "SELECT source FROM medicine_alias WHERE normalized_alias = ?",
                                (normalized_alias,),
                            ).fetchone()
                            if (
                                alias_row is not None
                                and str(alias_row["source"]) == BRAND_ALIAS_SOURCE_ID
                            ):
                                alias_stats[
                                    "would_remove_ambiguous_existing"
                                    if dry_run
                                    else "removed_ambiguous_existing"
                                ] += 1
                                if not dry_run:
                                    connection.execute(
                                        "DELETE FROM medicine_alias WHERE normalized_alias = ?",
                                        (normalized_alias,),
                                    )
                                    aliases.pop(normalized_alias, None)
                        continue
                    if not dry_run:
                        connection.execute(
                            """
                            INSERT INTO medicine_alias(
                                medicine_id, alias, normalized_alias, source
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                medicine_id_for_aliases,
                                alias,
                                normalized_alias,
                                BRAND_ALIAS_SOURCE_ID,
                            ),
                        )
                    aliases[normalized_alias] = medicine_id_for_aliases
                    added_aliases.append(alias)
                    alias_stats["would_add" if dry_run else "added"] += 1
                if added_aliases and row["status"] == "already_present":
                    row["status"] = (
                        "would_add_alias_existing" if dry_run else "aliases_added_existing"
                    )
            row["aliases_added"] = " | ".join(added_aliases)
            row["alias_conflicts"] = " | ".join(alias_conflicts)
            statuses[str(row["status"])] += 1
            report.append(row)
        if not dry_run:
            connection.commit()
        else:
            connection.rollback()
        return report, statuses, imported_categories, alias_stats
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.database.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    if args.dry_run and not database_path.is_file():
        raise FileNotFoundError(
            f"Dry-run requires an existing initialized database: {database_path}"
        )
    failures: list[dict[str, str]] = []
    crawler = CachedCrawler(output_dir / "cache", args.delay, args.refresh)

    candidates, listing_pages = discover_candidates(
        crawler,
        max_items=args.max_items,
        max_pages_per_kind=args.max_pages_per_kind,
        selected_kind=args.kind,
        page_order=args.page_order,
        targeted_items=args.targeted_items,
        include_clinical=args.include_clinical,
        failures=failures,
    )
    print(
        f"Discovered {len(candidates)} formulation pages from {listing_pages}",
        flush=True,
    )

    parsed_items: list[ParsedMedicine] = []
    parsed_names: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        try:
            html = crawler.fetch(candidate.url)
            parsed = parse_detail(html, candidate.kind, candidate.url)
            if not parsed.name:
                raise ValueError("detail page has no h1 medicine name")
            if REJECT_NAME_PATTERN.search(parsed.name):
                continue
            normalized = normalize_name(parsed.name)
            if normalized in parsed_names:
                continue
            parsed_items.append(parsed)
            parsed_names.add(normalized)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "stage": "detail",
                    "name": candidate.name,
                    "url": candidate.url,
                    "error": str(exc),
                }
            )
        if index == 1 or index % 25 == 0 or index == len(candidates):
            print(
                f"Details: {index}/{len(candidates)}; parsed={len(parsed_items)}; "
                f"failures={len(failures)}",
                flush=True,
            )

    enrichment_misses: list[dict[str, str]] = []
    enrichment_items: list[ParsedMedicine] = []
    if database_path.is_file() and not args.skip_existing_enrichment:
        enrichment_items, enrichment_misses = discover_existing_enrichment(
            crawler,
            database_path,
            already_parsed=parsed_names,
            max_items=args.max_existing_enrichment,
        )
        parsed_items.extend(enrichment_items)
        print(
            f"Existing efficacy enrichment: found={len(enrichment_items)}; "
            f"misses={len(enrichment_misses)}",
            flush=True,
        )

    backup_path: Path | None = None
    if args.dry_run:
        pass
    else:
        if database_path.is_file():
            backup_path = _backup_database(database_path, output_dir)
            print(f"Database backup: {backup_path}", flush=True)
        initialize_database(database_path, catalog_path)
    if not args.dry_run and backup_path is None:
        # A newly created database has no prior state to back up.
        print("Database did not exist; initialized a new catalog", flush=True)

    report, statuses, imported_categories, alias_stats = import_medicines(
        database_path,
        parsed_items,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
    )
    report_fields = (
        "status",
        "medicine_id",
        "name",
        "kind",
        "existing_category_id",
        "category_id",
        "confidence",
        "efficacy",
        "pharmacology",
        "atc",
        "aliases",
        "aliases_added",
        "alias_conflicts",
        "detail_url",
        "reason",
    )
    _write_csv(output_dir / "medicine_import.csv", report, report_fields)
    review_rows = [row for row in report if str(row["status"]).startswith("review_")]
    _write_csv(output_dir / "medicine_review.csv", review_rows, report_fields)
    _write_csv(
        output_dir / "crawl_failures.csv",
        failures,
        ("stage", "name", "url", "error"),
    )
    _write_csv(
        output_dir / "existing_efficacy_misses.csv",
        enrichment_misses,
        ("name", "category_id", "attempted_urls"),
    )

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        database_counts = {
            "medicines": int(connection.execute("SELECT COUNT(*) FROM medicine").fetchone()[0]),
            "aliases": int(connection.execute("SELECT COUNT(*) FROM medicine_alias").fetchone()[0]),
            "nonempty_efficacy": int(
                connection.execute(
                    "SELECT COUNT(*) FROM medicine WHERE TRIM(efficacy) <> ''"
                ).fetchone()[0]
            ),
            "specific_efficacy": int(
                connection.execute(
                    "SELECT COUNT(*) FROM medicine "
                    "WHERE efficacy_source NOT IN ('', 'catalog:category')"
                ).fetchone()[0]
            ),
            "generic_efficacy": int(
                connection.execute(
                    "SELECT COUNT(*) FROM medicine "
                    "WHERE efficacy_source = 'catalog:category'"
                ).fetchone()[0]
            ),
        }
        source_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM medicine WHERE source = ?", (SOURCE_ID,)
            ).fetchone()[0]
        )
        source_rows_by_category = {
            int(row["category_id"]): int(row["count"])
            for row in connection.execute(
                "SELECT category_id, COUNT(*) count FROM medicine "
                "WHERE source = ? GROUP BY category_id ORDER BY category_id",
                (SOURCE_ID,),
            )
        }
    finally:
        connection.close()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": BASE_URL,
        "robots_url": ROBOTS_URL,
        "request_delay_seconds": crawler.delay,
        "network_requests": crawler.network_requests,
        "cache_hits": crawler.cache_hits,
        "listing_pages": listing_pages,
        "page_order": args.page_order,
        "targeted_items": args.targeted_items,
        "include_clinical": args.include_clinical,
        "detail_candidates": len(candidates),
        "parsed_details": len(parsed_items),
        "existing_efficacy_found": len(enrichment_items),
        "existing_efficacy_misses": len(enrichment_misses),
        "statuses": dict(sorted(statuses.items())),
        "brand_aliases": dict(sorted(alias_stats.items())),
        "new_rows_by_category": dict(sorted(imported_categories.items())),
        "source_rows": source_rows,
        "source_rows_by_category": source_rows_by_category,
        "failures": len(failures),
        "dry_run": args.dry_run,
        "database": str(database_path),
        "database_backup": str(backup_path) if backup_path else None,
        "database_counts": database_counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"Import report: {output_dir / 'medicine_import.csv'}", flush=True)
    print(f"Review report: {output_dir / 'medicine_review.csv'}", flush=True)


if __name__ == "__main__":
    main()

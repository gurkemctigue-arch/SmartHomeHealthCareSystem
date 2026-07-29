# -*- coding: utf-8 -*-
"""SQLite medicine-name catalog used by the YOLO + OCR camera pipeline."""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_DB_PATH = ROOT / "db" / "medicine.db"
DEFAULT_CATALOG_PATH = SCRIPT_DIR / "medicine_catalog.yaml"


def normalize_name(value: str) -> str:
    """Normalize OCR text and catalog aliases for matching."""
    return re.sub(r"[\W_]+", "", str(value).casefold(), flags=re.UNICODE)


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS medicine_category (
    id INTEGER PRIMARY KEY CHECK (id BETWEEN 0 AND 17),
    name TEXT NOT NULL UNIQUE,
    guidance TEXT NOT NULL DEFAULT '',
    efficacy TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS medicine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    category_id INTEGER NOT NULL REFERENCES medicine_category(id),
    source TEXT NOT NULL DEFAULT 'catalog',
    notes TEXT NOT NULL DEFAULT '',
    efficacy TEXT NOT NULL DEFAULT '',
    efficacy_source TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS medicine_alias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_id INTEGER NOT NULL REFERENCES medicine(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL DEFAULT 'catalog'
);
CREATE INDEX IF NOT EXISTS idx_medicine_category ON medicine(category_id);
CREATE INDEX IF NOT EXISTS idx_alias_medicine ON medicine_alias(medicine_id);
"""


@dataclass(frozen=True)
class MedicineMatch:
    medicine_id: int
    medicine_name: str
    category_id: int
    category_name: str
    efficacy: str
    efficacy_source: str
    matched_alias: str
    score: float
    match_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _read_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        catalog = yaml.safe_load(stream) or {}
    categories = catalog.get("categories") or []
    medicines = catalog.get("medicines") or []
    ids = [int(item["id"]) for item in categories]
    if sorted(ids) != list(range(18)):
        raise ValueError("药品目录必须包含且只包含类别 ID 0..17")
    names = [str(item["name"]).strip() for item in categories]
    if len(set(names)) != 18:
        raise ValueError("18 个类别名称必须唯一")

    aliases: dict[str, str] = {}
    for item in medicines:
        category_id = int(item["category_id"])
        if category_id not in ids:
            raise ValueError(f"药品 {item.get('name')} 的类别不存在：{category_id}")
        values = [item["name"], *(item.get("aliases") or [])]
        for value in values:
            normalized = normalize_name(str(value))
            if len(normalized) < 2:
                raise ValueError(f"药名或别名过短：{value!r}")
            previous = aliases.get(normalized)
            if previous is not None and previous != str(item["name"]):
                raise ValueError(f"药名别名冲突：{value!r} 同时属于 {previous} 和 {item['name']}")
            aliases[normalized] = str(item["name"])
    return catalog


def _ensure_schema_columns(connection: sqlite3.Connection) -> None:
    category_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(medicine_category)")
    }
    if "efficacy" not in category_columns:
        connection.execute(
            "ALTER TABLE medicine_category "
            "ADD COLUMN efficacy TEXT NOT NULL DEFAULT ''"
        )

    medicine_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(medicine)")
    }
    if "efficacy" not in medicine_columns:
        connection.execute(
            "ALTER TABLE medicine ADD COLUMN efficacy TEXT NOT NULL DEFAULT ''"
        )
    if "efficacy_source" not in medicine_columns:
        connection.execute(
            "ALTER TABLE medicine "
            "ADD COLUMN efficacy_source TEXT NOT NULL DEFAULT ''"
        )


def initialize_database(
    db_path: Path = DEFAULT_DB_PATH,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> dict[str, int]:
    """Create/update the SQLite database without deleting user-added rows."""
    db_path = Path(db_path).expanduser().resolve()
    catalog_path = Path(catalog_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = _read_catalog(catalog_path)
    connection = _connect(db_path)
    try:
        connection.executescript(SCHEMA)
        _ensure_schema_columns(connection)
        with connection:
            for category in catalog["categories"]:
                connection.execute(
                    """
                    INSERT INTO medicine_category(id, name, guidance, efficacy)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        guidance = excluded.guidance,
                        efficacy = excluded.efficacy
                    """,
                    (
                        int(category["id"]),
                        str(category["name"]).strip(),
                        str(category.get("guidance", "")).strip(),
                        str(category.get("efficacy", "")).strip(),
                    ),
                )

            for item in catalog["medicines"]:
                name = str(item["name"]).strip()
                normalized_name = normalize_name(name)
                category_id = int(item["category_id"])
                category = next(
                    category
                    for category in catalog["categories"]
                    if int(category["id"]) == category_id
                )
                item_efficacy = str(item.get("efficacy", "")).strip()
                efficacy = item_efficacy or str(
                    category.get("efficacy", "")
                ).strip()
                efficacy_source = str(
                    item.get(
                        "efficacy_source",
                        "catalog:medicine" if item_efficacy else "catalog:category",
                    )
                ).strip()
                connection.execute(
                    """
                    INSERT INTO medicine(
                        name, normalized_name, category_id, source, notes,
                        efficacy, efficacy_source
                    )
                    VALUES (?, ?, ?, 'catalog', ?, ?, ?)
                    ON CONFLICT(normalized_name) DO UPDATE SET
                        name = excluded.name,
                        category_id = excluded.category_id,
                        notes = excluded.notes,
                        efficacy = CASE
                            WHEN excluded.efficacy_source = 'catalog:category'
                                AND medicine.efficacy_source NOT IN ('', 'catalog:category')
                            THEN medicine.efficacy
                            ELSE excluded.efficacy
                        END,
                        efficacy_source = CASE
                            WHEN excluded.efficacy_source = 'catalog:category'
                                AND medicine.efficacy_source NOT IN ('', 'catalog:category')
                            THEN medicine.efficacy_source
                            ELSE excluded.efficacy_source
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        name,
                        normalized_name,
                        category_id,
                        str(item.get("notes", "")).strip(),
                        efficacy,
                        efficacy_source,
                    ),
                )
                medicine_id = int(
                    connection.execute(
                        "SELECT id FROM medicine WHERE normalized_name = ?",
                        (normalized_name,),
                    ).fetchone()["id"]
                )
                for alias in dict.fromkeys([name, *(item.get("aliases") or [])]):
                    alias = str(alias).strip()
                    normalized_alias = normalize_name(alias)
                    connection.execute(
                        """
                        INSERT INTO medicine_alias(medicine_id, alias, normalized_alias, source)
                        VALUES (?, ?, ?, 'catalog')
                        ON CONFLICT(normalized_alias) DO UPDATE SET
                            medicine_id = excluded.medicine_id,
                            alias = excluded.alias
                        WHERE medicine_alias.source = 'catalog'
                        """,
                        (medicine_id, alias, normalized_alias),
                    )

            for category in catalog["categories"]:
                category_id = int(category["id"])
                efficacy = str(category.get("efficacy", "")).strip()
                if not efficacy:
                    continue
                connection.execute(
                    """
                    UPDATE medicine
                    SET efficacy = ?, efficacy_source = 'catalog:category',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE category_id = ? AND TRIM(efficacy) = ''
                    """,
                    (efficacy, category_id),
                )
        return database_stats(connection)
    finally:
        connection.close()


def database_stats(connection_or_path: sqlite3.Connection | Path) -> dict[str, int]:
    owns_connection = not isinstance(connection_or_path, sqlite3.Connection)
    connection = (
        _connect(Path(connection_or_path).expanduser().resolve())
        if owns_connection
        else connection_or_path
    )
    try:
        return {
            "categories": int(connection.execute("SELECT COUNT(*) FROM medicine_category").fetchone()[0]),
            "medicines": int(connection.execute("SELECT COUNT(*) FROM medicine").fetchone()[0]),
            "aliases": int(connection.execute("SELECT COUNT(*) FROM medicine_alias").fetchone()[0]),
        }
    finally:
        if owns_connection:
            connection.close()


def add_medicine(
    db_path: Path,
    name: str,
    category_id: int,
    aliases: Sequence[str] = (),
    notes: str = "",
    source: str = "user",
    efficacy: str = "",
    efficacy_source: str = "",
) -> int:
    """Insert a user-maintained medicine and aliases."""
    normalized_name = normalize_name(name)
    if len(normalized_name) < 2:
        raise ValueError("药名至少需要两个有效字符")
    source = str(source).strip()
    if not source:
        raise ValueError("source must not be empty")
    connection = _connect(Path(db_path).expanduser().resolve())
    try:
        connection.executescript(SCHEMA)
        _ensure_schema_columns(connection)
        if connection.execute(
            "SELECT 1 FROM medicine_category WHERE id = ?", (category_id,)
        ).fetchone() is None:
            raise ValueError(f"类别 ID 不存在：{category_id}")
        with connection:
            connection.execute(
                """
                INSERT INTO medicine(
                    name, normalized_name, category_id, source, notes,
                    efficacy, efficacy_source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_name) DO UPDATE SET
                    name = excluded.name,
                    category_id = excluded.category_id,
                    source = excluded.source,
                    notes = excluded.notes,
                    efficacy = CASE
                        WHEN excluded.efficacy <> '' THEN excluded.efficacy
                        ELSE medicine.efficacy
                    END,
                    efficacy_source = CASE
                        WHEN excluded.efficacy <> '' THEN excluded.efficacy_source
                        ELSE medicine.efficacy_source
                    END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    name.strip(),
                    normalized_name,
                    int(category_id),
                    source,
                    notes.strip(),
                    efficacy.strip(),
                    efficacy_source.strip(),
                ),
            )
            medicine_id = int(
                connection.execute(
                    "SELECT id FROM medicine WHERE normalized_name = ?", (normalized_name,)
                ).fetchone()["id"]
            )
            for alias in dict.fromkeys([name, *aliases]):
                normalized_alias = normalize_name(alias)
                if len(normalized_alias) < 2:
                    continue
                connection.execute(
                    """
                    INSERT INTO medicine_alias(medicine_id, alias, normalized_alias, source)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(normalized_alias) DO UPDATE SET
                        medicine_id = excluded.medicine_id,
                        alias = excluded.alias,
                        source = excluded.source
                    """,
                    (medicine_id, alias.strip(), normalized_alias, source),
                )
        return medicine_id
    finally:
        connection.close()


def _partial_similarity(alias: str, text: str) -> float:
    if not alias or not text:
        return 0.0
    if alias == text:
        return 1.0
    if len(text) <= len(alias) + 2:
        return SequenceMatcher(None, alias, text).ratio()
    window = len(alias)
    scores = [
        SequenceMatcher(None, alias, text[index : index + window]).ratio()
        for index in range(0, len(text) - window + 1)
    ]
    return max(scores, default=0.0)


class MedicineDatabase:
    """Read-mostly catalog with exact-substring and OCR-tolerant fuzzy lookup."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        initialize: bool = True,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.catalog_path = Path(catalog_path).expanduser().resolve()
        if initialize:
            initialize_database(self.db_path, self.catalog_path)
        self.connection = _connect(self.db_path)
        self.aliases = self._load_aliases()
        self._fuzzy_aliases: list[tuple[sqlite3.Row, str, Counter[str]]] = []
        self._fuzzy_aliases_by_char: dict[str, list[int]] = {}
        for row in self.aliases:
            alias = str(row["normalized_alias"])
            if len(alias) < 4:
                continue
            index = len(self._fuzzy_aliases)
            counts = Counter(alias)
            self._fuzzy_aliases.append((row, alias, counts))
            for character in counts:
                self._fuzzy_aliases_by_char.setdefault(character, []).append(index)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MedicineDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _load_aliases(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT
                    m.id AS medicine_id,
                    m.name AS medicine_name,
                    m.category_id,
                    c.name AS category_name,
                    m.efficacy,
                    m.efficacy_source,
                    a.alias,
                    a.normalized_alias
                FROM medicine_alias AS a
                JOIN medicine AS m ON m.id = a.medicine_id
                JOIN medicine_category AS c ON c.id = m.category_id
                ORDER BY LENGTH(a.normalized_alias) DESC, a.id ASC
                """
            )
        )

    @staticmethod
    def _match(row: sqlite3.Row, score: float, match_type: str) -> MedicineMatch:
        return MedicineMatch(
            medicine_id=int(row["medicine_id"]),
            medicine_name=str(row["medicine_name"]),
            category_id=int(row["category_id"]),
            category_name=str(row["category_name"]),
            efficacy=str(row["efficacy"]),
            efficacy_source=str(row["efficacy_source"]),
            matched_alias=str(row["alias"]),
            score=round(float(score), 4),
            match_type=match_type,
        )

    def lookup(
        self,
        text: str,
        lines: Iterable[str] = (),
        fuzzy_threshold: float = 0.76,
    ) -> MedicineMatch | None:
        normalized_text = normalize_name(text)
        normalized_lines = [normalize_name(line) for line in lines]
        normalized_lines = [line for line in normalized_lines if line]

        exact = [
            row
            for row in self.aliases
            if row["normalized_alias"]
            and str(row["normalized_alias"]) in normalized_text
        ]
        if exact:
            row = max(exact, key=lambda item: len(str(item["normalized_alias"])))
            return self._match(row, 1.0, "exact")

        best: tuple[float, int, sqlite3.Row] | None = None
        for line in normalized_lines:
            line_counts = Counter(line)
            candidate_indexes: set[int] = set()
            for character in line_counts:
                candidate_indexes.update(self._fuzzy_aliases_by_char.get(character, ()))
            for index in candidate_indexes:
                row, alias, alias_counts = self._fuzzy_aliases[index]
                overlap = sum(
                    min(count, line_counts.get(character, 0))
                    for character, count in alias_counts.items()
                )
                if len(line) <= len(alias) + 2:
                    upper_bound = (2.0 * overlap) / (len(alias) + len(line))
                else:
                    upper_bound = overlap / len(alias)
                if upper_bound < fuzzy_threshold:
                    continue
                score = _partial_similarity(alias, line)
                candidate = (score, len(alias), row)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is not None and best[0] >= fuzzy_threshold:
            return self._match(best[2], best[0], "fuzzy")
        return None

    def category(self, category_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT id, name, guidance, efficacy "
            "FROM medicine_category WHERE id = ?",
            (int(category_id),),
        ).fetchone()
        if row is None:
            raise KeyError(category_id)
        return dict(row)

    def stats(self) -> dict[str, int]:
        return database_stats(self.connection)

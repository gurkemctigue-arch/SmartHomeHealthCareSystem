# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from medicine_db import (
    DEFAULT_CATALOG_PATH,
    MedicineDatabase,
    add_medicine,
    initialize_database,
)


class MedicineDatabaseInitializationTests(unittest.TestCase):
    def test_category_fallback_does_not_replace_specific_web_efficacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "medicine.db"
            initialize_database(database_path, DEFAULT_CATALOG_PATH)
            connection = sqlite3.connect(database_path)
            try:
                row = connection.execute(
                    "SELECT id FROM medicine WHERE efficacy_source = 'catalog:category' LIMIT 1"
                ).fetchone()
                self.assertIsNotNone(row)
                medicine_id = int(row[0])
                connection.execute(
                    "UPDATE medicine SET efficacy = ?, efficacy_source = ? WHERE id = ?",
                    (
                        "这是从详情页获得的具体适应症。",
                        "https://www.yao86.com/example",
                        medicine_id,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            initialize_database(database_path, DEFAULT_CATALOG_PATH)
            connection = sqlite3.connect(database_path)
            try:
                efficacy, source = connection.execute(
                    "SELECT efficacy, efficacy_source FROM medicine WHERE id = ?",
                    (medicine_id,),
                ).fetchone()
            finally:
                connection.close()

            self.assertEqual(efficacy, "这是从详情页获得的具体适应症。")
            self.assertEqual(source, "https://www.yao86.com/example")

    def test_fuzzy_index_preserves_single_character_ocr_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "medicine.db"
            initialize_database(database_path, DEFAULT_CATALOG_PATH)
            add_medicine(
                database_path,
                "星河清宁颗粒",
                13,
                efficacy="测试用功效。",
                efficacy_source="test",
            )
            with MedicineDatabase(database_path, DEFAULT_CATALOG_PATH, initialize=False) as database:
                match = database.lookup("", lines=("星河青宁颗粒",), fuzzy_threshold=0.76)

            self.assertIsNotNone(match)
            self.assertEqual(match.medicine_name, "星河清宁颗粒")
            self.assertEqual(match.match_type, "fuzzy")
            self.assertGreaterEqual(match.score, 0.76)


if __name__ == "__main__":
    unittest.main()

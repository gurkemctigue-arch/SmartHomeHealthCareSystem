# -*- coding: utf-8 -*-
"""Initialize and maintain the medicine-name SQLite catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

from medicine_db import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DB_PATH,
    MedicineDatabase,
    add_medicine,
    initialize_database,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the 18-category medicine database")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH, help="Seed YAML path")
    parser.add_argument("--add", metavar="NAME", help="Add or update one medicine")
    parser.add_argument("--category", type=int, help="Category ID for --add (0..17)")
    parser.add_argument("--alias", action="append", default=[], help="Alias for --add; repeatable")
    parser.add_argument("--notes", default="", help="Optional notes for --add")
    parser.add_argument("--lookup", metavar="OCR_TEXT", help="Test a database lookup")
    args = parser.parse_args()
    if args.add and args.category is None:
        parser.error("--add requires --category")
    if args.category is not None and not 0 <= args.category <= 17:
        parser.error("--category must be between 0 and 17")
    return args


def main() -> None:
    args = parse_args()
    database_path = args.database.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    stats = initialize_database(database_path, catalog_path)

    if args.add:
        medicine_id = add_medicine(
            database_path,
            args.add,
            args.category,
            aliases=args.alias,
            notes=args.notes,
        )
        print(f"Added/updated medicine ID {medicine_id}: {args.add}")

    with MedicineDatabase(database_path, catalog_path, initialize=False) as database:
        stats = database.stats()
        print(f"Database: {database_path}")
        print(
            f"Categories: {stats['categories']}, medicines: {stats['medicines']}, "
            f"aliases: {stats['aliases']}"
        )
        print("Category table:")
        for category_id in range(18):
            category = database.category(category_id)
            print(f"  {category['id']:2d}  {category['name']}")
        if args.lookup:
            match = database.lookup(args.lookup, lines=args.lookup.splitlines())
            if match is None:
                print("Lookup: no catalog match (category: 其它)")
            else:
                print(
                    f"Lookup: {match.medicine_name} -> {match.category_name} "
                    f"({match.match_type}, score={match.score:.2f})"
                )


if __name__ == "__main__":
    main()

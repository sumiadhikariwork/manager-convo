#!/usr/bin/env python3
"""Create the database schema.

Run this once against a new database. On a serverless deployment set
AUTO_CREATE_TABLES=false so cold starts do not repeat the work, and run this
instead:

    DATABASE_URL=postgresql://... python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect  # noqa: E402

from app.db import DATABASE_URL, engine, init_db  # noqa: E402
from app.models import Base  # noqa: E402


def main() -> int:
    # Never print the URL itself - it carries the password.
    host = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    print(f"Applying schema to {host}")

    before = set(inspect(engine).get_table_names())
    init_db()
    after = set(inspect(engine).get_table_names())

    created = sorted(after - before)
    if created:
        print(f"Created: {', '.join(created)}")
    else:
        print("Schema already up to date.")
    print(f"Tables: {', '.join(sorted(after & set(Base.metadata.tables)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(sqlite_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    return connection


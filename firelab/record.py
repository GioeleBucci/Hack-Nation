"""The shared research record: an append-only, versioned item store in SQLite.

Every write appends a new row; writing an existing id creates version n+1 (e.g. a hypothesis
moving from "proposed" to "rejected"). Nothing is updated in place or deleted, so every
decision can be reconstructed with its full history. SQLite in WAL mode lets several agent
processes read and write concurrently. See docs/record.md.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from firelab import config
from firelab.schemas import ID_PREFIX

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    id         TEXT NOT NULL,
    version    INTEGER NOT NULL,
    payload    TEXT NOT NULL,
    author     TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (kind, id, version)
);
CREATE INDEX IF NOT EXISTS items_kind_id ON items (kind, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Record:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")   # serializes id/version assignment across processes
            yield conn
            if write:
                conn.execute("COMMIT")
        except BaseException:
            if write and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {"kind": r["kind"], "id": r["id"], "version": r["version"], "author": r["author"],
                "created_at": r["created_at"], "payload": json.loads(r["payload"])}

    @staticmethod
    def _next_id(c: sqlite3.Connection, kind: str) -> str:
        prefix = ID_PREFIX.get(kind, kind.upper())
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        numbers = [int(m.group(1)) for (i,) in c.execute("SELECT DISTINCT id FROM items WHERE kind = ?", (kind,))
                   if (m := pattern.match(i))]
        return f"{prefix}{max(numbers, default=0) + 1}"

    def put(self, kind: str, payload: dict, author: str | None = None) -> dict:
        """Append an item (new id, or a new version of an existing id)."""
        payload = dict(payload)
        with self._conn(write=True) as c:
            item_id = payload.get("id") or self._next_id(c, kind)
            payload["id"] = item_id
            (last,) = c.execute("SELECT COALESCE(MAX(version), 0) FROM items WHERE kind = ? AND id = ?",
                                (kind, item_id)).fetchone()
            created = _now()
            c.execute("INSERT INTO items (kind, id, version, payload, author, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                      (kind, item_id, last + 1, json.dumps(payload, default=str), author, created))
        return {"kind": kind, "id": item_id, "version": last + 1, "author": author,
                "created_at": created, "payload": payload}

    def get(self, kind: str | None = None, ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
        """Latest version of each matching item, oldest first."""
        sql = ("SELECT * FROM items i WHERE version = "
               "(SELECT MAX(version) FROM items j WHERE j.kind = i.kind AND j.id = i.id)")
        args: list = []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if ids:
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args.extend(ids)
        sql += " ORDER BY seq"
        with self._conn() as c:
            rows = [self._row(r) for r in c.execute(sql, args)]
        return rows[-limit:] if limit else rows

    def get_one(self, kind: str, item_id: str) -> dict | None:
        items = self.get(kind, [item_id])
        return items[0] if items else None

    def history(self, kind: str, item_id: str) -> list[dict]:
        with self._conn() as c:
            return [self._row(r) for r in c.execute(
                "SELECT * FROM items WHERE kind = ? AND id = ? ORDER BY version", (kind, item_id))]

    def all_versions(self) -> list[dict]:
        with self._conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM items ORDER BY seq")]

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {k: n for k, n in c.execute("SELECT kind, COUNT(DISTINCT id) FROM items GROUP BY kind")}

    def sims_used(self) -> int:
        return sum(int(e["payload"].get("cost_sims", 0)) for e in self.get("experiment"))

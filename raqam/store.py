"""Local offline queue. SQLite is enough (see plan §05).

ponytail: records stored as plaintext in a local .db; field-level encryption +
key management is a Phase-8 concern once this runs on a shared field phone.
Today the guarantee is: the raw image never lands here, only its SHA-256
(chain of custody, plan §06).
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "scans" / "queue.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           INTEGER PRIMARY KEY,
    ts           REAL,
    form         TEXT,
    field        TEXT,
    value        TEXT,          -- '?' marks an unresolved flagged digit
    needs_review INTEGER,
    reviewed     INTEGER DEFAULT 0,
    img_sha256   TEXT,
    cells_json   TEXT,
    synced       INTEGER DEFAULT 0
);
"""


@contextlib.contextmanager
def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def add(form, field, value, needs_review, img_sha256, cells) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO records(ts,form,field,value,needs_review,img_sha256,cells_json)"
            " VALUES(?,?,?,?,?,?,?)",
            (time.time(), form, field, value, int(needs_review), img_sha256,
             json.dumps(cells)),
        )
        return cur.lastrowid


def resolve(record_id: int, value: str) -> None:
    with _conn() as c:
        c.execute("UPDATE records SET value=?, needs_review=0, reviewed=1 WHERE id=?",
                  (value, record_id))


def pending():
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM records WHERE needs_review=1 AND reviewed=0 ORDER BY ts")]


def all_records():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM records ORDER BY ts")]


if __name__ == "__main__":
    import os
    tmp = DB.with_name("queue_test.db")
    globals()["DB"] = tmp
    tmp.unlink(missing_ok=True)
    rid = add("marksheet", "roll_no", "12?45", True, "abc123", [])
    assert len(pending()) == 1
    resolve(rid, "12345")
    assert pending() == []
    assert all_records()[0]["value"] == "12345"
    tmp.unlink(missing_ok=True)
    print("store ok")

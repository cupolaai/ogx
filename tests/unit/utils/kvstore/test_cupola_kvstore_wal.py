# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: the SQLite kvstore does not lock its readers out while it writes.

The model registry and the vector stores' file contents share one kvstore file.
Indexing a document writes every part with its own commit; in SQLite's default
rollback-journal mode each commit locks readers out, and a reader that kept
losing for 5 s failed with "database is locked" — GET /v1/models answered 500
during an index insert (Cupola faz8.0.584, seen live). In WAL mode a reader
never waits for a writer.
"""

import asyncio
import sqlite3

import aiosqlite

from ogx.core.storage.kvstore.config import SqliteKVStoreConfig
from ogx.core.storage.kvstore.sqlite import sqlite as sqlite_mod
from ogx.core.storage.kvstore.sqlite.sqlite import SqliteKVStoreImpl


def journal_mode(path: str) -> str:
    with sqlite3.connect(path) as conn:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]


async def test_a_reader_is_not_locked_out_while_a_writer_commits(tmp_path):
    path = str(tmp_path / "kvstore.db")
    kv = SqliteKVStoreImpl(SqliteKVStoreConfig(db_path=path))
    await kv.initialize()
    await kv.set("registry:a", "1")

    # Another connection holds the write lock a commit takes (an index insert).
    writer = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        writer.execute("BEGIN EXCLUSIVE")
        writer.execute("INSERT INTO kvstore (key, value) VALUES ('registry:b', '2')")
        # Rollback-journal mode: the reader waits 5 s, then "database is locked".
        assert await asyncio.wait_for(kv.get("registry:a"), timeout=2) == "1"
        assert await asyncio.wait_for(kv.values_in_range("registry:", "registry:\xff"), timeout=2) == ["1"]
        assert await asyncio.wait_for(kv.keys_in_range("registry:", "registry:\xff"), timeout=2) == ["registry:a"]
    finally:
        writer.execute("ROLLBACK")
        writer.close()


async def test_a_new_kvstore_file_is_in_wal_mode(tmp_path):
    path = str(tmp_path / "kvstore.db")
    await SqliteKVStoreImpl(SqliteKVStoreConfig(db_path=path)).initialize()
    assert journal_mode(path) == "wal"


async def test_an_existing_rollback_journal_file_turns_wal_and_keeps_its_data(tmp_path):
    path = str(tmp_path / "kvstore.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE kvstore (key TEXT PRIMARY KEY, value TEXT, expiration TIMESTAMP)")
        conn.execute("INSERT INTO kvstore (key, value) VALUES ('registry:old', 'kept')")
    assert journal_mode(path) == "delete"

    kv = SqliteKVStoreImpl(SqliteKVStoreConfig(db_path=path))
    await kv.initialize()
    assert journal_mode(path) == "wal"
    assert await kv.get("registry:old") == "kept"


async def test_every_operation_waits_5_s_and_syncs_normally(tmp_path, monkeypatch):
    real_connect = aiosqlite.connect
    connections: list[list[str]] = []

    def spying_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        statements: list[str] = []
        connections.append(statements)
        execute = conn.execute

        def recording_execute(sql, *a, **kw):
            statements.append(" ".join(sql.split()))
            return execute(sql, *a, **kw)

        conn.execute = recording_execute
        return conn

    kv = SqliteKVStoreImpl(SqliteKVStoreConfig(db_path=str(tmp_path / "kvstore.db")))
    await kv.initialize()
    monkeypatch.setattr(sqlite_mod.aiosqlite, "connect", spying_connect)

    await kv.set("k", "v")
    await kv.get("k")
    await kv.values_in_range("a", "z")
    await kv.keys_in_range("a", "z")
    await kv.delete("k")

    assert len(connections) == 5
    for statements in connections:
        assert statements[:2] == ["PRAGMA busy_timeout=5000", "PRAGMA synchronous=NORMAL"], statements
        assert len(statements) > 2, statements


async def test_an_in_memory_kvstore_still_works():
    kv = SqliteKVStoreImpl(SqliteKVStoreConfig(db_path=":memory:"))
    await kv.initialize()
    await kv.set("k", "v")
    assert await kv.get("k") == "v"
    await kv.shutdown()

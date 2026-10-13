# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: the startup deletion on a real PostgreSQL (DELETE, then VACUUM FULL).

Runs only when CUPOLA_TEST_POSTGRES_HOST is set (user/password/db: ogx/ogx/ogx unless
CUPOLA_TEST_POSTGRES_USER / _PASSWORD / _DB say otherwise).
"""

import asyncio
import os
import time

import asyncpg
import pytest

from ogx.core.routers import inference as router_module
from ogx.core.routers.inference import InferenceRouter
from ogx.core.storage.datatypes import InferenceStoreReference, PostgresSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx_api import OpenAIChatCompletion, OpenAIChatCompletionResponseMessage, OpenAIChoice, OpenAIUserMessageParam

HOST = os.environ.get("CUPOLA_TEST_POSTGRES_HOST")
PG = {
    "host": HOST,
    "port": 5432,
    "user": os.environ.get("CUPOLA_TEST_POSTGRES_USER", "ogx"),
    "password": os.environ.get("CUPOLA_TEST_POSTGRES_PASSWORD", "ogx"),
    "database": os.environ.get("CUPOLA_TEST_POSTGRES_DB", "ogx"),
}
pytestmark = pytest.mark.skipif(not HOST, reason="CUPOLA_TEST_POSTGRES_HOST not set")
DONE = (
    " (postgres: table rewritten by VACUUM FULL; the database server's WAL, replicas and backups keep older copies"
    " under its own retention)"
)


class _Log:
    def __init__(self):
        self.records = []

    def __getattr__(self, level):
        return lambda msg, **kw: self.records.append((level, msg))

    def messages(self, level):
        return [msg for lvl, msg in self.records if lvl == level and "deleted" in msg]


@pytest.fixture
async def pg(monkeypatch):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    conn = await asyncpg.connect(**PG)
    await conn.execute("DROP TABLE IF EXISTS inference_store")
    register_sqlstore_backends(
        {
            "sql_default": PostgresSqlStoreConfig(
                host=PG["host"], port=PG["port"], db=PG["database"], user=PG["user"], password=PG["password"]
            )
        }
    )
    yield conn
    await conn.close()


@pytest.fixture
def log(monkeypatch):
    rec = _Log()
    monkeypatch.setattr(router_module, "logger", rec)
    return rec


async def _store(rows: int) -> InferenceStore:
    store = InferenceStore(InferenceStoreReference(backend="sql_default", table_name="inference_store"), policy=[])
    await store.initialize()
    filler = "x" * 4000
    for i in range(rows):
        completion = OpenAIChatCompletion(
            id=f"chatcmpl-{i}",
            created=1760000000 + i,
            model="m",
            object="chat.completion",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChatCompletionResponseMessage(role="assistant", content=f"answer {i}"),
                    finish_reason="stop",
                )
            ],
        )
        await store._write_chat_completion(completion, [OpenAIUserMessageParam(role="user", content=f"{i} {filler}")])
    return store


async def _size(conn) -> int:
    return await conn.fetchval("SELECT pg_total_relation_size('inference_store')")


async def test_postgres_deletes_and_rewrites_the_table(pg, log):
    store = await _store(1200)
    before = await _size(pg)
    await InferenceRouter(None, store=store).initialize()
    assert await pg.fetchval("SELECT count(*) FROM inference_store") == 0
    assert log.messages("info") == [f"Cupola: inference_store off, deleted 1200 stored chat completions{DONE}"]
    after = await _size(pg)
    assert after <= 32768 and before > 20 * after, (before, after)
    await InferenceRouter(None, store=store).initialize()
    assert log.messages("info")[-1] == f"Cupola: inference_store off, deleted 0 stored chat completions{DONE}"


async def test_postgres_does_not_wait_for_another_sessions_lock(pg, log):
    store = await _store(5)
    holder = await asyncpg.connect(**PG)
    tx = holder.transaction()
    await tx.start()
    await holder.fetchval("SELECT count(*) FROM inference_store")  # ACCESS SHARE until the transaction ends
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(InferenceRouter(None, store=store).initialize(), timeout=15)
    finally:
        await tx.rollback()
        await holder.close()
    took = time.monotonic() - t0
    assert 1.5 < took < 6, took
    [warning] = log.messages("warning")
    assert warning.startswith("Cupola: inference_store off, deleted 5 stored chat completions (postgres: VACUUM FULL")
    assert "did not finish" in warning and "until the next start" in warning
    assert await pg.fetchval("SELECT count(*) FROM inference_store") == 0


async def test_postgres_statement_timeout_is_the_budget_left(pg, log, monkeypatch):
    store = await _store(1)
    monkeypatch.setattr(router_module, "CUPOLA_DELETE_STORED_SECONDS", 3.0)
    seen = []
    base = store.sql_store.sql_store
    execute = base.cupola_execute_autocommit

    async def spy(statements):
        seen.extend(statements)
        return await execute(statements)

    monkeypatch.setattr(base, "cupola_execute_autocommit", spy)
    await InferenceRouter(None, store=store).initialize()
    timeout = int(next(s for s in seen if s.startswith("SET statement_timeout")).split("= ")[1])
    assert 1000 < timeout <= 3000, timeout
    assert log.messages("info") == [f"Cupola: inference_store off, deleted 1 stored chat completions{DONE}"]

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: with CUPOLA_INFERENCE_STORE=off, startup deletes the stored chat completions."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ogx.core.datatypes import User
from ogx.core.routers import inference as router_module
from ogx.core.routers.inference import InferenceRouter
from ogx.core.storage.datatypes import InferenceStoreReference, SqliteSqlStoreConfig, StorageBackendType
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.inference import inference_store as store_module
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx_api import OpenAIChatCompletion, OpenAIChatCompletionResponseMessage, OpenAIChoice, OpenAIUserMessageParam
from ogx_api.internal.sqlstore import ColumnType


class _Log:
    def __init__(self):
        self.records = []

    def __getattr__(self, level):
        return lambda msg, **kw: self.records.append((level, msg, kw))

    def messages(self, level):
        return [msg for lvl, msg, _ in self.records if lvl == level]


@pytest.fixture(autouse=True)
def backends(tmp_path):
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=str(tmp_path / "sql_store.db"))})
    return tmp_path / "sql_store.db"


@pytest.fixture
def log(monkeypatch):
    rec = _Log()
    monkeypatch.setattr(router_module, "logger", rec)
    return rec


async def _store(rows: int, owner: User | None = None) -> InferenceStore:
    store = InferenceStore(InferenceStoreReference(backend="sql_default", table_name="inference_store"), policy=[])
    await store.initialize()
    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user", return_value=owner):
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
            await store._write_chat_completion(completion, [OpenAIUserMessageParam(role="user", content=f"q {i}")])
    return store


async def _count(store: InferenceStore) -> int:
    return len((await store.sql_store.sql_store.fetch_all(table="inference_store")).data)


async def test_off_deletes_every_row_and_logs_the_count(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(1200)  # three pages of 500
    router = InferenceRouter(None, store=store)
    await router.initialize()
    assert await _count(store) == 0
    assert router.store is None
    assert (
        log.messages("info")[-1]
        == "Cupola: inference_store off, deleted 1200 stored chat completions (sqlite: secure_delete on, WAL truncated)"
    )
    assert log.messages("warning") == []


async def test_the_next_start_deletes_none(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(3)
    await InferenceRouter(None, store=store).initialize()
    await InferenceRouter(None, store=store).initialize()
    assert (
        log.messages("info")[-1]
        == "Cupola: inference_store off, deleted 0 stored chat completions (sqlite: secure_delete on, WAL truncated)"
    )


async def test_rows_of_an_authenticated_user_are_deleted_too(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(4, owner=User("someone", {"roles": ["user"]}))
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 0
    assert (
        log.messages("info")[-1]
        == "Cupola: inference_store off, deleted 4 stored chat completions (sqlite: secure_delete on, WAL truncated)"
    )


async def test_other_tables_are_kept(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(2)
    base = store.sql_store.sql_store
    await base.create_table("responses", {"id": ColumnType.STRING})
    await base.insert("responses", {"id": "resp-1"})
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 0
    assert [r["id"] for r in (await base.fetch_all(table="responses")).data] == ["resp-1"]


@pytest.mark.parametrize("value", [None, "", "on", "offline"])
async def test_store_on_deletes_nothing(monkeypatch, log, value):
    if value is None:
        monkeypatch.delenv("CUPOLA_INFERENCE_STORE", raising=False)
    else:
        monkeypatch.setenv("CUPOLA_INFERENCE_STORE", value)
    store = await _store(3)
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 3
    assert not [m for m in log.messages("info") if "deleted" in m]


def _slow_deletes(monkeypatch, store: InferenceStore, seconds: float) -> None:
    base = store.sql_store.sql_store
    delete_many = base.delete_many

    async def slow(operations):
        await delete_many(operations)
        await asyncio.sleep(seconds)

    monkeypatch.setattr(base, "delete_many", slow)


async def test_time_limit_stops_between_pages_and_says_so(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    monkeypatch.setattr(router_module, "CUPOLA_DELETE_STORED_SECONDS", 1.0)
    store = await _store(1200)
    _slow_deletes(monkeypatch, store, 1.1)  # page 1 is deleted, page 2 starts after the limit
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 700
    assert log.messages("warning") == [
        "Cupola: inference_store off, deleted 500 stored chat completions, more remain"
        " (time limit 1 s), continues at the next start (sqlite: secure_delete on, WAL truncated)"
    ]
    assert not [m for m in log.messages("info") if "deleted" in m]


async def test_store_level_count_and_remaining(monkeypatch):
    store = await _store(501)
    assert await store.cupola_delete_stored_completions(deadline=float("-inf")) == (0, True)
    assert await _count(store) == 501
    assert await store.cupola_delete_stored_completions(deadline=float("inf")) == (501, False)
    assert await _count(store) == 0
    assert await store.cupola_delete_stored_completions(deadline=float("inf")) == (0, False)
    assert await store.cupola_delete_stored_completions(deadline=float("-inf")) == (0, False)


async def test_a_failure_does_not_stop_startup(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(2)

    async def broken(deadline):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(store, "cupola_delete_stored_completions", broken)
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 2
    [(level, msg, kw)] = [r for r in log.records if r[0] == "warning"]
    assert msg.startswith("Cupola: could not delete the stored chat completions")
    assert kw == {"error_type": "RuntimeError", "error": "disk I/O error"}


async def test_the_deletion_runs_once_per_router(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(1)
    router = InferenceRouter(None, store=store)
    await router.initialize()
    await router.initialize()
    assert [m for m in log.messages("info") if "deleted" in m] == [
        "Cupola: inference_store off, deleted 1 stored chat completions (sqlite: secure_delete on, WAL truncated)"
    ]


async def test_sqlite_wal_is_truncated_after_the_deletion(monkeypatch, log, backends):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(300)
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 0
    assert os.path.getsize(str(backends) + "-wal") == 0


def _fake_execute(monkeypatch, store: InferenceStore, result=None, error=None):
    calls = []

    async def execute(statements):
        calls.append(list(statements))
        if error:
            raise error
        return result

    monkeypatch.setattr(store.sql_store.sql_store, "cupola_execute_autocommit", execute)
    return calls


@pytest.mark.parametrize(
    "rows, said",
    [
        ([(0,), (0, 0, 0)], "sqlite: secure_delete is off in this SQLite build, deleted content may remain"),
        ([(1,), (1, 7, 3)], "sqlite: the WAL is busy and was not truncated, deleted content may remain in it"),
        ([None, None], "sqlite: secure_delete is off in this SQLite build"),
    ],
)
async def test_sqlite_without_the_guarantee_is_a_warning(monkeypatch, log, rows, said):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(2)
    calls = _fake_execute(monkeypatch, store, result=rows)
    await InferenceRouter(None, store=store).initialize()
    assert calls == [["PRAGMA secure_delete", "PRAGMA wal_checkpoint(TRUNCATE)"]]
    [warning] = log.messages("warning")
    assert warning.startswith("Cupola: inference_store off, deleted 2 stored chat completions (")
    assert said in warning
    assert not [m for m in log.messages("info") if "deleted" in m]


async def test_sqlite_scrub_failure_is_a_warning_and_startup_goes_on(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(2)
    _fake_execute(monkeypatch, store, error=RuntimeError("database is locked"))
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 0
    assert log.messages("warning") == [
        "Cupola: inference_store off, deleted 2 stored chat completions (sqlite: the WAL could not be truncated"
        " (RuntimeError), deleted content may remain in it until the next start)"
    ]


def _as_backend(monkeypatch, kind):
    monkeypatch.setitem(store_module._SQLSTORE_BACKENDS, "sql_default", SimpleNamespace(type=kind))


async def test_unknown_backend_deletes_and_says_there_is_no_guarantee(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(3)
    calls = _fake_execute(monkeypatch, store, result=[])
    _as_backend(monkeypatch, "sql_other")
    await InferenceRouter(None, store=store).initialize()
    assert await _count(store) == 0
    assert calls == []
    assert log.messages("warning") == [
        "Cupola: inference_store off, deleted 3 stored chat completions (no residue guarantee for storage"
        " backend 'sql_other': deleted content may remain on its disk)"
    ]


async def test_postgres_runs_vacuum_full_within_the_budget(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(3)
    calls = _fake_execute(monkeypatch, store, result=[None, None, None])
    _as_backend(monkeypatch, StorageBackendType.SQL_POSTGRES)
    await InferenceRouter(None, store=store).initialize()
    [[lock, timeout, vacuum]] = calls
    assert lock == "SET lock_timeout = 2000"
    assert timeout.startswith("SET statement_timeout = ")
    assert 8000 < int(timeout.split("= ")[1]) <= 10000
    assert vacuum == 'VACUUM FULL "inference_store"'
    assert log.messages("info")[-1] == (
        "Cupola: inference_store off, deleted 3 stored chat completions (postgres: table rewritten by VACUUM FULL;"
        " the database server's WAL, replicas and backups keep older copies under its own retention)"
    )
    assert log.messages("warning") == []


async def test_postgres_vacuum_failure_is_a_warning(monkeypatch, log):
    monkeypatch.setenv("CUPOLA_INFERENCE_STORE", "off")
    store = await _store(1)

    class LockNotAvailableError(Exception):
        pass

    _fake_execute(monkeypatch, store, error=LockNotAvailableError("canceling statement due to lock timeout"))
    _as_backend(monkeypatch, StorageBackendType.SQL_POSTGRES)
    await InferenceRouter(None, store=store).initialize()
    assert log.messages("warning") == [
        "Cupola: inference_store off, deleted 1 stored chat completions (postgres: VACUUM FULL did not finish"
        " (LockNotAvailableError), deleted content stays in the table files until the next start)"
    ]


@pytest.mark.parametrize("deadline_from_now", [float("-inf"), -0.5])
async def test_postgres_without_time_left_skips_vacuum(monkeypatch, deadline_from_now):
    store = await _store(1)
    calls = _fake_execute(monkeypatch, store, result=[None, None, None])
    _as_backend(monkeypatch, StorageBackendType.SQL_POSTGRES)
    assert await store.cupola_scrub(deadline=router_module.time.monotonic() + deadline_from_now) == (
        "postgres: no time left for VACUUM FULL in this start, deleted content stays in the table files until the"
        " next start",
        False,
    )
    assert calls == []


async def test_postgres_table_name_is_quoted(monkeypatch):
    store = await _store(0)
    store.reference = store.reference.model_copy(update={"table_name": 'odd"name'})
    calls = _fake_execute(monkeypatch, store, result=[None, None, None])
    _as_backend(monkeypatch, StorageBackendType.SQL_POSTGRES)
    await store.cupola_scrub(deadline=router_module.time.monotonic() + 5)
    assert calls[0][2] == 'VACUUM FULL "odd""name"'


async def test_postgres_unbounded_deadline_is_capped(monkeypatch):
    store = await _store(0)
    calls = _fake_execute(monkeypatch, store, result=[None, None, None])
    _as_backend(monkeypatch, StorageBackendType.SQL_POSTGRES)
    assert (await store.cupola_scrub(deadline=float("inf")))[1] is True
    assert calls[0][1] == "SET statement_timeout = 3600000"

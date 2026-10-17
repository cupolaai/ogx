# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# CUPOLA (faz8.0.593) — a store's record is changed by three paths: an attach adds the
# file (under the store's lock), an update changes name / metadata, a delete removes a
# file. The update and the delete took their copy of the record, awaited the storage and
# put the copy back, so a file attached meanwhile was lost: Cupola uploads documents one
# after another and updates the store's metadata on each, and OGX lost half of a store's
# files while their parts stayed searchable (seen live, 10 of 20).

import asyncio
import json
import random
from unittest.mock import AsyncMock, MagicMock

from ogx.providers.utils.memory.openai_vector_store_mixin import OPENAI_VECTOR_STORES_PREFIX, OpenAIVectorStoreMixin
from ogx_api import Chunk, ChunkMetadata, VectorStoreChunkingStrategyAuto
from ogx_api.vector_io.models import OpenAIAttachFileRequest, OpenAIUpdateVectorStoreRequest

STORE = "vs_lock"


class SlowKV:
    """An in-memory kvstore whose writes take a moment, as a real disk does."""

    def __init__(self, delay: float):
        self.delay = delay
        self.data: dict[str, str] = {}

    async def set(self, key, value, expiration=None):
        await asyncio.sleep(self.delay * random.random())
        self.data[key] = value

    async def get(self, key):
        await asyncio.sleep(self.delay * random.random())
        return self.data.get(key)

    async def delete(self, key):
        await asyncio.sleep(self.delay * random.random())
        self.data.pop(key, None)

    async def values_in_range(self, start_key, end_key):
        return [v for k, v in self.data.items() if start_key <= k < end_key]

    async def keys_in_range(self, start_key, end_key):
        return [k for k in self.data if start_key <= k < end_key]


class Mixin(OpenAIVectorStoreMixin):
    async def register_vector_store(self, vector_store):
        pass

    async def unregister_vector_store(self, vector_store_id):
        pass

    async def insert_chunks(self, request):
        pass

    async def query_chunks(self, request):
        pass

    async def delete_chunks(self, request):
        pass


def make_mixin(delay: float = 0.004) -> Mixin:
    files = AsyncMock()
    files.openai_retrieve_file = AsyncMock(return_value=MagicMock(filename="doc.txt"))
    processor = AsyncMock()
    processor.process_file = AsyncMock(
        return_value=MagicMock(
            chunks=[
                Chunk(content="a part", chunk_id="c1", metadata={}, chunk_metadata=ChunkMetadata(document_id="d1"))
            ],
            metadata={},
        )
    )
    inference = AsyncMock()
    inference.openai_embeddings = AsyncMock(return_value=MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])]))
    m = Mixin(inference_api=inference, files_api=files, kvstore=SlowKV(delay), file_processor_api=processor)
    m.openai_vector_stores[STORE] = {
        "id": STORE,
        "object": "vector_store",
        "created_at": 0,
        "name": "store",
        "usage_bytes": 0,
        "status": "completed",
        "file_ids": [],
        "file_counts": {"total": 0, "completed": 0, "cancelled": 0, "failed": 0, "in_progress": 0},
        "metadata": {"embedding_model": "m", "embedding_dimension": "3"},
    }
    return m


async def attach(m: Mixin, file_id: str):
    r = await m.openai_attach_file_to_vector_store(
        vector_store_id=STORE,
        request=OpenAIAttachFileRequest(file_id=file_id, chunking_strategy=VectorStoreChunkingStrategyAuto()),
    )
    assert r.status == "completed", r.last_error


async def update(m: Mixin, n: int):
    await asyncio.sleep(0.002 * random.random())
    await m.openai_update_vector_store(
        STORE,
        OpenAIUpdateVectorStoreRequest(metadata={"embedding_model": "m", "embedding_dimension": "3", "n": str(n)}),
    )


def persisted(m: Mixin) -> dict:
    return json.loads(m.kvstore.data[f"{OPENAI_VECTOR_STORES_PREFIX}{STORE}"])


async def test_back_to_back_attaches_with_metadata_updates_keep_every_file():
    random.seed(593)
    for _ in range(5):
        m = make_mixin()
        files = [f"file-{i}" for i in range(20)]
        await asyncio.gather(*[attach(m, f) for f in files], *[update(m, n) for n in range(20)])
        rec = m.openai_vector_stores[STORE]
        assert sorted(rec["file_ids"]) == sorted(files), f"lost {sorted(set(files) - set(rec['file_ids']))}"
        assert rec["file_counts"]["total"] == 20 and rec["file_counts"]["completed"] == 20
        # what is on disk agrees with memory, so a restart keeps every file too
        assert sorted(persisted(m)["file_ids"]) == sorted(files)
        assert persisted(m)["file_counts"]["total"] == 20


async def test_update_keeps_its_own_changes():
    m = make_mixin()
    await attach(m, "file-a")
    r = await m.openai_update_vector_store(
        STORE, OpenAIUpdateVectorStoreRequest(name="renamed", metadata={"embedding_model": "m", "k": "v"})
    )
    assert r.name == "renamed" and r.metadata["k"] == "v"
    assert m.openai_vector_stores[STORE]["file_ids"] == ["file-a"]
    assert persisted(m)["name"] == "renamed" and persisted(m)["file_ids"] == ["file-a"]


async def test_delete_while_attaching_keeps_the_new_file():
    random.seed(594)
    for _ in range(5):
        m = make_mixin()
        await attach(m, "file-old")
        await attach(m, "file-keep")
        await asyncio.gather(
            m.openai_delete_vector_store_file(vector_store_id=STORE, file_id="file-old"),
            *[attach(m, f"file-new-{i}") for i in range(10)],
        )
        want = ["file-keep"] + [f"file-new-{i}" for i in range(10)]
        rec = m.openai_vector_stores[STORE]
        assert sorted(rec["file_ids"]) == sorted(want), f"got {sorted(rec['file_ids'])}"
        assert rec["file_counts"]["total"] == 11 and rec["file_counts"]["completed"] == 11
        assert sorted(persisted(m)["file_ids"]) == sorted(want)


async def test_delete_does_not_change_a_record_another_reader_holds():
    # the delete works on copies: a record handed out before the delete stays as it was
    m = make_mixin(delay=0)
    await attach(m, "file-a")
    await attach(m, "file-b")
    before = m.openai_vector_stores[STORE]
    await m.openai_delete_vector_store_file(vector_store_id=STORE, file_id="file-a")
    assert before["file_ids"] == ["file-a", "file-b"] and before["file_counts"]["total"] == 2
    assert m.openai_vector_stores[STORE]["file_ids"] == ["file-b"]

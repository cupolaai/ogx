# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a Milvus vector store created before a restart is found again.

After a pod restart the provider cache is cold and the VECTOR_DBS_PREFIX key
was never persisted, so the lookup rebuilds the VectorStore from the
persisted OpenAI metadata, with the live client.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.remote.vector_io.milvus.milvus import MilvusVectorIOAdapter
from ogx_api import VectorStoreNotFoundError


def cold_adapter(openai_vector_stores):
    adapter = MilvusVectorIOAdapter.__new__(MilvusVectorIOAdapter)
    adapter.cache = {}
    adapter.kvstore = AsyncMock()
    adapter.kvstore.get.return_value = None  # the key upstream never writes
    adapter.openai_vector_stores = openai_vector_stores
    adapter.config = None
    adapter.client = MagicMock(name="live-client")
    adapter.inference_api = MagicMock(name="inference")
    return adapter


async def test_a_store_from_before_the_restart_is_rebuilt_from_its_metadata():
    adapter = cold_adapter(
        {
            "vs_1": {
                "name": "docs",
                "metadata": {
                    "embedding_model": "sentence-transformers/intfloat/multilingual-e5-large",
                    "embedding_dimension": "1024",
                    "provider_id": "milvus",
                    "provider_vector_store_id": "vs_1",
                },
            }
        }
    )
    index = await adapter._get_and_cache_vector_store_index("vs_1")

    store = index.vector_store
    assert store.identifier == "vs_1"
    assert store.provider_id == "milvus"
    assert store.provider_resource_id == "vs_1"
    assert store.embedding_model == "sentence-transformers/intfloat/multilingual-e5-large"
    assert store.embedding_dimension == 1024
    assert store.vector_store_name == "docs"
    assert index.index.client is adapter.client  # the live client, not one from boot
    assert adapter.cache["vs_1"] is index


async def test_defaults_when_the_metadata_names_no_provider():
    adapter = cold_adapter({"vs_2": {"metadata": {"embedding_model": "m", "embedding_dimension": 384}}})
    store = (await adapter._get_and_cache_vector_store_index("vs_2")).vector_store
    assert store.provider_id == "milvus"
    assert store.provider_resource_id == "vs_2"


@pytest.mark.parametrize(
    "stores",
    [
        {},
        {"vs_3": {"metadata": {}}},
        {"vs_3": {"metadata": {"embedding_model": "m"}}},
        {"vs_3": {"metadata": {"embedding_dimension": 384}}},
    ],
)
async def test_a_store_without_its_embedding_metadata_is_still_not_found(stores):
    adapter = cold_adapter(stores)
    with pytest.raises(VectorStoreNotFoundError):
        await adapter._get_and_cache_vector_store_index("vs_3")

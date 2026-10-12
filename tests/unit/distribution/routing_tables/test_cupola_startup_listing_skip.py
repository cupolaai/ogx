# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a provider pinned to its models is not listed at startup.

A pinned provider (refresh_models false + allowed_models) is not listed, not
at startup and not later, and counts as listed; every other provider is
listed as upstream.
"""

import types

from ogx.core.routing_tables.models import ModelsRoutingTable


class Provider:
    def __init__(self, refresh, allowed, has_config=True):
        self.refresh = refresh
        self.calls = 0
        if has_config:
            self.config = types.SimpleNamespace(allowed_models=allowed, refresh_models=refresh)

    async def should_refresh_models(self):
        return self.refresh

    async def list_models(self):
        self.calls += 1
        return []


async def test_only_pinned_providers_are_not_listed():
    providers = {
        "openai-ext": Provider(False, ["gpt-x"]),
        "vllm-local": Provider(False, None),
        "openai-refreshing": Provider(True, ["gpt-y"]),
        "inline": Provider(False, None, has_config=False),
    }
    table = ModelsRoutingTable.__new__(ModelsRoutingTable)
    table.impls_by_provider_id = providers
    table.listed_providers = set()
    registered = []

    async def update_registered_models(provider_id, models):
        registered.append(provider_id)

    table.update_registered_models = update_registered_models

    await table.refresh()  # startup
    await table.refresh()  # a later periodic refresh

    assert providers["openai-ext"].calls == 0
    assert "openai-ext" in table.listed_providers
    assert providers["vllm-local"].calls == 1 and "vllm-local" in registered
    assert providers["openai-refreshing"].calls == 2
    assert providers["inline"].calls == 1

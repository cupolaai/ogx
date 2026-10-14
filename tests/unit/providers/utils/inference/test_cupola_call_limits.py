# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: the call limits for external providers (decision 2026-10-14).

connect 5 s, one retry, first response 180 s, 60 s between chunks, no total
limit on a stream (a long answer is never cut), 110 s without a stream. The
timing tests use short limits; the values are pinned separately.
"""

import asyncio
from functools import partial
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.remote.inference.anthropic.anthropic import AnthropicInferenceAdapter
from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
from ogx.providers.remote.inference.openai.openai import OpenAIInferenceAdapter
from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from ogx.providers.utils.inference import cupola_limits
from ogx.providers.utils.inference.cupola_limits import call_with_limits
from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import OpenAIChatCompletionRequestWithExtraBody
from ogx_api.common.provider_errors import ProviderTimeoutError
from tests.unit.providers.utils.inference.openai_mixin_helpers import OpenAIMixinImpl

FAST = {"first_response": 0.3, "idle": 0.2, "non_stream": 0.3}


class FakeStream:
    """An SDK-like stream: chunks after the given delays; records close()."""

    def __init__(self, delays):
        self.delays, self.closed = list(delays), False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, d in enumerate(self.delays):
            await asyncio.sleep(d)
            yield f"c{i}"

    async def close(self):
        self.closed = True


async def collect(stream_delays, **limits):
    s = FakeStream(stream_delays)

    async def make():
        return s

    out = await call_with_limits(make, True, **(FAST | limits))
    return s, [c async for c in out]


def test_the_values_are_the_decided_ones():
    assert cupola_limits.CONNECT_TIMEOUT == 5.0
    assert cupola_limits.MAX_RETRIES == 1
    assert cupola_limits.FIRST_RESPONSE_TIMEOUT == 180.0
    assert cupola_limits.IDLE_TIMEOUT == 60.0
    assert cupola_limits.NON_STREAM_TIMEOUT == 110.0
    p = cupola_limits.client_params()
    assert p["max_retries"] == 1
    assert p["timeout"].connect == 5.0 and p["timeout"].read == 180.0


async def test_a_normal_stream_passes_whole():
    s, chunks = await collect([0, 0.01, 0.01])
    assert chunks == ["c0", "c1", "c2"] and s.closed


async def test_a_long_answer_that_keeps_coming_is_not_cut():
    # 8 chunks 0.1 s apart: 0.8 s in all, far over the first-response limit (0.3 s),
    # every gap under the idle limit (0.2 s).
    _, chunks = await collect([0.05] + [0.1] * 7)
    assert len(chunks) == 8


async def test_no_first_response_in_time():
    s = FakeStream([5])

    async def make():
        return s

    with pytest.raises(ProviderTimeoutError) as e:
        await call_with_limits(make, True, **FAST)
    assert e.value.phase == "first_response" and s.closed


async def test_a_slow_connect_counts_toward_the_first_response():
    async def make():
        await asyncio.sleep(5)

    with pytest.raises(ProviderTimeoutError) as e:
        await call_with_limits(make, True, **FAST)
    assert e.value.phase == "first_response"


async def test_silence_after_the_first_chunk():
    s = FakeStream([0, 0.01, 5])

    async def make():
        return s

    out = await call_with_limits(make, True, **FAST)
    got = []
    with pytest.raises(ProviderTimeoutError) as e:
        async for c in out:
            got.append(c)
    assert got == ["c0", "c1"] and e.value.phase == "idle" and s.closed


async def test_an_empty_stream():
    _, chunks = await collect([])
    assert chunks == []


async def test_without_a_stream():
    async def quick():
        return "answer"

    async def slow():
        await asyncio.sleep(5)

    assert await call_with_limits(quick, False, **FAST) == "answer"
    with pytest.raises(ProviderTimeoutError) as e:
        await call_with_limits(slow, False, **FAST)
    assert e.value.phase == "response"


def test_only_the_external_providers_are_limited():
    assert OpenAIInferenceAdapter.model_fields["cupola_call_limits"].default is True
    assert AnthropicInferenceAdapter.model_fields["cupola_call_limits"].default is True
    assert BedrockInferenceAdapter.model_fields["cupola_call_limits"].default is True
    assert VLLMInferenceAdapter.model_fields["cupola_call_limits"].default is False


class LimitedImpl(OpenAIMixinImpl):
    cupola_call_limits: bool = True


def test_a_limited_provider_gets_one_retry_and_connect_5s():
    c = LimitedImpl(config=RemoteInferenceProviderConfig()).client
    assert c.max_retries == 1
    assert c.timeout.connect == 5.0 and c.timeout.read == 180.0


def test_an_unlimited_provider_keeps_the_sdk_defaults():
    c = OpenAIMixinImpl(config=RemoteInferenceProviderConfig()).client
    assert c.max_retries == 2
    assert c.timeout.read == 600.0


async def test_the_mixin_applies_the_limits_to_a_chat_call(monkeypatch):
    monkeypatch.setattr(cupola_limits, "call_with_limits", partial(cupola_limits.call_with_limits, **FAST))
    impl = LimitedImpl(config=RemoteInferenceProviderConfig())
    impl.model_store = MagicMock(has_model=AsyncMock(return_value=False))

    async def silent(**_):
        await asyncio.sleep(5)

    client = MagicMock()
    client.chat.completions.create = silent
    monkeypatch.setattr(type(impl), "client", property(lambda self: client))
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    with pytest.raises(ProviderTimeoutError):
        await impl.openai_chat_completion(params)


async def test_the_mixin_leaves_an_unlimited_provider_alone(monkeypatch):
    called = {}

    async def guarded(*a, **k):
        called["yes"] = True

    monkeypatch.setattr(cupola_limits, "call_with_limits", guarded)
    impl = OpenAIMixinImpl(config=RemoteInferenceProviderConfig())
    impl.model_store = MagicMock(has_model=AsyncMock(return_value=False))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock(id="x"))
    monkeypatch.setattr(type(impl), "client", property(lambda self: client))
    params = OpenAIChatCompletionRequestWithExtraBody(model="m", messages=[{"role": "user", "content": "hi"}])
    await impl.openai_chat_completion(params)
    assert "yes" not in called

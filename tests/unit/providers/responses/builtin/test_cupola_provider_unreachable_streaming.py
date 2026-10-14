# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a Responses turn whose provider is unreachable or silent fails with that code.

Studio and Cupola AI Chat use the Responses API; before 2026-10-14 such a turn
failed as server_error with the generic text.
"""

import asyncio

import httpx
from openai import APIConnectionError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta

from ogx.providers.utils.inference.cupola_limits import call_with_limits
from ogx_api.common.provider_errors import TIMEOUT_MESSAGE, UNREACHABLE_MESSAGE, ProviderTimeoutError

MODEL = "openai-or/gemma"
REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


async def failed_event(impl):
    result = await impl.create_openai_response(input="hi", model=MODEL, stream=True, store=False)
    events = [e async for e in result]
    assert events[-1].type == "response.failed"
    return events[-1].response.error


async def test_an_unreachable_provider(openai_responses_impl, mock_inference_api):
    exc = APIConnectionError(request=REQ)
    exc.__cause__ = httpx.ConnectError("All connection attempts failed")
    mock_inference_api.openai_chat_completion.side_effect = exc
    error = await failed_event(openai_responses_impl)
    assert error.code == "provider_unreachable"
    assert error.message == UNREACHABLE_MESSAGE
    assert error.provider_status == 502
    assert error.provider_detail == "All connection attempts failed"


async def test_a_silent_provider(openai_responses_impl, mock_inference_api):
    mock_inference_api.openai_chat_completion.side_effect = ProviderTimeoutError("first_response", 180)
    error = await failed_event(openai_responses_impl)
    assert error.code == "provider_timeout"
    assert error.message == TIMEOUT_MESSAGE
    assert error.provider_status == 504


async def test_a_provider_that_goes_silent_mid_stream(openai_responses_impl, mock_inference_api):
    # The real limit (call_with_limits, idle 0.2 s here) cuts a stream that goes
    # quiet after its first chunk: the text so far was streamed, then the turn
    # fails as provider_timeout.
    async def provider_stream():
        yield ChatCompletionChunk(
            id="c1",
            choices=[Choice(index=0, delta=ChoiceDelta(content="Hel"), finish_reason=None)],
            created=1,
            model=MODEL,
            object="chat.completion.chunk",
        )
        await asyncio.sleep(5)

    async def limited(*_a, **_k):
        async def make():
            return provider_stream()

        return await call_with_limits(make, True, first_response=1, idle=0.2)

    mock_inference_api.openai_chat_completion.side_effect = limited
    result = await openai_responses_impl.create_openai_response(input="hi", model=MODEL, stream=True, store=False)
    events = [e async for e in result]
    assert [e.delta for e in events if e.type == "response.output_text.delta"] == ["Hel"]
    assert events[-1].type == "response.failed"
    error = events[-1].response.error
    assert error.code == "provider_timeout"
    assert error.message == TIMEOUT_MESSAGE
    assert error.provider_status == 504
    assert error.provider_detail == "The model provider sent nothing for 0.2 s (idle)."

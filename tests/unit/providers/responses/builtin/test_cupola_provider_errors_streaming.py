# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a Responses call that fails on a provider error keeps what the provider said.

Driven through the real orchestrator with the inference API raising the
provider SDK's errors (OpenRouter's live 429 body, 2026-10-12).
"""

import httpx
import pytest
from openai import APIError, AuthenticationError, RateLimitError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta

from ogx_api import InternalServerError
from ogx_api.common.provider_errors import ProviderRequestError

MODEL = "openai-or/gemma"
REQ = httpx.Request("POST", "http://provider.example/v1/chat/completions")
BODY_429 = {
    "error": {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {"raw": "m:free is temporarily rate-limited upstream.", "provider_name": "Google AI Studio"},
    }
}


def rate_limited():
    return RateLimitError("x", response=httpx.Response(429, request=REQ), body=BODY_429)


async def failed_event(impl):
    result = await impl.create_openai_response(input="hi", model=MODEL, stream=True, store=False)
    events = [e async for e in result]
    assert events[-1].type == "response.failed"
    return events[-1].response.error


async def test_a_streamed_turn_keeps_the_providers_status_and_words(openai_responses_impl, mock_inference_api):
    mock_inference_api.openai_chat_completion.side_effect = rate_limited()
    error = await failed_event(openai_responses_impl)
    assert error.code == "rate_limit_exceeded"
    assert error.message == "Provider returned error"
    assert error.provider_status == 429
    assert error.provider == "Google AI Studio"
    assert error.provider_detail == "m:free is temporarily rate-limited upstream."


async def test_an_error_inside_the_providers_stream_is_a_provider_error(openai_responses_impl, mock_inference_api):
    async def stream_then_fail():
        yield ChatCompletionChunk(
            id="c1",
            choices=[Choice(index=0, delta=ChoiceDelta(content="Hel"), finish_reason=None)],
            created=1,
            model=MODEL,
            object="chat.completion.chunk",
        )
        raise APIError("Provider returned error", REQ, body=BODY_429["error"])

    mock_inference_api.openai_chat_completion.side_effect = lambda *a, **k: stream_then_fail()
    error = await failed_event(openai_responses_impl)
    assert error.provider_status == 429 and error.code == "rate_limit_exceeded"


async def test_a_401_keeps_its_status(openai_responses_impl, mock_inference_api):
    mock_inference_api.openai_chat_completion.side_effect = AuthenticationError(
        "x", response=httpx.Response(401, request=REQ), body={"error": {"message": "User not found.", "code": 401}}
    )
    error = await failed_event(openai_responses_impl)
    assert error.provider_status == 401 and error.provider_detail == "User not found."
    assert error.code == "invalid_prompt"  # as upstream maps a 4xx


async def test_a_non_provider_failure_has_no_provider_fields(openai_responses_impl, mock_inference_api):
    mock_inference_api.openai_chat_completion.side_effect = RuntimeError("boom")
    error = await failed_event(openai_responses_impl)
    assert error.code == "server_error"
    assert error.provider_status is None and error.provider is None and error.provider_detail is None
    assert error.model_dump() == {"code": "server_error", "message": error.message}


async def test_without_a_stream_the_provider_error_is_raised_with_its_fields(openai_responses_impl, mock_inference_api):
    mock_inference_api.openai_chat_completion.side_effect = rate_limited()
    with pytest.raises(ProviderRequestError) as exc_info:
        await openai_responses_impl.create_openai_response(input="hi", model=MODEL, stream=False, store=False)
    assert exc_info.value.error["provider_status"] == 429
    assert exc_info.value.error["code"] == "rate_limit_exceeded"


async def test_without_a_stream_other_failures_are_raised_as_before(openai_responses_impl, mock_inference_api):
    mock_inference_api.openai_chat_completion.side_effect = RuntimeError("boom")
    with pytest.raises(InternalServerError):
        await openai_responses_impl.create_openai_response(input="hi", model=MODEL, stream=False, store=False)

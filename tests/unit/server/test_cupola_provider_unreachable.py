# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a provider that cannot be reached or does not answer in time is a provider error.

Before 2026-10-14 a connection error had no mapping and the client got the
generic 500 ("Internal server error: An unexpected error occurred.", seen live
with an unreachable OpenRouter). Now: 502 provider_unreachable / 504
provider_timeout, with the cause as provider_detail, on every path.
"""

import asyncio
import json
import socket

import httpx
import pytest
from fastapi import FastAPI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from starlette.requests import Request

from ogx.core.server.server import global_exception_handler
from ogx.providers.utils.inference.cupola_limits import call_with_limits
from ogx_api.common.provider_errors import (
    TIMEOUT_MESSAGE,
    UNREACHABLE_MESSAGE,
    ProviderTimeoutError,
    provider_error_fields,
    provider_error_response,
)
from ogx_api.inference.fastapi_routes import _sse_generator, create_router

REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def chained(outer: BaseException, cause: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


def dns_failure():
    return chained(
        APIConnectionError(request=REQ),
        chained(
            httpx.ConnectError("[Errno -3] Temporary failure in name resolution"),
            socket.gaierror(-3, "Temporary failure in name resolution"),
        ),
    )


def refused():
    return chained(APIConnectionError(request=REQ), httpx.ConnectError("All connection attempts failed"))


def connect_timeout():
    return chained(APITimeoutError(request=REQ), httpx.ConnectTimeout("timed out"))


def read_timeout():
    return chained(APITimeoutError(request=REQ), httpx.ReadTimeout("timed out"))


@pytest.mark.parametrize(
    "exc", [dns_failure(), refused(), connect_timeout()], ids=["dns", "refused", "connect-timeout"]
)
def test_an_unreachable_provider(exc):
    pf = provider_error_fields(exc)
    assert pf["code"] == "provider_unreachable" and pf["provider_status"] == 502
    assert pf["message"] == UNREACHABLE_MESSAGE
    status, body = provider_error_response(exc)
    assert status == 502 and body["error"]["code"] == "provider_unreachable"


def test_the_cause_is_the_detail():
    # the innermost message: the resolver's own words
    assert provider_error_fields(dns_failure())["provider_detail"] == "[Errno -3] Temporary failure in name resolution"
    assert provider_error_fields(refused())["provider_detail"] == "All connection attempts failed"


@pytest.mark.parametrize(
    "exc",
    [read_timeout(), ProviderTimeoutError("first_response", 180), ProviderTimeoutError("idle", 60)],
    ids=["read-timeout", "first-response", "idle"],
)
def test_a_provider_that_does_not_answer_in_time(exc):
    pf = provider_error_fields(exc)
    assert pf["code"] == "provider_timeout" and pf["provider_status"] == 504
    assert pf["message"] == TIMEOUT_MESSAGE
    assert provider_error_response(exc)[0] == 504


def test_the_limit_is_named_in_the_detail():
    assert provider_error_fields(ProviderTimeoutError("idle", 60))["provider_detail"] == (
        "The model provider sent nothing for 60 s (idle)."
    )


def test_a_rate_limit_is_unchanged():
    body = {"error": {"message": "Provider returned error", "code": 429}}
    exc = RateLimitError("x", response=httpx.Response(429, request=REQ), body=body)
    pf = provider_error_fields(exc)
    assert pf["code"] == "rate_limit_exceeded" and pf["provider_status"] == 429


async def test_before_a_stream_the_server_answers_502_not_500():
    req = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [], "query_string": b""}
    )
    resp = await global_exception_handler(req, refused())
    body = json.loads(resp.body)
    assert resp.status_code == 502
    assert body["error"]["code"] == "provider_unreachable"
    assert body["error"]["message"] == UNREACHABLE_MESSAGE


async def test_inside_a_chat_completions_stream_a_silent_provider_is_a_timeout_event():
    async def stream_then_silence():
        yield {"chunk": 1}
        raise ProviderTimeoutError("idle", 60)

    events = [e async for e in _sse_generator(stream_then_silence(), context="chat_completion")]
    last = json.loads(events[-1][len("data: ") :])
    assert last["error"]["code"] == "provider_timeout"
    assert last["error"]["provider_status"] == 504
    assert last["error"]["message"] == TIMEOUT_MESSAGE


class _SilentProvider:
    """An inference API whose provider stream goes through the real call limits
    (short here): ``hang_first`` sends nothing, otherwise one chunk then silence."""

    def __init__(self, hang_first: bool):
        self.hang_first = hang_first

    async def openai_chat_completion(self, params):
        async def provider_stream():
            if not self.hang_first:
                yield {"id": "c1", "object": "chat.completion.chunk", "choices": []}
            await asyncio.sleep(5)

        async def make():
            return provider_stream()

        return await call_with_limits(make, bool(params.stream), first_response=0.2, idle=0.2)


async def _post_stream(impl) -> httpx.Response:
    app = FastAPI()
    app.include_router(create_router(impl))
    app.exception_handler(Exception)(global_exception_handler)  # as server.py registers it
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://ogx") as c:
        return await c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )


async def test_over_http_no_first_response_is_a_504_json_before_any_stream():
    # A streamed request whose provider sends nothing: the stream never starts,
    # so the answer is an HTTP error with a JSON body, as OpenAI's API answers an
    # error found before the stream (the SDK raises it before iterating).
    resp = await _post_stream(_SilentProvider(hang_first=True))
    assert resp.status_code == 504
    assert resp.headers["content-type"].startswith("application/json")
    err = resp.json()["error"]
    assert err["code"] == "provider_timeout" and err["message"] == TIMEOUT_MESSAGE
    assert err["provider_detail"] == "The model provider sent nothing for 0.2 s (first_response)."
    assert "data:" not in resp.text


async def test_over_http_silence_after_the_first_chunk_is_an_error_event_in_the_stream():
    # Once the stream has started the status is sent: the timeout is the stream's last event.
    resp = await _post_stream(_SilentProvider(hang_first=False))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    data = [line[len("data: ") :] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert len(data) == 2
    err = json.loads(data[-1])["error"]
    assert err["code"] == "provider_timeout" and err["provider_status"] == 504

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a provider error keeps its status and structured fields on every path.

The bodies are OpenRouter's live 429 (free shared pool, 2026-10-12) and its
401 for an unknown key.
"""

import json

import httpx
import pytest
from openai import APIError, AuthenticationError, RateLimitError
from starlette.requests import Request

from ogx.core.server.server import global_exception_handler
from ogx_api.common.errors import OpenAIErrorDetail, OpenAIErrorResponse
from ogx_api.common.provider_errors import ProviderRequestError, provider_error_fields
from ogx_api.inference.fastapi_routes import _sse_generator
from ogx_api.openai_responses import OpenAIResponseError

REQ = httpx.Request("POST", "http://provider.example/v1/chat/completions")
BODY_429 = {
    "error": {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {"raw": "m:free is temporarily rate-limited upstream.", "provider_name": "Google AI Studio"},
    },
    "user_id": "user_x",
}
BODY_401 = {"error": {"message": "User not found.", "code": 401}}


def status_error(cls, status, body):
    return cls("x", response=httpx.Response(status, request=REQ), body=body)


async def handle(exc, path="/v1/chat/completions"):
    req = Request({"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""})
    resp = await global_exception_handler(req, exc)
    return resp.status_code, json.loads(resp.body)


def test_fields_of_a_provider_error():
    pf = provider_error_fields(status_error(RateLimitError, 429, BODY_429))
    assert pf["provider_status"] == 429
    assert pf["provider"] == "Google AI Studio"
    assert pf["provider_detail"] == "m:free is temporarily rate-limited upstream."
    assert pf["code"] == "rate_limit_exceeded"
    assert pf["message"] == "Provider returned error"


def test_an_error_inside_a_stream_takes_its_status_from_the_body():
    pf = provider_error_fields(APIError("Provider returned error", REQ, body=BODY_429["error"]))
    assert pf["provider_status"] == 429 and pf["provider"] == "Google AI Studio"


def test_the_detail_is_the_message_when_there_is_no_raw_text_and_is_bounded():
    pf = provider_error_fields(status_error(AuthenticationError, 401, BODY_401))
    assert pf["provider_detail"] == "User not found." and pf["provider"] is None and pf["code"] is None
    long = status_error(AuthenticationError, 401, {"error": {"message": "x" * 5000}})
    assert len(provider_error_fields(long)["provider_detail"]) == 1000


class NotFromTheProviderSDKError(Exception):
    status_code = 429
    body = BODY_429


@pytest.mark.parametrize(
    "exc",
    # A connection error is a provider error since 2026-10-14 (provider_unreachable):
    # test_cupola_provider_unreachable.py.
    [ValueError("x"), NotFromTheProviderSDKError(), APIError("x", REQ, body=None)],
)
def test_other_errors_are_not_provider_errors(exc):
    assert provider_error_fields(exc) is None


def test_unset_fields_are_omitted_so_other_errors_serialize_as_before():
    assert (
        OpenAIResponseError(code="server_error", message="m").model_dump_json()
        == '{"code":"server_error","message":"m"}'
    )
    assert OpenAIErrorResponse(error=OpenAIErrorDetail(message="m")).to_dict() == {"error": {"message": "m"}}
    with_fields = OpenAIResponseError(code="c", message="m", provider_status=401, provider_detail="d")
    assert json.loads(with_fields.model_dump_json()) == {
        "code": "c",
        "message": "m",
        "provider_status": 401,
        "provider_detail": "d",
    }


async def test_chat_completions_answer_with_the_providers_status_and_fields():
    status, body = await handle(status_error(RateLimitError, 429, BODY_429))
    assert status == 429
    assert body == {
        "error": {
            "message": "Provider returned error",
            "code": "rate_limit_exceeded",
            "provider_status": 429,
            "provider": "Google AI Studio",
            "provider_detail": "m:free is temporarily rate-limited upstream.",
        }
    }
    status, body = await handle(status_error(AuthenticationError, 401, BODY_401))
    assert status == 401 and body["error"]["provider_status"] == 401
    assert "Error code" not in json.dumps(body)  # no Python repr of the provider's body


async def test_responses_without_a_stream_keep_the_status():
    status, body = await handle(
        ProviderRequestError({"code": "invalid_prompt", "message": "m", "provider_status": 403}), path="/v1/responses"
    )
    assert status == 403 and body["error"]["provider_status"] == 403 and body["error"]["code"] == "invalid_prompt"


async def test_other_errors_are_answered_as_before():
    status, body = await handle(ValueError("bad value"))
    assert status == 400 and body == {"error": {"message": "bad value"}}


async def test_a_provider_error_inside_a_chat_completions_stream():
    async def stream_then_fail():
        yield {"chunk": 1}
        raise APIError("Provider returned error", REQ, body=BODY_429["error"])

    events = [e async for e in _sse_generator(stream_then_fail(), context="chat_completion")]
    last = json.loads(events[-1][len("data: ") :])
    assert last["error"]["provider_status"] == 429
    assert last["error"]["code"] == "rate_limit_exceeded"
    assert last["error"]["provider"] == "Google AI Studio"


async def test_any_other_error_inside_a_stream_is_answered_as_before():
    async def stream_then_fail():
        yield {"chunk": 1}
        raise RuntimeError("boom")

    events = [e async for e in _sse_generator(stream_then_fail(), context="chat_completion")]
    assert json.loads(events[-1][len("data: ") :]) == {
        "error": {"message": "Internal server error: An unexpected error occurred.", "code": "500"}
    }

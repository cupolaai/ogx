# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: when the client goes away, the provider's stream is closed (2026-10-15).

Live (faz8.0.574 F5k), a MaaS client cut a stream at 30 s and the provider kept
streaming to the end: the close in call_with_limits was itself cancelled —
anyio re-delivers a cancellation every loop cycle and asyncio forwards it to
the task _preserve_context_for_sse awaits. The close is now shielded and
bounded (close_shielded), also when the client leaves before the first chunk,
and _postprocess_chunk closes the inner stream for every provider.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import StreamingResponse

from ogx.providers.utils.inference import cupola_limits
from ogx.providers.utils.inference.cupola_limits import call_with_limits, close_shielded
from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import OpenAIChatCompletionRequestWithExtraBody
from ogx_api.inference.fastapi_routes import _preserve_context_for_sse, _sse_generator
from tests.unit.providers.utils.inference.openai_mixin_helpers import OpenAIMixinImpl


class SlowCloseStream:
    """An SDK-like stream: one chunk every `every` seconds, for ever; its close
    takes a few awaits (as a network close does) and then records itself."""

    def __init__(self, every=0.02, first_after=0.0, close_hangs=False):
        self.every, self.first_after, self.close_hangs = every, first_after, close_hangs
        self.closed = False
        self.close_started = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        await asyncio.sleep(self.first_after)
        i = 0
        while True:
            yield {"id": "c", "object": "chat.completion.chunk", "choices": [], "n": i}
            i += 1
            await asyncio.sleep(self.every)

    async def close(self):
        self.close_started = True
        if self.close_hangs:
            await asyncio.sleep(3600)
        for _ in range(5):
            await asyncio.sleep(0.005)
        self.closed = True


async def _wait_for(cond, seconds=2.0):
    end = asyncio.get_running_loop().time() + seconds
    while not cond():
        if asyncio.get_running_loop().time() > end:
            return False
        await asyncio.sleep(0.01)
    return True


async def test_the_close_survives_a_cancellation_delivered_every_cycle():
    # anyio's delivery: the waiting task is cancelled again on every loop cycle.
    s = SlowCloseStream()
    task = asyncio.ensure_future(close_shielded(s))
    # the close has started, as when a disconnect cancels a stream already closing
    assert await _wait_for(lambda: s.close_started)
    while not task.done():
        task.cancel()
        await asyncio.sleep(0)
    assert await _wait_for(lambda: s.closed), "the close was cut by the cancellation"


async def test_a_hanging_close_is_given_up_after_the_limit():
    s = SlowCloseStream(close_hangs=True)
    await asyncio.wait_for(close_shielded(s, timeout=0.2), 2)
    assert s.close_started and not s.closed
    assert await _wait_for(lambda: not cupola_limits._closing), "a close task was left running"


async def test_the_default_close_limit_is_5_seconds():
    assert cupola_limits.CLOSE_TIMEOUT == 5.0


async def test_leaving_before_the_first_chunk_closes_the_stream():
    s = SlowCloseStream(first_after=5)

    async def make():
        return s

    task = asyncio.ensure_future(call_with_limits(make, True, first_response=10, idle=10))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_for(lambda: s.closed), "the provider stream stayed open"


async def test_closing_the_mixin_stream_closes_the_provider_stream_also_unlimited(
    monkeypatch,
):
    # In-cluster model servers (vLLM) have no Cupola limits; the stream is closed all the same.
    s = SlowCloseStream()
    impl = OpenAIMixinImpl(config=RemoteInferenceProviderConfig())
    impl.model_store = MagicMock(has_model=AsyncMock(return_value=False))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=s)
    monkeypatch.setattr(type(impl), "client", property(lambda self: client))
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    gen = await impl.openai_chat_completion(params)
    await gen.__anext__()
    await gen.aclose()
    assert await _wait_for(lambda: s.closed)


class LimitedImpl(OpenAIMixinImpl):
    cupola_call_limits: bool = True


async def test_closing_the_mixin_stream_closes_the_provider_stream_through_the_limits(
    monkeypatch,
):
    # External providers: the mixin stream wraps call_with_limits' generator, which
    # must be closed with aclose() so its own close reaches the provider stream.
    s = SlowCloseStream()
    # Keep a reference to the limits' generator, as a caller may: then only an
    # explicit aclose() — not the garbage collector — closes it.
    kept = []
    real = cupola_limits.call_with_limits

    async def keeping(*a, **k):
        gen = await real(*a, **k)
        kept.append(gen)
        return gen

    monkeypatch.setattr(cupola_limits, "call_with_limits", keeping)
    impl = LimitedImpl(config=RemoteInferenceProviderConfig())
    impl.model_store = MagicMock(has_model=AsyncMock(return_value=False))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=s)
    monkeypatch.setattr(type(impl), "client", property(lambda self: client))
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    gen = await impl.openai_chat_completion(params)
    await gen.__anext__()
    await gen.aclose()
    assert await _wait_for(lambda: s.closed)


async def test_the_limits_close_survives_a_cancellation_delivered_every_cycle():
    # The consumer of call_with_limits' stream is cancelled again on every loop
    # cycle (anyio, forwarded by asyncio) while it waits for the next chunk.
    s = SlowCloseStream(every=10)

    async def make():
        return s

    stream = await call_with_limits(make, True, first_response=5, idle=30)
    await stream.__anext__()
    task = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.05)
    while not task.done():
        task.cancel()
        await asyncio.sleep(0)
    assert await _wait_for(lambda: s.closed), "the limits' close was cut by the cancellation"


@pytest.mark.parametrize("limited", [True, False], ids=["external-provider", "in-cluster"])
async def test_a_client_disconnect_through_the_real_server_path_closes_the_provider_stream(monkeypatch, limited):
    """Starlette's StreamingResponse (ASGI 2.3, as uvicorn's h11 reports) cancels the
    stream through anyio when the client disconnects; the provider stream must close."""
    s = SlowCloseStream(every=0.02)
    impl = (LimitedImpl if limited else OpenAIMixinImpl)(config=RemoteInferenceProviderConfig())
    impl.model_store = MagicMock(has_model=AsyncMock(return_value=False))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=s)
    monkeypatch.setattr(type(impl), "client", property(lambda self: client))
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    result = await impl.openai_chat_completion(params)
    response = StreamingResponse(
        _preserve_context_for_sse(_sse_generator(result, context="chat_completion")),
        media_type="text/event-stream",
    )

    bodies: list[bytes] = []
    gone = asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            bodies.append(message["body"])
            if len(bodies) == 3:
                gone.set()  # the client leaves after three chunks

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
    }
    await asyncio.wait_for(response(scope, receive, send), 5)
    assert len(bodies) >= 3 and json.loads(bodies[0].decode()[len("data: ") :])["n"] == 0
    assert await _wait_for(lambda: s.closed), "the client left and the provider stream stayed open"

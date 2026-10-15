# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: a client disconnect closes the provider's connection — over a real network.

Live (Cupola faz8.0.576 L1, OGX 1.1.3-cupola.12) the provider connection stayed
ESTABLISHED after the client left although Cupola's close ran: httpcore's own
close was cut by the cancellation _preserve_context_for_sse forwarded into the
task it awaits on every anyio delivery cycle, and httpcore had already marked
the stream closed, so every later close was a no-op. Tests with fake streams
(a Python close()) could not see it; this one uses the real SDK, httpx and
httpcore against a real TCP provider.
"""

import asyncio
import json
import socket

import pytest
import uvicorn
from fastapi import FastAPI

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import OpenAIChatCompletionRequestWithExtraBody
from ogx_api.inference.fastapi_routes import create_router
from ogx_api.responses.fastapi_routes import create_router as create_responses_router
from tests.unit.providers.utils.inference.openai_mixin_helpers import OpenAIMixinImpl

CHUNK = {
    "id": "c",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
}


class TCPProvider:
    """An OpenAI-compatible provider on a real socket: one SSE chunk every
    `every` seconds until the client closes the connection, which it records."""

    def __init__(self, every=0.05, first_after=0.0):
        self.every, self.first_after = every, first_after
        self.closed = asyncio.Event()
        self.requests = 0

    async def handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(
            (int(line.split(b":")[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0
        )
        await reader.readexactly(length)
        self.requests += 1
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")

        async def watch():  # the client closing its end
            try:
                await reader.read()
            finally:
                self.closed.set()

        watcher = asyncio.create_task(watch())
        try:
            await asyncio.sleep(self.first_after)
            while not self.closed.is_set():
                data = ("data: " + json.dumps(CHUNK) + "\n\n").encode()
                writer.write(b"%x\r\n%s\r\n" % (len(data), data))
                await writer.drain()
                await asyncio.sleep(self.every)
        except (ConnectionError, OSError):
            self.closed.set()
        finally:
            watcher.cancel()
            writer.close()


class NetworkImpl(OpenAIMixinImpl):
    base: str = ""
    cupola_call_limits: bool = False

    def get_base_url(self) -> str:
        return self.base


class LimitedNetworkImpl(NetworkImpl):
    cupola_call_limits: bool = True


class StreamingResponses:
    """A Responses implementation that streams the provider's answer the way the
    builtin one does (consuming openai_chat_completion's stream in a generator), so
    the Responses route is exercised over the real SDK, httpx and httpcore."""

    def __init__(self, inference):
        self.inference = inference

    async def create_openai_response(self, request):
        params = OpenAIChatCompletionRequestWithExtraBody(
            model=request.model, messages=[{"role": "user", "content": "hi"}], stream=True
        )
        stream = await self.inference.openai_chat_completion(params)

        async def events():
            async for chunk in stream:
                yield {"type": "response.output_text.delta", "delta": chunk.choices[0].delta.content}

        return events()


async def _serve_and_disconnect(impl_cls, provider, after_chunks=2, api="chat"):
    """The whole server path: uvicorn (h11) serving the real inference router (or the
    real Responses router, api="responses"); a raw TCP client reads `after_chunks`
    chunks and closes its socket."""
    prov = await asyncio.start_server(provider.handle, "127.0.0.1", 0)
    prov_port = prov.sockets[0].getsockname()[1]
    impl = impl_cls(config=RemoteInferenceProviderConfig(), base=f"http://127.0.0.1:{prov_port}/v1")

    async def no_model(*_a, **_k):
        return False

    impl.model_store = type("S", (), {"has_model": staticmethod(no_model)})()
    app = FastAPI()
    if api == "responses":
        app.include_router(create_responses_router(StreamingResponses(impl)))
        path = b"/v1/responses"
        request = {"model": "m", "input": "hi", "stream": True}
    else:
        app.include_router(create_router(impl))
        path = b"/v1/chat/completions"
        request = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, http="h11", loop="asyncio", lifespan="off", log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        body = json.dumps(request).encode()
        writer.write(
            b"POST %s HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n"
            b"content-length: %d\r\n\r\n%s" % (path, len(body), body)
        )
        await writer.drain()
        got = b""
        while got.count(b"data: ") < after_chunks:
            part = await asyncio.wait_for(reader.read(4096), 10)
            if not part:
                break
            got += part
        writer.close()  # the client leaves
        closed = True
        try:
            await asyncio.wait_for(provider.closed.wait(), 3)
        except TimeoutError:
            closed = False
        return closed, got.count(b"data: ")
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 10)
        prov.close()


@pytest.mark.parametrize("impl_cls", [LimitedNetworkImpl, NetworkImpl], ids=["external-provider", "in-cluster"])
async def test_a_client_disconnect_closes_the_provider_connection(impl_cls):
    # A chunk every second: when the client leaves, the provider read is in flight
    # inside httpcore — where live the close was cut.
    provider = TCPProvider(every=1.0)
    closed, chunks = await _serve_and_disconnect(impl_cls, provider, after_chunks=2)
    assert provider.requests == 1 and chunks >= 2
    assert closed, "the client left and the provider connection stayed open"


@pytest.mark.parametrize("impl_cls", [LimitedNetworkImpl, NetworkImpl], ids=["external-provider", "in-cluster"])
async def test_a_client_disconnect_on_the_responses_route_closes_the_provider_connection(impl_cls):
    # Studio and Chat stream through /v1/responses; their Stop is this disconnect
    # (Cupola faz8.0.576 L3: on 1.1.3-cupola.13 the provider stream ran to its end).
    provider = TCPProvider(every=1.0)
    closed, chunks = await _serve_and_disconnect(impl_cls, provider, after_chunks=2, api="responses")
    assert provider.requests == 1 and chunks >= 2
    assert closed, "the client left /v1/responses and the provider connection stayed open"


def test_the_responses_route_streams_through_the_inference_routes_wrapper():
    import ogx_api.inference.fastapi_routes as inference_routes
    import ogx_api.responses.fastapi_routes as responses_routes

    assert responses_routes._preserve_context_for_sse is inference_routes._preserve_context_for_sse


async def test_closing_between_chunks_closes_the_stream_in_its_own_task():
    import ogx_api.inference.fastapi_routes as fr

    closed = asyncio.Event()

    async def gen():
        try:
            while True:
                yield "data: x\n\n"
                await asyncio.sleep(0.01)
        finally:
            closed.set()

    stream = gen()  # kept here, as a caller may: only an explicit aclose() closes it, not the GC
    wrapper = fr._preserve_context_for_sse(stream)
    await wrapper.__anext__()  # the chunk task is done: the stream waits between chunks
    await wrapper.aclose()
    await asyncio.wait_for(closed.wait(), 2)
    del stream


async def test_a_hanging_clean_up_is_cancelled_after_the_limit(monkeypatch):
    import ogx_api.inference.fastapi_routes as fr

    monkeypatch.setattr(fr, "_CUPOLA_CLEANUP_LIMIT", 0.2)
    started = asyncio.Event()

    async def gen():
        try:
            yield "data: x\n\n"
            started.set()
            await asyncio.sleep(3600)
        finally:
            await asyncio.sleep(3600)  # a clean-up that never ends

    wrapper = fr._preserve_context_for_sse(gen())
    await wrapper.__anext__()
    consumer = asyncio.ensure_future(wrapper.__anext__())
    await started.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert len(fr._cupola_cleanups) == 1
    for _ in range(100):
        if not fr._cupola_cleanups:
            break
        await asyncio.sleep(0.02)
    assert not fr._cupola_cleanups, "a hanging clean-up was left running past the limit"

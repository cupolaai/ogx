# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cupola: limits on calls to an external model provider (decision 2026-10-14).

The provider SDK's defaults let a provider that accepts the connection and
then stays silent hold a chat for up to 600 s per attempt, with two retries.
For the external providers (OpenAI-compatible, Anthropic, AWS Bedrock) Cupola
sets instead:

- connect timeout 5 s and one retry (the SDK's retry rules otherwise);
- the first response — the first chunk of a stream — within 180 s, the retry
  included (reasoning models may think long before they answer);
- in a stream, at most 60 s of silence between chunks; there is no total
  limit, so a long answer that keeps coming is never cut;
- a whole answer without a stream within 110 s (under the 120 s the Cupola
  backend waits for one).

A provider stream is closed when its consumer stops — the client went away, a
Stop, an error — through close_shielded: the close runs in its own task, so a
cancelled caller cannot cut it, and at most CLOSE_TIMEOUT seconds (2026-10-15:
the client's disconnect never reached the provider, because the close was
itself cancelled — anyio re-delivers the cancellation every loop cycle and
asyncio forwards it to the task _preserve_context_for_sse awaits).

A limit raises ProviderTimeoutError, which the API answers as
``provider_timeout`` (HTTP 504, or an error event inside a stream). In-cluster
model servers (vLLM) are not limited: a large model on CPU may take long to
start answering. The values are fixed here, not a setting.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from ogx_api.common.provider_errors import ProviderTimeoutError

CONNECT_TIMEOUT = 5.0
MAX_RETRIES = 1
FIRST_RESPONSE_TIMEOUT = 180.0
IDLE_TIMEOUT = 60.0
NON_STREAM_TIMEOUT = 110.0
CLOSE_TIMEOUT = 5.0

# The close tasks still running; a reference keeps each alive until it ends.
_closing: set[asyncio.Task[None]] = set()


def client_params() -> dict[str, Any]:
    """AsyncOpenAI parameters for a limited provider: one retry, connect 5 s.

    The read timeout is the first-response limit, a backstop under the limits
    applied around each call by call_with_limits.
    """
    return {"max_retries": MAX_RETRIES, "timeout": httpx.Timeout(FIRST_RESPONSE_TIMEOUT, connect=CONNECT_TIMEOUT)}


async def call_with_limits(
    make_call: Callable[[], Awaitable[Any]],
    stream: bool,
    *,
    first_response: float = FIRST_RESPONSE_TIMEOUT,
    idle: float = IDLE_TIMEOUT,
    non_stream: float = NON_STREAM_TIMEOUT,
) -> Any:
    """Run one provider call under the limits; a stream is returned re-wrapped."""
    if not stream:
        try:
            return await asyncio.wait_for(make_call(), non_stream)
        except TimeoutError as e:
            raise ProviderTimeoutError("response", non_stream) from e

    holder: dict[str, Any] = {}

    async def first_chunk() -> tuple[AsyncIterator[Any], Any, bool]:
        resp = await make_call()
        holder["resp"] = resp
        it = resp.__aiter__()
        try:
            return it, await it.__anext__(), False
        except StopAsyncIteration:
            return it, None, True

    try:
        it, first, empty = await asyncio.wait_for(first_chunk(), first_response)
    except TimeoutError as e:
        await close_shielded(holder.get("resp"))
        raise ProviderTimeoutError("first_response", first_response) from e
    except BaseException:
        # Cancelled while waiting (the client went away): close all the same.
        await close_shielded(holder.get("resp"))
        raise

    async def chunks() -> AsyncIterator[Any]:
        try:
            if empty:
                return
            yield first
            while True:
                try:
                    chunk = await asyncio.wait_for(it.__anext__(), idle)
                except StopAsyncIteration:
                    return
                except TimeoutError as e:
                    raise ProviderTimeoutError("idle", idle) from e
                yield chunk
        finally:
            await close_shielded(holder.get("resp"))

    return chunks()


async def close_shielded(resp: Any, timeout: float = CLOSE_TIMEOUT) -> None:
    """Close a provider stream even while the caller is being cancelled.

    The close runs in its own task: a cancelled caller gets its cancellation
    back at once, the close goes on, for at most `timeout` seconds.
    """
    if resp is None:
        return
    task = asyncio.ensure_future(_close_within(resp, timeout))
    _closing.add(task)
    task.add_done_callback(_closing.discard)
    await asyncio.shield(task)


async def _close_within(resp: Any, timeout: float) -> None:
    # A close that fails or hangs must not hide the real error or hang a task.
    with contextlib.suppress(Exception):
        await asyncio.wait_for(_close(resp), timeout)


async def _close(resp: Any) -> None:
    # An async generator (a wrapped stream) closes with aclose(), the SDK's
    # stream with close().
    close = getattr(resp, "aclose", None) or getattr(resp, "close", None)
    if close is None:
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result

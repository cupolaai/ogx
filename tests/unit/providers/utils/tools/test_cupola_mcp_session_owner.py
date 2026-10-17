# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# CUPOLA (faz8.0.593) — after one MCP tool call through a streamed response, OGX burned a
# whole core, idle, until restarted: MCPSessionManager entered the MCP client's anyio task
# group in the task that first needed the session and left it in __aexit__, run by another
# task at the end of the stream; anyio refused that exit, the cancel scope stayed, and
# anyio's _deliver_cancellation rescheduled itself on every loop turn. Each session now
# lives in one owner task. These tests open a session in one task, use it, close the
# manager from another task and measure the process's CPU time while idle — against a
# real MCP server (the SDK's FastMCP) and one that answers JSON and refuses GET (405), as
# Cupola's load-test stub does.

import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ogx.providers.utils.tools.mcp import MCPSessionManager


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def fastmcp_app():
    server = FastMCP("cupola-test")

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text."""
        return "echo: " + text

    return server.streamable_http_app(), "/mcp"


def json_405_app():
    async def mcp(request: Request):
        q = json.loads(await request.body())
        if "id" not in q:
            return Response(status_code=202)
        rid, method = q["id"], q["method"]
        if method == "initialize":
            result = {
                "protocolVersion": q["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stub", "version": "1"},
            }
            return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result}, headers={"Mcp-Session-Id": "s1"})
        if method == "tools/list":
            tool = {"name": "echo", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}
            return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": {"tools": [tool]}})
        if method == "tools/call":
            text = q["params"]["arguments"]["text"]
            result = {"content": [{"type": "text", "text": "echo: " + text}], "isError": False}
            return JSONResponse({"jsonrpc": "2.0", "id": rid, "result": result})
        return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": method}})

    async def no_stream(request: Request):
        return Response(status_code=405)

    return Starlette(routes=[Route("/mcp", mcp, methods=["POST"]), Route("/mcp", no_stream, methods=["GET"])]), "/mcp"


@pytest.fixture(params=["fastmcp", "json-405"])
def mcp_url(request):
    app, path = fastmcp_app() if request.param == "fastmcp" else json_405_app()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the MCP server did not start"
    yield f"http://127.0.0.1:{port}{path}"
    server.should_exit = True
    thread.join(timeout=10)


async def idle_cpu(seconds: float) -> float:
    """CPU seconds this process uses while the event loop has nothing to do."""
    t0 = time.process_time()
    await asyncio.sleep(seconds)
    return time.process_time() - t0


async def test_session_closed_from_another_task_leaves_the_loop_idle(mcp_url):
    manager = MCPSessionManager()

    async def use():  # the task that first needs the session, as a tool call does
        session = await manager.get_session(mcp_url, {})
        tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["echo"]
        result = await session.call_tool("echo", {"text": "hi"})
        assert result.content[0].text == "echo: hi"

    await asyncio.create_task(use())
    # the end of the streamed response: another task closes the manager
    t0 = time.monotonic()
    await asyncio.create_task(manager.__aexit__(None, None, None))
    took = time.monotonic() - t0
    # closed by the time __aexit__ returns, promptly (signalled, not timed out)
    left = [t.get_name() for t in asyncio.all_tasks() if t.get_name().startswith("mcp-session")]
    assert left == [], f"owner tasks left running: {left}"
    assert took < 2, f"closing took {took:.1f} s"
    await asyncio.sleep(0.2)

    used = await idle_cpu(1.5)
    assert used < 0.5, f"{used:.2f} CPU seconds in 1.5 s idle — something keeps the loop busy"


async def test_one_session_per_server_reused_across_tasks(mcp_url):
    async with MCPSessionManager() as manager:
        sessions = await asyncio.gather(*[asyncio.create_task(manager.get_session(mcp_url, {})) for _ in range(5)])
        assert all(s is sessions[0] for s in sessions)
        results = await asyncio.gather(
            *[asyncio.create_task(sessions[0].call_tool("echo", {"text": str(i)})) for i in range(5)]
        )
        assert sorted(r.content[0].text for r in results) == [f"echo: {i}" for i in range(5)]
    assert [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp-session")] == []


async def test_closing_while_the_closer_is_cancelled_still_closes_the_session(mcp_url):
    manager = MCPSessionManager()
    await asyncio.create_task(manager.get_session(mcp_url, {}))
    closer = asyncio.create_task(manager.__aexit__(None, None, None))
    await asyncio.sleep(0)
    closer.cancel()  # a stream whose client went away
    try:
        await closer
    except asyncio.CancelledError:
        pass
    deadline = time.time() + 5
    while time.time() < deadline and any(t.get_name().startswith("mcp-session") for t in asyncio.all_tasks()):
        await asyncio.sleep(0.05)
    assert [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp-session")] == []
    assert await idle_cpu(1.0) < 0.4


async def test_unreachable_server_raises_and_leaves_nothing_running():
    manager = MCPSessionManager()
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(manager.get_session(f"http://127.0.0.1:{_free_port()}/mcp", {}), timeout=30)
    await asyncio.sleep(0.1)
    assert [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp-session")] == []
    await manager.__aexit__(None, None, None)

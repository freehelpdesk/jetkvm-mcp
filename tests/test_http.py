"""Streamable HTTP for Grok and Grok Bot: auth, and a real tool call through it."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import pytest
import uvicorn
from fake_device import FakeJetKVM
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image

from jetkvm_mcp.client import JetKVMConfig
from jetkvm_mcp.server import Settings, build_http_app, main, validate_http_token

TOKEN = "a" * 32


def test_http_token_rules():
    assert validate_http_token(TOKEN) == TOKEN
    for bad in (None, "", "short-token", "has a space here!!", "slash/in-the-token", "ü" * 16):
        with pytest.raises(ValueError):
            validate_http_token(bad)


def test_settings_read_http_token(monkeypatch):
    monkeypatch.setenv("JETKVM_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("JETKVM_HTTP_TOKEN", f"  {TOKEN}  ")
    assert Settings.from_env().http_token == TOKEN


def test_serve_http_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.setenv("JETKVM_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("JETKVM_HTTP_TOKEN", raising=False)
    with pytest.raises(SystemExit) as raised:
        main(["serve", "--http"])
    assert raised.value.code == 2


@asynccontextmanager
async def _listening(app) -> AsyncIterator[int]:
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = False
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            if task.done():
                await task
                raise RuntimeError("HTTP server exited before it started")
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("HTTP server did not start")
        sockets = server.servers[0].sockets
        assert sockets is not None
        yield sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        await task


def _image(result) -> Image.Image:
    block = next(b for b in result.content if b.type == "image")
    return Image.open(io.BytesIO(base64.b64decode(block.data))).convert("RGB")


async def test_http_serves_grok_clients():
    device = FakeJetKVM()
    url = await device.start()
    try:
        settings = Settings(
            jetkvm=JetKVMConfig(url=url, password="hunter2", connect_timeout=15),
            http_token=TOKEN,
        )
        async with _listening(build_http_app(settings)) as port:
            base = f"http://127.0.0.1:{port}"
            async with httpx2.AsyncClient() as http:
                health = await http.get(f"{base}/healthz")
                assert health.status_code == 200 and health.text == "ok\n"
                refused = await http.post(
                    f"{base}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    headers={"content-type": "application/json"},
                )
                assert refused.status_code == 401
                assert refused.headers["www-authenticate"] == "Bearer"
                wrong = await http.post(
                    f"{base}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    headers={"content-type": "application/json", "authorization": "Bearer " + "b" * 32},
                )
                assert wrong.status_code == 401
                raw = await http.post(
                    f"{base}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
                    headers={"content-type": "application/json", "authorization": TOKEN},
                )
                assert raw.status_code != 401

            async with httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=httpx2.Timeout(30, read=60),
            ) as http:
                async with streamable_http_client(f"{base}/mcp", http_client=http) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        listed = await session.list_tools()
                        assert "screenshot" in {tool.name for tool in listed.tools}

            async with httpx2.AsyncClient(timeout=httpx2.Timeout(30, read=60)) as http:
                async with streamable_http_client(f"{base}/mcp/{TOKEN}", http_client=http) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        names = {tool.name for tool in (await session.list_tools()).tools}
                        assert "screenshot" in names and "click" in names
                        shot = await session.call_tool("screenshot", {})
            assert not shot.is_error
            image = _image(shot)
            assert image.size == (1280, 720)
            red = image.getpixel((150, 100))
            assert red[0] > 150 and red[1] < 90 and red[2] < 90
    finally:
        await device.stop()

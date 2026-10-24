"""End-to-end: MCP client -> jetkvm-mcp server -> WebRTC -> simulated JetKVM."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from fake_device import RED_SQUARE, FakeJetKVM
from mcp import Client
from PIL import Image

from jetkvm_mcp import client as client_module
from jetkvm_mcp.client import JetKVMClient, JetKVMConfig
from jetkvm_mcp.server import Settings, build_server


async def eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


@pytest.fixture
async def device() -> AsyncIterator[tuple[FakeJetKVM, str]]:
    dev = FakeJetKVM()
    url = await dev.start()
    yield dev, url
    await dev.stop()


def settings(url: str, password: str | None = "hunter2", **kwargs) -> Settings:
    return Settings(jetkvm=JetKVMConfig(url=url, password=password, connect_timeout=15), **kwargs)


@asynccontextmanager
async def mcp(cfg: Settings) -> AsyncIterator[Client]:
    async with Client(build_server(cfg)) as c:
        yield c


def text_of(result) -> str:
    return " ".join(block.text for block in result.content if block.type == "text")


def image_of(result) -> Image.Image:
    block = next(b for b in result.content if b.type == "image")
    return Image.open(io.BytesIO(base64.b64decode(block.data))).convert("RGB")


def is_red(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    return r > 150 and g < 90 and b < 90


async def test_lists_tools(device):
    _, url = device
    async with mcp(settings(url)) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert {"screenshot", "click", "type_text", "press_keys", "power", "mount_virtual_media"} <= names
    assert "device_rpc" not in names  # opt-in only


async def test_screenshot_shows_screen_in_native_space(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("screenshot", {})
    assert not result.is_error, text_of(result)
    assert "Screen 1280x720." in text_of(result)
    image = image_of(result)
    assert image.size == (1280, 720)
    assert is_red(image.getpixel((150, 100)))
    assert not is_red(image.getpixel((20, 20)))
    assert dev.logins == 1


async def test_screenshot_downscales_and_clicks_follow(device):
    dev, url = device
    async with mcp(settings(url, max_edge=640)) as c:
        shot = await c.call_tool("screenshot", {})
        assert "Screen 640x360 (scaled from native 1280x720)" in text_of(shot)
        image = image_of(shot)
        assert image.size == (640, 360)
        assert is_red(image.getpixel((75, 50)))  # the square, at half scale

        result = await c.call_tool("click", {"x": 75, "y": 50})
        assert not result.is_error, text_of(result)
    await eventually(lambda: len(dev.pointer) >= 3)
    x, y, _ = dev.pointer[-1]
    # Where the host would put the cursor, in native pixels: inside the red square.
    native_x, native_y = x / 32767 * 1280, y / 32767 * 720
    left, top, right, bottom = RED_SQUARE
    assert left <= native_x < right and top <= native_y < bottom
    assert [b for _, _, b in dev.pointer[-3:]] == [0, 1, 0]


async def test_double_click_with_modifier(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("click", {"x": 10, "y": 10, "count": 2, "modifiers": "shift"})
        assert not result.is_error, text_of(result)
    await eventually(lambda: len(dev.pointer) >= 5 and len(dev.keyboard) >= 2)
    assert [b for _, _, b in dev.pointer] == [0, 1, 0, 1, 0]
    assert dev.keyboard == [(0x02, ()), (0, ())]


async def test_type_text_runs_as_device_macro(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("type_text", {"text": "Hi!", "delay_ms": 5})
        assert not result.is_error, text_of(result)
    assert dev.keyboard == [
        (0x02, (0x0B,)),
        (0, ()),
        (0, (0x0C,)),
        (0, ()),
        (0x02, (0x1E,)),
        (0, ()),
    ]
    assert "keyboardReport" not in dev.rpc_log  # went over hidrpc, not JSON-RPC


async def test_type_text_compensates_for_caps_lock(device):
    dev, url = device
    dev.led_state["caps_lock"] = True
    async with mcp(settings(url)) as c:
        result = await c.call_tool("type_text", {"text": "aA", "delay_ms": 1})
        assert "Caps Lock is on" in text_of(result)
    assert [mod for mod, keys in dev.keyboard if keys] == [0x02, 0]


async def test_long_text_is_chunked(device):
    dev, url = device
    text = "abc" * 70  # 420 steps -> 4 macro chunks
    async with mcp(settings(url)) as c:
        result = await c.call_tool("type_text", {"text": text, "delay_ms": 1})
        assert not result.is_error, text_of(result)
    assert len([1 for _, keys in dev.keyboard if keys]) == len(text)


async def test_press_keys_chord_and_sequence(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        await c.call_tool("press_keys", {"keys": "ctrl+alt+delete"})
        await c.call_tool("press_keys", {"keys": "down enter", "interval_ms": 10})
    assert dev.keyboard == [
        (0x05, ()),
        (0x05, (0x4C,)),
        (0x05, ()),
        (0, ()),
        (0, (0x51,)),
        (0, ()),
        (0, (0x28,)),
        (0, ()),
    ]


async def test_bad_key_name_is_a_tool_error(device):
    _, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("press_keys", {"keys": "ctrl+banana"})
    assert result.is_error
    assert "unknown key name 'banana'" in text_of(result)


async def test_scroll_sends_both_wheel_axes(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        await c.call_tool("scroll", {"direction": "down", "amount": 2, "x": 100, "y": 100})
        await c.call_tool("scroll", {"direction": "right", "amount": 1})
    assert dev.wheel == [(-1, 0), (-1, 0), (0, 1)]


async def test_drag_holds_button_across_moves(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("drag", {"start_x": 10, "start_y": 10, "end_x": 500, "end_y": 300, "duration_ms": 100})
        assert not result.is_error, text_of(result)
    await eventually(lambda: dev.pointer and dev.pointer[-1][2] == 0)
    buttons = [b for _, _, b in dev.pointer]
    assert buttons[0] == 0 and buttons[-1] == 0 and set(buttons[1:-1]) == {1}
    assert dev.pointer[-2][:2] == (round(500.5 / 1280 * 32767), round(300.5 / 720 * 32767))


async def test_relative_move_splits_into_signed_bytes(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        await c.call_tool("mouse_move_relative", {"dx": 300, "dy": -10})
    await eventually(lambda: len(dev.relative) >= 3)
    assert sum(dx for dx, _, _ in dev.relative) == 300
    assert sum(dy for _, dy, _ in dev.relative) == -10
    assert all(-127 <= dx <= 127 for dx, _, _ in dev.relative)


async def test_out_of_range_coordinates_rejected(device):
    _, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("click", {"x": 5000, "y": 10})
    assert result.is_error
    assert "outside the screen" in text_of(result)


async def test_power_requires_extension(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        result = await c.call_tool("power", {"action": "power_button"})
        assert result.is_error
        assert "ATX power extension is not active" in text_of(result)
        dev.extension = "atx-power"
        result = await c.call_tool("power", {"action": "reset_button"})
        assert not result.is_error, text_of(result)
    assert dev.atx_actions == ["reset"]


async def test_status_and_media(device):
    _, url = device
    async with mcp(settings(url)) as c:
        status = json.loads(text_of(await c.call_tool("get_status", {})))
        media = json.loads(text_of(await c.call_tool("list_virtual_media", {})))
    assert status["device"] == {"url": url, "firmware": "0.0.0-fake", "hid_transport": "hidrpc"}
    assert status["usb"] == "configured"
    assert status["screenshot_size"] == {"width": 1280, "height": 720}
    assert media["stored_images"]["files"][0]["filename"] == "ubuntu.iso"
    assert media["mounted"] is None  # omitempty result


async def test_wrong_password(device):
    dev, url = device
    async with mcp(settings(url, password="wrong")) as c:
        result = await c.call_tool("screenshot", {})
    assert result.is_error
    assert "rejected the password" in text_of(result)
    assert dev.logins == 0


async def test_missing_password(device):
    _, url = device
    async with mcp(settings(url, password=None)) as c:
        result = await c.call_tool("screenshot", {})
    assert result.is_error
    assert "set JETKVM_PASSWORD" in text_of(result)


async def test_no_password_mode():
    dev = FakeJetKVM(password=None)
    url = await dev.start()
    try:
        async with mcp(settings(url, password=None)) as c:
            result = await c.call_tool("screenshot", {})
        assert not result.is_error, text_of(result)
    finally:
        await dev.stop()


async def test_reconnects_after_another_session_takes_over(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        assert not (await c.call_tool("screenshot", {})).is_error
        # Someone opens the web UI: the device moves to the new session.
        browser = JetKVMClient(JetKVMConfig(url=url, password="hunter2"))
        await browser.session()
        await asyncio.sleep(0.3)
        result = await c.call_tool("press_keys", {"keys": "a"})
        assert not result.is_error, text_of(result)
        assert "had taken over" in text_of(result)
        await browser.close()
    assert len(dev.sessions) == 3
    assert dev.keyboard[-2:] == [(0, (0x04,)), (0, ())]


async def test_json_rpc_fallback_without_hidrpc(monkeypatch):
    monkeypatch.setattr(client_module, "HANDSHAKE_TIMEOUT", 0.5)
    dev = FakeJetKVM(hidrpc=False)
    url = await dev.start()
    try:
        async with mcp(settings(url)) as c:
            status = json.loads(text_of(await c.call_tool("get_status", {})))
            assert status["device"]["hid_transport"] == "json-rpc"
            await c.call_tool("type_text", {"text": "ab", "delay_ms": 1})
            await c.call_tool("click", {"x": 10, "y": 10})
        assert dev.keyboard == [(0, (0x04,)), (0, ()), (0, (0x05,)), (0, ())]
        assert [b for _, _, b in dev.pointer] == [0, 1, 0]
    finally:
        await dev.stop()


async def test_zoom_returns_native_crop(device):
    _, url = device
    async with mcp(settings(url, max_edge=640)) as c:
        # Screenshot space is half scale; this region covers the red square's top-left area.
        result = await c.call_tool("zoom", {"x": 40, "y": 15, "width": 40, "height": 20})
    assert not result.is_error, text_of(result)
    assert "shown at 80x40" in text_of(result)
    image = image_of(result)
    assert image.size == (80, 40)
    assert is_red(image.getpixel((60, 30))) and not is_red(image.getpixel((5, 5)))


async def test_auto_screenshot_after_actions(device):
    _, url = device
    async with mcp(settings(url, auto_screenshot=True)) as c:
        result = await c.call_tool("press_keys", {"keys": "esc"})
    assert [b.type for b in result.content] == ["text", "image"]
    assert text_of(result).startswith("Pressed esc. Screen 1280x720")


async def test_hold_buttons_and_release_all(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        await c.call_tool("hold_keys", {"keys": "shift", "duration_ms": 200})
        await c.call_tool("mouse_move", {"x": 10, "y": 10})
        await c.call_tool("mouse_button", {"action": "down", "button": "right"})
        await c.call_tool("mouse_move", {"x": 20, "y": 20})  # still held: a right-drag
        await c.call_tool("release_all", {})
    await eventually(lambda: len(dev.pointer) >= 4)
    assert dev.keyboard[:2] == [(0x02, ()), (0, ())]
    assert dev.keyboard[-1] == (0, ())
    assert [b for _, _, b in dev.pointer] == [0, 2, 2, 0]


async def test_mount_virtual_media(device):
    dev, url = device
    async with mcp(settings(url)) as c:
        await c.call_tool("mount_virtual_media", {"source": "https://example.com/os.iso"})
        assert dev.mounted == {"source": "HTTP", "mode": "CDROM", "url": "https://example.com/os.iso"}
        await c.call_tool("mount_virtual_media", {"source": "ubuntu.iso", "mode": "Disk"})
        assert dev.mounted == {"source": "Storage", "mode": "Disk", "filename": "ubuntu.iso"}
        await c.call_tool("unmount_virtual_media", {})
        assert dev.mounted is None


async def test_large_keyframes_decode(monkeypatch):
    # A busy 1080p screen at a high bitrate produces keyframes far larger than aiortc's
    # default 128-packet jitter buffer; without enlarging it, no frame ever decodes.
    from aiortc.codecs import h264

    monkeypatch.setattr(h264, "DEFAULT_BITRATE", 20_000_000)
    monkeypatch.setattr(h264, "MAX_BITRATE", 20_000_000)
    dev = FakeJetKVM(width=1920, height=1080, noise=True)
    url = await dev.start()
    try:
        async with mcp(settings(url)) as c:
            result = await c.call_tool("screenshot", {})
        assert not result.is_error, text_of(result)
        assert image_of(result).size == (1920, 1080)
    finally:
        await dev.stop()


async def test_raw_rpc_is_opt_in(device):
    _, url = device
    async with mcp(settings(url, enable_raw_rpc=True)) as c:
        result = await c.call_tool("device_rpc", {"method": "getATXState"})
    assert json.loads(text_of(result)) == {"power": True, "hdd": False}

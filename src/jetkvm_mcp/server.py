"""MCP server that operates a computer through a JetKVM: screen, keyboard, mouse, power, media."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hmac
import io
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from urllib.parse import unquote

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from PIL import Image as PILImage
from pydantic import Field
from starlette.types import Receive, Scope, Send

from . import keyboard
from .client import JetKVMClient, JetKVMConfig, JetKVMError
from .display import ScreenGeometry

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Operates a physical computer through a JetKVM: the server sees the computer's HDMI
output and acts as its USB keyboard and mouse. This works at every stage, including
BIOS/UEFI setup, boot menus, OS installers and lock screens, and needs no software
on the computer.

- Start with `screenshot`. Mouse coordinates are pixels in the screenshot's
  coordinate space (its size is stated next to every screenshot). Use `zoom` to read
  small text; zoom never changes the coordinate space.
- Take a screenshot after acting to confirm what happened.
- `type_text` assumes the computer uses a US keyboard layout.
- If the screenshot reports that USB is not attached, input is not reaching the computer
  (it may be off or asleep); try `power` with action "usb_wake", or power it on.
- Power actions can lose unsaved work. Confirm with the user before using them.
"""

BUTTONS = {"left": 1, "right": 2, "middle": 4, "back": 8, "forward": 16}
Button = Literal["left", "right", "middle", "back", "forward"]

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
ACTION = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

Coord = Annotated[int, Field(ge=0, description="Pixel coordinate in the screenshot's coordinate space.")]
OptCoord = Annotated[
    int | None,
    Field(ge=0, description="Pixel coordinate in screenshot space; omit to use the current pointer position."),
]


def _env_bool(env: dict[str, str] | os._Environ[str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    jetkvm: JetKVMConfig
    max_edge: int = 1920
    image_format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = 85
    typing_delay_ms: int = 20
    auto_screenshot: bool = False
    enable_raw_rpc: bool = False
    http_token: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | os._Environ[str] = os.environ) -> Settings:
        url = env.get("JETKVM_URL", "").strip()
        if not url:
            raise ValueError("JETKVM_URL is not set (e.g. JETKVM_URL=http://192.168.1.50)")
        image_format = env.get("JETKVM_SCREENSHOT_FORMAT", "jpeg").strip().lower()
        if image_format not in ("jpeg", "png"):
            raise ValueError("JETKVM_SCREENSHOT_FORMAT must be 'jpeg' or 'png'")
        return cls(
            jetkvm=JetKVMConfig(
                url=url,
                password=env.get("JETKVM_PASSWORD") or None,
                verify_tls=_env_bool(env, "JETKVM_VERIFY_TLS", True),
                connect_timeout=float(env.get("JETKVM_CONNECT_TIMEOUT", "20")),
                ice_servers=[s.strip() for s in env.get("JETKVM_ICE_SERVERS", "").split(",") if s.strip()],
            ),
            max_edge=int(env.get("JETKVM_MAX_SCREENSHOT_EDGE", "1920")),
            image_format=image_format,  # type: ignore[arg-type]
            jpeg_quality=int(env.get("JETKVM_JPEG_QUALITY", "85")),
            typing_delay_ms=int(env.get("JETKVM_TYPING_DELAY_MS", "20")),
            auto_screenshot=_env_bool(env, "JETKVM_AUTO_SCREENSHOT", False),
            enable_raw_rpc=_env_bool(env, "JETKVM_ENABLE_RAW_RPC", False),
            http_token=(env.get("JETKVM_HTTP_TOKEN") or "").strip() or None,
        )


def _tool_errors(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Report expected failures to the agent; MCPServer hides messages of other exceptions."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except (JetKVMError, ValueError) as e:
            raise ToolError(str(e)) from e

    return wrapper


def build_server(settings: Settings, client: JetKVMClient | None = None) -> MCPServer:
    client = client or JetKVMClient(settings.jetkvm)

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.close()

    server = MCPServer("jetkvm", instructions=INSTRUCTIONS, lifespan=lifespan)

    # -- helpers ---------------------------------------------------------------

    async def geometry() -> ScreenGeometry:
        session = await client.session()
        if session.frame is not None:
            width, height = session.frame.width, session.frame.height
        else:
            width, height = session.video_state.get("width"), session.video_state.get("height")
            if not width or not height:
                frame, _ = await client.latest_frame()
                if frame is None:
                    raise JetKVMError(f"screen size is unknown: {_describe_video(session.video_state)}")
                width, height = frame.width, frame.height
        return ScreenGeometry(width, height, settings.max_edge)

    def encode(image: PILImage.Image) -> Image:
        buf = io.BytesIO()
        if settings.image_format == "png":
            image.save(buf, "PNG")
        else:
            image.convert("RGB").save(buf, "JPEG", quality=settings.jpeg_quality)
        return Image(data=buf.getvalue(), format=settings.image_format)

    async def status_notes() -> list[str]:
        session = await client.session()
        notes = client.pop_notices()
        if session.video_state and (not session.video_state.get("ready", True) or session.video_state.get("error")):
            notes.append(f"Video: {_describe_video(session.video_state)}.")
        if session.usb_state and session.usb_state != "configured":
            notes.append(
                f"USB is '{session.usb_state}': keyboard/mouse input is not reaching the computer "
                "(it may be off or asleep)."
            )
        if session.led_state.get("caps_lock"):
            notes.append("Caps Lock is on.")
        return notes

    async def capture() -> list[Any]:
        frame, age = await client.latest_frame()
        session = await client.session()
        if frame is None:
            raise JetKVMError(f"no video frame received: {_describe_video(session.video_state)}")
        geo = ScreenGeometry(frame.width, frame.height, settings.max_edge)

        def render() -> Image:
            image = frame.to_image()
            if geo.scale < 1:
                image = image.resize((geo.width, geo.height), PILImage.Resampling.LANCZOS)
            return encode(image)

        image = await asyncio.to_thread(render)
        caption = f"Screen {geo.width}x{geo.height}"
        if geo.scale < 1:
            caption += f" (scaled from native {geo.native_width}x{geo.native_height})"
        caption += ". Mouse tools use this coordinate space."
        notes = await status_notes()
        if age > 3:
            notes.append(f"This frame is {age:.0f}s old; the video stream may have stalled.")
        return [" ".join([caption, *notes]), image]

    async def done(message: str) -> Any:
        if settings.auto_screenshot:
            await asyncio.sleep(0.5)
            text, image = await capture()
            return [f"{message} {text}", image]
        notes = await status_notes()
        return " ".join([message, *notes])

    async def move_to(x: int | None, y: int | None) -> None:
        if x is None and y is None:
            return
        if x is None or y is None:
            raise ValueError("give both x and y, or neither")
        geo = await geometry()
        await client.pointer(*geo.to_hid(x, y))
        await asyncio.sleep(0.03)

    def modifier_mask(modifiers: str | None) -> int:
        if not modifiers:
            return 0
        combo = keyboard.parse_combo(modifiers)
        if combo.keys:
            raise ValueError("modifiers may only contain modifier keys, e.g. 'ctrl' or 'shift+alt'")
        return combo.modifier

    # -- screen ----------------------------------------------------------------

    @server.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def screenshot(
        delay_seconds: Annotated[
            float, Field(ge=0, le=60, description="Wait this long before capturing, e.g. while something loads.")
        ] = 0,
    ) -> Any:
        """Capture the computer's screen as an image.

        All mouse coordinates are in this image's pixel space.
        """
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        return await capture()

    @server.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def zoom(
        x: Coord,
        y: Coord,
        width: Annotated[int, Field(gt=0, description="Region width in screenshot pixels.")],
        height: Annotated[int, Field(gt=0, description="Region height in screenshot pixels.")],
    ) -> Any:
        """Show a region of the screen at full native resolution, to read small text or details.

        The region is given in screenshot coordinates; mouse tools keep using full-screenshot coordinates.
        """
        frame, _ = await client.latest_frame()
        if frame is None:
            raise JetKVMError("no video frame received yet")
        geo = ScreenGeometry(frame.width, frame.height, settings.max_edge)
        box = geo.to_native_box(x, y, width, height)

        def render() -> tuple[Image, tuple[int, int]]:
            crop = frame.to_image().crop(box)
            limit = settings.max_edge
            if max(crop.size) > limit:
                crop.thumbnail((limit, limit), PILImage.Resampling.LANCZOS)
            return encode(crop), crop.size

        image, size = await asyncio.to_thread(render)
        return [
            f"Region x={x} y={y} {width}x{height} (screenshot coordinates), shown at {size[0]}x{size[1]}.",
            image,
        ]

    @server.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def get_status() -> Any:
        """Report video signal, USB attachment, keyboard LEDs, power extension and virtual media state."""
        session = await client.session()
        await session.refresh_state()

        async def optional(method: str) -> Any:
            try:
                return await client.rpc(method)
            except JetKVMError as e:
                return f"unavailable ({e})"

        extension = await optional("getActiveExtension")
        status: dict[str, Any] = {
            "device": {
                "url": settings.jetkvm.base_url,
                "firmware": session.device_version,
                "hid_transport": "hidrpc" if session.hid_rpc else "json-rpc",
            },
            "video": session.video_state,
            "usb": session.usb_state,
            "keyboard_leds": session.led_state,
            "active_extension": extension or None,
            "virtual_media": await optional("getVirtualMediaState"),
        }
        try:
            geo = await geometry()
            status["screenshot_size"] = {"width": geo.width, "height": geo.height}
        except JetKVMError:
            pass
        if extension == "atx-power":
            status["atx"] = await optional("getATXState")
        elif extension == "dc-power":
            status["dc_power"] = await optional("getDCPowerState")
        return json.dumps(status, indent=2, default=str)

    # -- mouse -----------------------------------------------------------------

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def mouse_move(x: Coord, y: Coord) -> Any:
        """Move the pointer to a screen position (held buttons stay held, so this can drag)."""
        await move_to(x, y)
        return await done(f"Moved pointer to ({x}, {y}).")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def click(
        x: OptCoord = None,
        y: OptCoord = None,
        button: Button = "left",
        count: Annotated[int, Field(ge=1, le=3, description="2 for double-click, 3 for triple-click.")] = 1,
        modifiers: Annotated[
            str | None, Field(description="Modifier keys held during the click, e.g. 'ctrl', 'shift', 'cmd+shift'.")
        ] = None,
    ) -> Any:
        """Click a mouse button, optionally at a position and with modifier keys held."""
        mask = BUTTONS[button]
        modifier = modifier_mask(modifiers)
        await move_to(x, y)
        if modifier:
            await client.keyboard_report(modifier)
            await asyncio.sleep(0.05)
        try:
            for i in range(count):
                await client.set_buttons(client.buttons | mask)
                await asyncio.sleep(0.05)
                await client.set_buttons(client.buttons & ~mask)
                if i < count - 1:
                    await asyncio.sleep(0.08)
        finally:
            if modifier:
                await asyncio.sleep(0.05)
                await client.keyboard_report(0)
        where = f" at ({x}, {y})" if x is not None else ""
        times = {1: "", 2: " (double)", 3: " (triple)"}[count]
        return await done(f"Clicked {button}{times}{where}.")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def drag(
        start_x: Coord,
        start_y: Coord,
        end_x: Coord,
        end_y: Coord,
        button: Button = "left",
        duration_ms: Annotated[int, Field(ge=50, le=10000)] = 500,
    ) -> Any:
        """Press a mouse button at the start position, move smoothly to the end, and release."""
        geo = await geometry()
        sx, sy = geo.to_hid(start_x, start_y)
        ex, ey = geo.to_hid(end_x, end_y)
        mask = BUTTONS[button]
        await client.pointer(sx, sy)
        await asyncio.sleep(0.05)
        await client.set_buttons(client.buttons | mask)
        await asyncio.sleep(0.1)
        try:
            steps = max(5, duration_ms // 20)
            for i in range(1, steps + 1):
                t = i / steps
                await client.pointer(round(sx + (ex - sx) * t), round(sy + (ey - sy) * t))
                await asyncio.sleep(duration_ms / 1000 / steps)
            await asyncio.sleep(0.1)
        finally:
            await client.set_buttons(client.buttons & ~mask)
        return await done(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y}).")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def mouse_button(action: Literal["down", "up"], button: Button = "left") -> Any:
        """Press or release a mouse button without moving, for custom drags or long presses."""
        mask = BUTTONS[button]
        await client.set_buttons(client.buttons | mask if action == "down" else client.buttons & ~mask)
        return await done(f"Mouse {button} {action}.")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def scroll(
        direction: Literal["up", "down", "left", "right"],
        amount: Annotated[int, Field(ge=1, le=50, description="Number of wheel notches.")] = 3,
        x: OptCoord = None,
        y: OptCoord = None,
    ) -> Any:
        """Scroll the mouse wheel, optionally after moving the pointer over the area to scroll."""
        await move_to(x, y)
        dy, dx = {"up": (1, 0), "down": (-1, 0), "left": (0, -1), "right": (0, 1)}[direction]
        for _ in range(amount):
            await client.wheel(dy, dx)
            await asyncio.sleep(0.03)
        return await done(f"Scrolled {direction} {amount}.")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def mouse_move_relative(
        dx: Annotated[int, Field(ge=-4000, le=4000, description="Horizontal movement in mouse units.")],
        dy: Annotated[int, Field(ge=-4000, le=4000, description="Vertical movement in mouse units.")],
    ) -> Any:
        """Move the pointer relatively, like a physical mouse.

        Only for environments that ignore the absolute pointer (some BIOS setups, games);
        prefer mouse_move otherwise. Distance depends on the computer's pointer acceleration.
        """
        await client.relative_move(dx, dy)
        return await done(f"Moved pointer by ({dx}, {dy}).")

    # -- keyboard --------------------------------------------------------------

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def type_text(
        text: Annotated[str, Field(min_length=1, max_length=20000)],
        delay_ms: Annotated[
            int | None, Field(ge=1, le=1000, description="Per-key press and release time. Raise for slow targets.")
        ] = None,
    ) -> Any:
        """Type text as keystrokes (US keyboard layout). Newlines press Enter, tabs press Tab.

        Typing takes about 2 x delay_ms per character (40 ms by default).
        """
        session = await client.session()
        steps = keyboard.text_to_steps(
            text, delay_ms or settings.typing_delay_ms, caps_lock=bool(session.led_state.get("caps_lock"))
        )
        await client.run_macro(steps)
        return await done(f"Typed {len(text)} characters.")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def press_keys(
        keys: Annotated[
            str,
            Field(
                description=(
                    "Key chord(s) separated by spaces, pressed in order. Chords join keys with '+'. "
                    "Examples: 'enter', 'ctrl+c', 'cmd+space', 'ctrl+alt+delete', 'alt+tab', "
                    "'shift+f10', 'down down enter'. Names: a-z, 0-9, f1-f24, enter, esc, tab, space, "
                    "backspace, delete, insert, home, end, pageup, pagedown, up, down, left, right, "
                    "capslock, printscreen, menu, kp0-kp9, symbols like '/' or '?', and modifiers "
                    "ctrl, shift, alt/option, cmd/win/super (prefix r for right-hand, e.g. ralt)."
                )
            ),
        ],
        repeat: Annotated[int, Field(ge=1, le=100, description="Press the whole sequence this many times.")] = 1,
        interval_ms: Annotated[int, Field(ge=10, le=5000, description="Pause after each chord.")] = 50,
    ) -> Any:
        """Press keys or keyboard shortcuts."""
        combos = keyboard.parse_sequence(keys)
        steps: list[keyboard.MacroStep] = []
        for _ in range(repeat):
            for combo in combos:
                steps += keyboard.combo_to_steps(combo, hold_ms=50, release_ms=interval_ms)
        await client.run_macro(steps)
        return await done(f"Pressed {keys}" + (f" x{repeat}" if repeat > 1 else "") + ".")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def hold_keys(
        keys: Annotated[str, Field(description="One chord to hold, e.g. 'shift' or 'f12' or 'ctrl+alt'.")],
        duration_ms: Annotated[int, Field(ge=50, le=60000)],
    ) -> Any:
        """Hold a key chord down for a duration, then release it (e.g. hold a key during boot)."""
        combo = keyboard.parse_combo(keys)
        steps = keyboard.combo_to_steps(combo, hold_ms=duration_ms, release_ms=20)
        await client.run_macro(steps)
        return await done(f"Held {keys} for {duration_ms} ms.")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def release_all() -> Any:
        """Release every key and mouse button and stop any typing in progress."""
        await client.release_all()
        return await done("Released all keys and buttons.")

    # -- power -----------------------------------------------------------------

    @server.tool(annotations=DESTRUCTIVE, structured_output=False)
    @_tool_errors
    async def power(
        action: Annotated[
            Literal["power_button", "power_button_long", "reset_button", "dc_on", "dc_off", "usb_wake"],
            Field(
                description=(
                    "power_button: short press (turn on, or ask the OS to shut down). "
                    "power_button_long: 5 s press (force off). reset_button: hard reset. "
                    "These three need the JetKVM ATX extension. dc_on/dc_off need the DC power extension. "
                    "usb_wake: send a USB remote-wakeup to a sleeping computer (no extension needed)."
                )
            ),
        ],
    ) -> Any:
        """Control the computer's power. Force-off and reset lose unsaved work: confirm with the user first."""
        if action == "usb_wake":
            await client.rpc("wakeHost")
            return await done("Sent USB wake-up.")
        extension = await client.rpc("getActiveExtension")
        if action in ("dc_on", "dc_off"):
            if extension != "dc-power":
                raise JetKVMError(f"the DC power extension is not active (active extension: {extension or 'none'})")
            await client.rpc("setDCPowerState", {"enabled": action == "dc_on"})
            return await done(f"DC power {'on' if action == 'dc_on' else 'off'}.")
        if extension != "atx-power":
            raise JetKVMError(f"the ATX power extension is not active (active extension: {extension or 'none'})")
        atx_action = {"power_button": "power-short", "power_button_long": "power-long", "reset_button": "reset"}
        await client.rpc("setATXPowerAction", {"action": atx_action[action]}, timeout=20)
        return await done(f"Pressed {action.replace('_', ' ')}.")

    @server.tool(annotations=ACTION, structured_output=False)
    @_tool_errors
    async def wake_on_lan(
        mac_address: Annotated[str, Field(description="The computer's network MAC address, e.g. 'AA:BB:CC:DD:EE:FF'.")],
        broadcast_ip: Annotated[str | None, Field(description="Optional broadcast address for the packet.")] = None,
    ) -> Any:
        """Send a Wake-on-LAN magic packet from the JetKVM's network."""
        params: dict[str, Any] = {"macAddress": mac_address}
        if broadcast_ip:
            params["broadcastIP"] = broadcast_ip
        await client.rpc("sendWOLMagicPacket", params)
        return await done(f"Sent Wake-on-LAN to {mac_address}.")

    # -- virtual media ---------------------------------------------------------

    @server.tool(annotations=READ_ONLY, structured_output=False)
    @_tool_errors
    async def list_virtual_media() -> Any:
        """List disk images stored on the JetKVM and what is currently mounted."""
        result = {
            "mounted": await client.rpc("getVirtualMediaState"),
            "stored_images": await client.rpc("listStorageFiles"),
            "storage_space": await client.rpc("getStorageSpace"),
        }
        return json.dumps(result, indent=2, default=str)

    @server.tool(annotations=DESTRUCTIVE, structured_output=False)
    @_tool_errors
    async def mount_virtual_media(
        source: Annotated[
            str, Field(description="An http(s) URL of an image, or the filename of an image stored on the JetKVM.")
        ],
        mode: Annotated[
            Literal["CDROM", "Disk"], Field(description="CDROM for .iso installers, Disk for writable USB-drive images.")
        ] = "CDROM",
    ) -> Any:
        """Attach a disk image to the computer as a USB CD-ROM or drive (e.g. to boot an OS installer)."""
        if source.startswith(("http://", "https://")):
            await client.rpc("mountWithHTTP", {"url": source, "mode": mode}, timeout=60)
        else:
            await client.rpc("mountWithStorage", {"filename": source, "mode": mode}, timeout=60)
        return await done(f"Mounted {source} as {mode}.")

    @server.tool(annotations=DESTRUCTIVE, structured_output=False)
    @_tool_errors
    async def unmount_virtual_media() -> Any:
        """Detach the currently mounted virtual media."""
        await client.rpc("unmountImage", timeout=30)
        return await done("Unmounted virtual media.")

    # -- escape hatch ----------------------------------------------------------

    if settings.enable_raw_rpc:

        @server.tool(annotations=DESTRUCTIVE, structured_output=False)
        @_tool_errors
        async def device_rpc(
            method: Annotated[str, Field(description="JetKVM JSON-RPC method name, e.g. 'getNetworkState'.")],
            params: dict[str, Any] | None = None,
        ) -> Any:
            """Call any JetKVM JSON-RPC method directly. Can change device settings; use with care."""
            return json.dumps(await client.rpc(method, params, timeout=60), indent=2, default=str)

    return server


_MIN_HTTP_TOKEN_LEN = 16
_HTTP_TOKEN_EXTRA = set("-._~")


def validate_http_token(token: str | None) -> str:
    """Reject tokens that are short or unsafe to put in a URL path."""
    if (
        not token
        or len(token) < _MIN_HTTP_TOKEN_LEN
        or any(not (char.isascii() and (char.isalnum() or char in _HTTP_TOKEN_EXTRA)) for char in token)
    ):
        raise ValueError(
            "JETKVM_HTTP_TOKEN must be at least 16 characters of letters, digits, and -._~ "
            "(for example: openssl rand -hex 32)"
        )
    return token


def _tokens_match(presented: str, expected: str) -> bool:
    if not presented or len(presented) > 512:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def _presented_secret(headers: list[tuple[bytes, bytes]]) -> str | None:
    """Token from `Authorization`. Grok sends the raw token or a Bearer scheme."""
    for key, value in headers:
        if key.lower() != b"authorization":
            continue
        text = value.decode("latin-1").strip()
        if text.lower().startswith("bearer "):
            text = text[7:].strip()
        return text or None
    return None


class HttpAuth:
    """Bearer check in front of the Streamable HTTP app.

    Grok connectors that can only store a URL may pass the same token as
    `/mcp/<token>`. Host-header checks stay off: a tunnel's Host is not the
    bind address, and every MCP request still needs the token.
    """

    def __init__(self, app: Any, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        if scope.get("method") == "GET" and scope.get("path") == "/healthz":
            await _plain(send, 200, "ok\n")
            return

        headers = list(scope.get("headers") or [])
        path = scope.get("path") or ""
        if path.startswith("/mcp/"):
            segment = unquote(path.removeprefix("/mcp/").split("/", 1)[0])
            if _tokens_match(segment, self._token):
                rewritten = dict(scope)
                rewritten["path"] = "/mcp"
                rewritten["raw_path"] = b"/mcp"
                rewritten["headers"] = [
                    (key, value) for key, value in headers if key.lower() != b"authorization"
                ]
                rewritten["headers"].append((b"authorization", b"Bearer " + self._token.encode("utf-8")))
                await self._app(rewritten, receive, send)
                return

        presented = _presented_secret(headers)
        if presented is None or not _tokens_match(presented, self._token):
            await _plain(send, 401, "Unauthorized\n", www_authenticate=b"Bearer")
            return
        await self._app(scope, receive, send)


async def _plain(send: Send, status: int, body: str, www_authenticate: bytes | None = None) -> None:
    payload = body.encode()
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"content-length", str(len(payload)).encode()),
    ]
    if www_authenticate is not None:
        headers.append((b"www-authenticate", www_authenticate))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


def build_http_app(settings: Settings) -> HttpAuth:
    """Streamable HTTP app for Grok and Grok Bot. Requires `settings.http_token`."""
    token = validate_http_token(settings.http_token)
    app = build_server(settings).streamable_http_app(
        json_response=True,
        stateless_http=True,
        # A public tunnel presents its own Host. The token is what admits a caller.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return HttpAuth(app, token)


def serve_http(settings: Settings, host: str, port: int) -> None:
    import uvicorn

    app = build_http_app(settings)
    print(f"JetKVM MCP listening on http://{host}:{port}/mcp", file=sys.stderr)
    print(
        "Send Authorization: Bearer $JETKVM_HTTP_TOKEN. "
        "Clients that can only store a URL may use /mcp/$JETKVM_HTTP_TOKEN — that path is a secret.",
        file=sys.stderr,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


def _describe_video(state: dict[str, Any]) -> str:
    if not state:
        return "video state unknown"
    error = state.get("error")
    meanings = {
        "no_signal": "no HDMI signal (the computer may be off, asleep, or the cable unplugged)",
        "no_lock": "HDMI signal present but not locked",
        "out_of_range": "HDMI signal out of supported range",
    }
    if error:
        return meanings.get(error, error)
    if not state.get("ready", True):
        return "video not ready"
    return f"{state.get('width')}x{state.get('height')} at {state.get('fps', 0):.0f} fps"


async def _check(settings: Settings, output: str) -> int:
    """Connect once, print status, save a screenshot. For setup troubleshooting."""
    client = JetKVMClient(settings.jetkvm)
    try:
        session = await client.session()
        print(f"Connected to {settings.jetkvm.base_url} (firmware {session.device_version})")
        print(f"HID transport: {'hidrpc' if session.hid_rpc else 'json-rpc'}")
        print(f"Video: {_describe_video(session.video_state)}")
        print(f"USB: {session.usb_state}")
        print(f"Keyboard LEDs: {session.led_state}")
        frame, age = await client.latest_frame(fresh_timeout=3)
        if frame is None:
            print("No video frame received.")
            return 1
        await asyncio.to_thread(frame.to_image().save, output)
        print(f"Saved {frame.width}x{frame.height} screenshot to {output} (frame age {age:.1f}s)")
        return 0
    except JetKVMError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jetkvm-mcp", description="MCP server for controlling a computer via JetKVM.")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the MCP server (stdio by default)")
    serve.add_argument(
        "--http",
        action="store_true",
        help="Streamable HTTP on /mcp for Grok and Grok Bot (requires JETKVM_HTTP_TOKEN)",
    )
    serve.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    check = sub.add_parser("check", help="test the connection and save a screenshot")
    check.add_argument("-o", "--output", default="jetkvm-screenshot.png")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("JETKVM_LOG_LEVEL", "WARNING").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.from_env()
    except ValueError as e:
        parser.error(str(e))

    if args.command == "check":
        sys.exit(asyncio.run(_check(settings, args.output)))
    if getattr(args, "http", False):
        try:
            validate_http_token(settings.http_token)
        except ValueError as e:
            parser.error(str(e))
        serve_http(settings, args.host, args.port)
        return
    build_server(settings).run("stdio")


if __name__ == "__main__":
    main()

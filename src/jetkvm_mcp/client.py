"""Connection to a JetKVM: HTTP auth, WebSocket signaling, WebRTC video and data channels.

Protocol notes (from the JetKVM firmware, github.com/jetkvm/kvm):

* Auth: POST /auth/login-local {"password"} sets an `authToken` cookie. The device
  keeps a single token, so logging in signs out other browsers on their next request.
* Signaling: WebSocket at /webrtc/signaling/client. The device sends
  {"type":"device-metadata"}; the client sends {"type":"offer","data":{"sd":base64(json(offer))}};
  the device answers {"type":"answer","data":base64(json(answer))} and trickles
  {"type":"new-ice-candidate","data":RTCIceCandidateInit}, sometimes before the answer.
* Data channels: "rpc" carries JSON-RPC 2.0 (each request is handled in its own
  goroutine, so ordering is only guaranteed if callers await each response);
  "hidrpc" carries binary HID reports processed in order (see hidrpc.py).
* Only one WebRTC session is active at a time; a new one kicks the old one with an
  "otherSessionConnected" event.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.jitterbuffer import JitterBuffer
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp

from . import hidrpc
from .keyboard import LED_CAPS_LOCK, LED_NUM_LOCK, LED_SCROLL_LOCK, MacroStep

log = logging.getLogger(__name__)

# aiortc's default video jitter buffer holds 128 RTP packets, which is smaller than a
# single 1080p keyframe from the JetKVM encoder. The device ignores keyframe requests
# (PLI), so a keyframe that doesn't fit would never decode. Must be a power of two.
VIDEO_JITTER_CAPACITY = 2048

HANDSHAKE_TIMEOUT = 3.0


class JetKVMError(Exception):
    pass


class AuthError(JetKVMError):
    pass


class RPCError(JetKVMError):
    def __init__(self, method: str, error: Any):
        self.error = error
        message = (error.get("data") or error.get("message")) if isinstance(error, dict) else error
        super().__init__(f"{method} failed: {message}")


@dataclass
class JetKVMConfig:
    url: str
    password: str | None = None
    verify_tls: bool = True
    connect_timeout: float = 20.0
    ice_servers: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        url = self.url.strip().rstrip("/")
        if "://" not in url:
            url = "http://" + url
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"invalid JetKVM URL: {self.url!r}")
        return f"{parts.scheme}://{parts.netloc}"

    @property
    def signaling_url(self) -> str:
        base = self.base_url
        return ("wss" + base[5:] if base.startswith("https") else "ws" + base[4:]) + "/webrtc/signaling/client"

    @property
    def ssl(self) -> bool:
        return self.verify_tls


def decode_led_byte(raw: int) -> dict[str, bool]:
    return {
        "num_lock": bool(raw & LED_NUM_LOCK),
        "caps_lock": bool(raw & LED_CAPS_LOCK),
        "scroll_lock": bool(raw & LED_SCROLL_LOCK),
    }


def _enlarge_jitter_buffer(receiver: Any) -> None:
    attr = "_RTCRtpReceiver__jitter_buffer"
    if hasattr(receiver, attr):
        setattr(receiver, attr, JitterBuffer(capacity=VIDEO_JITTER_CAPACITY, is_video=True))
    else:
        log.warning("could not enlarge aiortc's video jitter buffer; large keyframes may be dropped")


class Session:
    """One WebRTC session with the device. Replaced wholesale on reconnect."""

    def __init__(self, config: JetKVMConfig):
        self.config = config
        self.pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(urls=u) for u in config.ice_servers])
        )
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.device_version: str | None = None
        self.hid_rpc = False
        self.kicked = False
        self.close_reason: str | None = None
        self.closed = asyncio.Event()

        self.video_state: dict[str, Any] = {}
        self.led_state: dict[str, bool] = {}
        self.usb_state: str | None = None
        self.events: dict[str, Any] = {}

        self.frame: Any = None  # latest decoded av.VideoFrame
        self.frame_time = 0.0
        self._frame_cond = asyncio.Condition()

        self._pending: dict[int, tuple[str, asyncio.Future[Any]]] = {}
        self._next_id = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._handshake = asyncio.Event()
        self._macro_done = asyncio.Event()

        self._video = self.pc.addTransceiver("video", direction="recvonly")
        _enlarge_jitter_buffer(self._video.receiver)
        self.rpc_channel = self.pc.createDataChannel("rpc")
        self.hid_channel = self.pc.createDataChannel("hidrpc")
        self._rpc_open = asyncio.Event()
        self._hid_open = asyncio.Event()
        self.rpc_channel.on("open", self._rpc_open.set)
        self.rpc_channel.on("message", self._on_rpc_message)
        self.rpc_channel.on("close", lambda: self._mark_closed("the rpc data channel closed"))
        self.hid_channel.on("open", self._hid_open.set)
        self.hid_channel.on("message", self._on_hid_message)
        self.pc.on("track", self._on_track)
        self.pc.on("connectionstatechange", self._on_connection_state)

    # -- lifecycle -----------------------------------------------------------

    async def open(self, http: aiohttp.ClientSession) -> None:
        await self.pc.setLocalDescription(await self.pc.createOffer())
        try:
            self.ws = await http.ws_connect(self.config.signaling_url, ssl=self.config.ssl, max_msg_size=0)
        except aiohttp.WSServerHandshakeError as e:
            if e.status == 401:
                raise AuthError("the device rejected our credentials") from e
            raise JetKVMError(f"signaling WebSocket failed with HTTP {e.status}") from e

        local = self.pc.localDescription
        sd = base64.b64encode(json.dumps({"type": local.type, "sdp": local.sdp}).encode()).decode()
        await self.ws.send_json({"type": "offer", "data": {"sd": sd}})

        answered: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._spawn(self._read_signaling(answered))
        await answered
        await self._rpc_open.wait()
        await self._hid_open.wait()

        self.hid_channel.send(hidrpc.handshake())
        try:
            await asyncio.wait_for(self._handshake.wait(), HANDSHAKE_TIMEOUT)
            self.hid_rpc = True
        except TimeoutError:
            log.warning("device did not answer the hidrpc handshake; falling back to JSON-RPC for HID")

        await self.refresh_state()

    async def refresh_state(self) -> None:
        try:
            self.video_state = await self.rpc("getVideoState") or {}
            self.led_state = await self.rpc("getKeyboardLedState") or {}
            self.usb_state = await self.rpc("getUSBState")
        except JetKVMError as e:
            log.warning("could not read initial device state: %s", e)

    async def close(self) -> None:
        self._mark_closed("closed by client")
        for task in list(self._tasks):
            task.cancel()
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
        await self.pc.close()

    def _mark_closed(self, reason: str) -> None:
        if self.closed.is_set():
            return
        self.close_reason = reason
        self.closed.set()
        for _, fut in self._pending.values():
            if not fut.done():
                fut.set_exception(JetKVMError(f"connection lost: {reason}"))

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_connection_state(self) -> None:
        state = self.pc.connectionState
        log.debug("peer connection state: %s", state)
        if state in ("failed", "closed"):
            self._mark_closed(f"WebRTC connection {state}")

    # -- signaling -----------------------------------------------------------

    async def _read_signaling(self, answered: asyncio.Future[None]) -> None:
        early_candidates: list[dict[str, Any]] = []
        try:
            assert self.ws is not None
            async for msg in self.ws:
                if msg.type != aiohttp.WSMsgType.TEXT or msg.data == "pong":
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "device-metadata":
                    self.device_version = (data.get("data") or {}).get("deviceVersion")
                elif kind == "answer":
                    answer = json.loads(base64.b64decode(data["data"]))
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
                    )
                    for candidate in early_candidates:
                        await self._add_remote_candidate(candidate)
                    early_candidates.clear()
                    if not answered.done():
                        answered.set_result(None)
                elif kind == "new-ice-candidate":
                    candidate = data.get("data") or {}
                    if self.pc.remoteDescription is None:
                        early_candidates.append(candidate)
                    else:
                        await self._add_remote_candidate(candidate)
                elif "error" in data and not answered.done():
                    answered.set_exception(JetKVMError(f"device rejected the session: {data['error']}"))
        except Exception as e:  # noqa: BLE001 - surfaced through `answered` or logged
            if not answered.done():
                answered.set_exception(e)
            else:
                log.debug("signaling reader stopped: %s", e)
        finally:
            if not answered.done():
                answered.set_exception(JetKVMError("signaling closed before the device answered"))

    async def _add_remote_candidate(self, init: dict[str, Any]) -> None:
        text = init.get("candidate") or ""
        if not text:
            return
        try:
            candidate = candidate_from_sdp(text.removeprefix("candidate:"))
        except Exception:  # noqa: BLE001 - malformed or unsupported candidate types are skipped
            log.debug("ignoring unparsable ICE candidate %r", text)
            return
        # The device (pion) bundles everything on one transport, owned by the first
        # m-line (our video transceiver). Its sdpMid is sometimes empty, so pin it.
        candidate.sdpMid = self._video.mid
        candidate.sdpMLineIndex = 0
        try:
            await self.pc.addIceCandidate(candidate)
        except Exception as e:  # noqa: BLE001
            log.debug("could not add ICE candidate %r: %s", text, e)

    # -- data channels -------------------------------------------------------

    def _on_rpc_message(self, message: str | bytes) -> None:
        try:
            data = json.loads(message)
        except ValueError:
            return
        if "id" in data and "method" not in data:
            pending = self._pending.get(data["id"])
            if pending is not None and not pending[1].done():
                method, fut = pending
                if data.get("error"):
                    fut.set_exception(RPCError(method, data["error"]))
                else:
                    # The device omits "result" when it is a zero value (Go omitempty).
                    fut.set_result(data.get("result"))
            return
        method = data.get("method")
        if not method:
            return
        params = data.get("params")
        self.events[method] = params
        if method == "videoInputState" and isinstance(params, dict):
            self.video_state = params
        elif method == "keyboardLedState" and isinstance(params, dict):
            self.led_state = params
        elif method == "usbState":
            self.usb_state = params
        elif method == "otherSessionConnected":
            self.kicked = True
            self._mark_closed("another JetKVM session (e.g. a browser tab) took over")

    def _on_hid_message(self, message: str | bytes) -> None:
        if isinstance(message, str):
            return
        try:
            msg = hidrpc.parse(message)
        except ValueError:
            return
        if msg.type == hidrpc.TYPE_HANDSHAKE:
            self._handshake.set()
        elif msg.type == hidrpc.TYPE_KEYBOARD_LED_STATE and msg.payload:
            self.led_state = decode_led_byte(msg.payload[0])
        elif msg.type == hidrpc.TYPE_KEYBOARD_MACRO_STATE and msg.payload:
            if msg.payload[0] == 0:
                self._macro_done.set()

    def _on_track(self, track: Any) -> None:
        if track.kind == "video":
            self._spawn(self._read_video(track))

    async def _read_video(self, track: Any) -> None:
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            async with self._frame_cond:
                self.frame = frame
                self.frame_time = time.monotonic()
                self._frame_cond.notify_all()

    async def wait_for_frame(self, newer_than: float, timeout: float) -> bool:
        """Wait for a frame received after `newer_than` (monotonic). False on timeout."""
        async with self._frame_cond:
            try:
                await asyncio.wait_for(
                    self._frame_cond.wait_for(lambda: self.frame is not None and self.frame_time > newer_than),
                    timeout,
                )
                return True
            except TimeoutError:
                return False

    # -- requests ------------------------------------------------------------

    def _check_open(self) -> None:
        if self.closed.is_set():
            raise JetKVMError(f"connection lost: {self.close_reason}")

    async def rpc(self, method: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
        self._check_open()
        self._next_id += 1
        request_id = self._next_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (method, fut)
        try:
            self.rpc_channel.send(
                json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}, "id": request_id})
            )
            try:
                return await asyncio.wait_for(fut, timeout)
            except TimeoutError:
                raise JetKVMError(f"{method} timed out after {timeout:.0f}s") from None
        finally:
            self._pending.pop(request_id, None)

    def send_hid(self, data: bytes) -> None:
        self._check_open()
        self.hid_channel.send(data)

    async def run_macro(self, steps: list[MacroStep]) -> None:
        if not self.hid_rpc:
            for step in steps:
                await self.rpc("keyboardReport", {"modifier": step.modifier, "keys": list(step.keys)})
                await asyncio.sleep(step.delay_ms / 1000)
            return
        for chunk in hidrpc.chunk_steps(steps):
            self._macro_done.clear()
            self.send_hid(hidrpc.keyboard_macro(chunk))
            budget = sum(s.delay_ms for s in chunk) / 1000 + 5.0
            try:
                await asyncio.wait_for(self._macro_done.wait(), budget)
            except BaseException as e:
                # Timed out or the tool call was cancelled: stop the device typing.
                if not self.closed.is_set():
                    try:
                        self.hid_channel.send(hidrpc.cancel_keyboard_macro())
                    except Exception:  # noqa: BLE001
                        pass
                if isinstance(e, TimeoutError):
                    raise JetKVMError("the device did not report the keyboard macro as finished") from None
                raise
            self._check_open()


class JetKVMClient:
    """Lazily connected, self-healing handle to one JetKVM device."""

    def __init__(self, config: JetKVMConfig):
        self.config = config
        self._http: aiohttp.ClientSession | None = None
        self._session: Session | None = None
        self._lock = asyncio.Lock()
        self._notices: list[str] = []
        # Pointer state survives reconnects so held buttons and position stay consistent.
        self.pointer_pos = (16384, 16384)
        self.buttons = 0

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._http is not None:
            await self._http.close()
            self._http = None

    def pop_notices(self) -> list[str]:
        notices, self._notices = self._notices, []
        return notices

    async def session(self) -> Session:
        async with self._lock:
            current = self._session
            if current is not None and not current.closed.is_set():
                return current
            if current is not None:
                if current.kicked:
                    self._notices.append(
                        "Another JetKVM session (e.g. a browser tab) had taken over; "
                        "it was disconnected so this one could reconnect."
                    )
                await current.close()
                self._session = None
            self._session = await self._connect()
            return self._session

    async def _http_session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            # unsafe=True: accept cookies from bare IP addresses, which is how most JetKVMs are reached.
            self._http = aiohttp.ClientSession(
                cookie_jar=aiohttp.CookieJar(unsafe=True),
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._http

    async def _ensure_auth(self, http: aiohttp.ClientSession) -> None:
        base = self.config.base_url
        try:
            async with http.get(f"{base}/device", ssl=self.config.ssl) as resp:
                if resp.status == 200:
                    return
                if resp.status != 401:
                    raise JetKVMError(f"unexpected HTTP {resp.status} from {base}/device")
        except aiohttp.ClientError as e:
            raise JetKVMError(f"cannot reach JetKVM at {base}: {e}") from e

        async with http.get(f"{base}/device/status", ssl=self.config.ssl) as resp:
            if resp.status == 200 and not (await resp.json()).get("isSetup", True):
                raise JetKVMError("the JetKVM has not been set up yet; finish onboarding in its web UI first")
        if not self.config.password:
            raise AuthError("the JetKVM is password protected; set JETKVM_PASSWORD")
        async with http.post(f"{base}/auth/login-local", json={"password": self.config.password}, ssl=self.config.ssl) as resp:
            if resp.status == 200:
                return
            if resp.status == 401:
                raise AuthError("the JetKVM rejected the password in JETKVM_PASSWORD")
            if resp.status == 429:
                raise AuthError(f"too many failed logins; retry in {resp.headers.get('Retry-After', '?')}s")
            raise AuthError(f"login failed with HTTP {resp.status}: {await resp.text()}")

    async def _connect(self) -> Session:
        http = await self._http_session()
        async with asyncio.timeout(self.config.connect_timeout):
            await self._ensure_auth(http)
        session = Session(self.config)
        try:
            async with asyncio.timeout(self.config.connect_timeout):
                await session.open(http)
        except TimeoutError:
            await session.close()
            raise JetKVMError(
                f"timed out connecting to the JetKVM after {self.config.connect_timeout:.0f}s "
                "(is UDP between this machine and the device blocked?)"
            ) from None
        except BaseException:
            await session.close()
            raise
        log.info("connected to JetKVM %s (firmware %s, hidrpc=%s)", self.config.base_url, session.device_version, session.hid_rpc)
        return session

    # -- HID -----------------------------------------------------------------

    async def keyboard_report(self, modifier: int, keys: tuple[int, ...] | list[int] = ()) -> None:
        s = await self.session()
        if s.hid_rpc:
            s.send_hid(hidrpc.keyboard_report(modifier, keys))
        else:
            await s.rpc("keyboardReport", {"modifier": modifier, "keys": list(keys)})

    async def run_macro(self, steps: list[MacroStep]) -> None:
        s = await self.session()
        await s.run_macro(steps)

    async def pointer(self, x: int, y: int, buttons: int | None = None) -> None:
        if buttons is not None:
            self.buttons = buttons
        self.pointer_pos = (x, y)
        s = await self.session()
        if s.hid_rpc:
            s.send_hid(hidrpc.pointer_report(x, y, self.buttons))
        else:
            await s.rpc("absMouseReport", {"x": x, "y": y, "buttons": self.buttons})

    async def set_buttons(self, buttons: int) -> None:
        await self.pointer(*self.pointer_pos, buttons=buttons)

    async def relative_move(self, dx: int, dy: int) -> None:
        s = await self.session()
        # Relative reports carry signed bytes; split larger moves into steps.
        while dx or dy:
            step_x = max(-127, min(127, dx))
            step_y = max(-127, min(127, dy))
            if s.hid_rpc:
                s.send_hid(hidrpc.mouse_report(step_x, step_y, self.buttons))
            else:
                await s.rpc("relMouseReport", {"dx": step_x, "dy": step_y, "buttons": self.buttons})
            dx -= step_x
            dy -= step_y
            await asyncio.sleep(0.005)

    async def wheel(self, dy: int, dx: int = 0) -> None:
        s = await self.session()
        await s.rpc("wheelReport", {"wheelY": dy, "wheelX": dx})

    async def release_all(self) -> None:
        s = await self.session()
        if s.hid_rpc:
            s.send_hid(hidrpc.cancel_keyboard_macro())
        await self.keyboard_report(0)
        await self.set_buttons(0)

    # -- device --------------------------------------------------------------

    async def rpc(self, method: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
        s = await self.session()
        return await s.rpc(method, params, timeout)

    async def latest_frame(self, fresh_timeout: float = 1.5) -> tuple[Any, float]:
        """Return (frame, age_seconds), preferring a frame newer than this call.

        Falls back to the most recent frame if none arrives within `fresh_timeout`
        (the device may send few frames for a static screen).
        """
        s = await self.session()
        requested = time.monotonic()
        if not await s.wait_for_frame(requested, fresh_timeout) and s.frame is None:
            # A brand-new session needs a keyframe first (the encoder sends one every ~0.5s).
            await s.wait_for_frame(0, 5.0)
        if s.frame is None:
            return None, 0.0
        return s.frame, max(0.0, time.monotonic() - s.frame_time)

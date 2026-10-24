"""A simulated JetKVM for end-to-end tests.

Implements the device side of the protocol the way the firmware does
(github.com/jetkvm/kvm: web.go, cloud.go, webrtc.go, jsonrpc.go, hidrpc.go):

* cookie auth with a single rotating token,
* WebSocket signaling where ICE candidates are trickled *before* the answer and
  carry an empty sdpMid, as pion does,
* an H.264-only video track,
* JSON-RPC on "rpc" with Go's omitempty results and strict parameter checks,
* binary hidrpc on "hidrpc": handshake, reports, and device-side keyboard macros
  executed sequentially with macro-state notifications,
* session takeover with an "otherSessionConnected" event.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import uuid
from typing import Any

import av
from aiohttp import WSMsgType, web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from PIL import Image, ImageDraw

RED_SQUARE = (100, 50, 200, 150)  # left, top, right, bottom in native pixels


class PatternTrack(VideoStreamTrack):
    """A dark screen with a red square at RED_SQUARE, or random noise (huge keyframes)."""

    def __init__(self, width: int, height: int, noise: bool = False):
        super().__init__()
        if noise:
            self._image = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
            return
        image = Image.new("RGB", (width, height), (20, 20, 40))
        left, top, right, bottom = RED_SQUARE
        ImageDraw.Draw(image).rectangle([left, top, right - 1, bottom - 1], fill=(220, 30, 30))
        self._image = image

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = av.VideoFrame.from_image(self._image)
        frame.pts = pts
        frame.time_base = time_base
        return frame


class FakeJetKVM:
    def __init__(
        self,
        password: str | None = "hunter2",
        width: int = 1280,
        height: int = 720,
        hidrpc: bool = True,
        noise: bool = False,
    ):
        self.password = password
        self.width = width
        self.height = height
        self.hidrpc = hidrpc
        self.noise = noise
        self.extension = ""
        self.led_state = {"num_lock": True, "caps_lock": False, "scroll_lock": False}

        self.token: str | None = None
        self.logins = 0
        self.sessions: list[DeviceSession] = []
        self.current: DeviceSession | None = None

        # What the "host computer" received.
        self.keyboard: list[tuple[int, tuple[int, ...]]] = []
        self.pointer: list[tuple[int, int, int]] = []
        self.relative: list[tuple[int, int, int]] = []
        self.wheel: list[tuple[int, int]] = []
        self.atx_actions: list[str] = []
        self.mounted: dict[str, str] | None = None
        self.rpc_log: list[str] = []
        self._runner: web.AppRunner | None = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/auth/login-local", self._login)
        app.router.add_get("/device/status", self._status)
        app.router.add_get("/device", self._device)
        app.router.add_get("/webrtc/signaling/client", self._signaling)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        for session in self.sessions:
            await session.close()
        if self._runner is not None:
            await self._runner.cleanup()

    # -- HTTP --------------------------------------------------------------------

    def _authorized(self, request: web.Request) -> bool:
        if self.password is None:
            return True
        return self.token is not None and request.cookies.get("authToken") == self.token

    async def _login(self, request: web.Request) -> web.Response:
        if self.password is None:
            return web.json_response({"error": "Login is disabled in noPassword mode"}, status=400)
        body = await request.json()
        if body.get("password") != self.password:
            return web.json_response({"error": "Invalid password"}, status=401)
        self.logins += 1
        self.token = str(uuid.uuid4())
        response = web.json_response({"message": "Login successful"})
        response.set_cookie("authToken", self.token, path="/", httponly=True)
        return response

    async def _status(self, request: web.Request) -> web.Response:
        return web.json_response({"isSetup": True})

    async def _device(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        mode = "noPassword" if self.password is None else "password"
        return web.json_response({"authMode": mode, "deviceId": "fake-device"})

    async def _signaling(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "device-metadata", "data": {"deviceVersion": "0.0.0-fake"}})
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            if msg.data == "ping":
                await ws.send_str("pong")
                continue
            data = json.loads(msg.data)
            if data.get("type") == "offer":
                await self._new_session(ws, data["data"]["sd"])
        return ws

    async def _new_session(self, ws: web.WebSocketResponse, sd: str) -> None:
        offer = json.loads(base64.b64decode(sd))
        # No ICE servers: aiortc's default is a public STUN server, which only adds delay here.
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        session = DeviceSession(self, pc)
        pc.on("datachannel", session.on_datachannel)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        pc.addTrack(PatternTrack(self.width, self.height, self.noise))
        h264 = [c for c in RTCRtpSender.getCapabilities("video").codecs if c.mimeType == "video/H264"]
        for transceiver in pc.getTransceivers():
            if transceiver.kind == "video":
                transceiver.setCodecPreferences(h264)
        await pc.setLocalDescription(await pc.createAnswer())

        lines = pc.localDescription.sdp.split("\r\n")
        candidates = list(dict.fromkeys(line[2:] for line in lines if line.startswith("a=candidate:")))
        answer_sdp = "\r\n".join(
            line for line in lines if not line.startswith("a=candidate:") and line != "a=end-of-candidates"
        )
        for candidate in candidates:
            await ws.send_json(
                {"type": "new-ice-candidate", "data": {"candidate": candidate, "sdpMid": "", "sdpMLineIndex": 0}}
            )
        answer = base64.b64encode(json.dumps({"type": "answer", "sdp": answer_sdp}).encode()).decode()
        await ws.send_json({"type": "answer", "data": answer})

        previous = self.current
        if previous is not None:
            previous.send_event("otherSessionConnected", None)
            asyncio.get_running_loop().call_later(1, lambda: asyncio.ensure_future(previous.close()))
        self.current = session
        self.sessions.append(session)

    # -- JSON-RPC ----------------------------------------------------------------

    def call(self, method: str, p: dict[str, Any]) -> Any:
        self.rpc_log.append(method)
        if method == "getVideoState":
            return {"ready": True, "streaming": 1, "width": self.width, "height": self.height, "fps": 30.0}
        if method == "getKeyboardLedState":
            return self.led_state
        if method == "getUSBState":
            return "configured"
        if method == "keyboardReport":
            self.keyboard.append((int(p["modifier"]), tuple(int(k) for k in p["keys"] if k)))
            return None
        if method == "absMouseReport":
            self.pointer.append((int(p["x"]), int(p["y"]), int(p["buttons"])))
            return None
        if method == "relMouseReport":
            self.relative.append((int(p["dx"]), int(p["dy"]), int(p["buttons"])))
            return None
        if method == "wheelReport":
            self.wheel.append((int(p["wheelY"]), int(p["wheelX"])))
            return None
        if method == "getActiveExtension":
            return self.extension
        if method == "setATXPowerAction":
            if p["action"] not in ("power-short", "power-long", "reset"):
                raise ValueError(f"invalid action: {p['action']}")
            self.atx_actions.append(p["action"])
            return None
        if method == "getATXState":
            return {"power": True, "hdd": False}
        if method == "mountWithHTTP":
            self.mounted = {"source": "HTTP", "mode": p["mode"], "url": p["url"]}
            return None
        if method == "mountWithStorage":
            self.mounted = {"source": "Storage", "mode": p["mode"], "filename": p["filename"]}
            return None
        if method == "unmountImage":
            self.mounted = None
            return None
        if method == "getVirtualMediaState":
            return self.mounted
        if method == "wakeHost":
            return None
        if method == "listStorageFiles":
            return {"files": [{"filename": "ubuntu.iso", "size": 123, "createdAt": "2026-01-01T00:00:00Z"}]}
        if method == "getStorageSpace":
            return {"bytesUsed": 123, "bytesFree": 1000}
        raise LookupError(method)


class DeviceSession:
    def __init__(self, device: FakeJetKVM, pc: RTCPeerConnection):
        self.device = device
        self.pc = pc
        self.rpc_channel: Any = None
        self.hid_channel: Any = None
        self.keyboard_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.macro_cancel = asyncio.Event()
        self.worker = asyncio.ensure_future(self._keyboard_worker())

    async def close(self) -> None:
        self.worker.cancel()
        await self.pc.close()

    def on_datachannel(self, channel: Any) -> None:
        if channel.label == "rpc":
            self.rpc_channel = channel
            channel.on("message", self._on_rpc)
            device = self.device
            self.send_event("videoInputState", device.call("getVideoState", {}))
            self.send_event("usbState", "configured")
        elif channel.label == "hidrpc" and self.device.hidrpc:
            self.hid_channel = channel
            channel.on("message", self._on_hid)

    def send_event(self, method: str, params: Any) -> None:
        if self.rpc_channel is not None and self.rpc_channel.readyState == "open":
            event: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                event["params"] = params
            self.rpc_channel.send(json.dumps(event))

    def _on_rpc(self, message: str) -> None:
        request = json.loads(message)
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": request.get("id")}
        try:
            result = self.device.call(request["method"], request.get("params") or {})
            if result not in (None, False, 0, "", [], {}):  # Go's omitempty
                response["result"] = result
        except LookupError as e:
            if isinstance(e, KeyError):
                response["error"] = {"code": -32603, "message": "Internal error", "data": f"missing parameter: {e.args[0]}"}
            else:
                response["error"] = {"code": -32601, "message": "Method not found"}
        except ValueError as e:
            response["error"] = {"code": -32603, "message": "Internal error", "data": str(e)}
        self.rpc_channel.send(json.dumps(response))

    def _on_hid(self, message: bytes) -> None:
        kind = message[0]
        if kind == 0x08:  # cancel bypasses the queue, as in the firmware
            self.macro_cancel.set()
        elif kind in (0x01, 0x02, 0x07):
            self.keyboard_queue.put_nowait(message)
        elif kind == 0x03:
            self.device.pointer.append(struct.unpack(">iiB", message[1:10]))
        elif kind == 0x06:
            self.device.relative.append(struct.unpack(">bbB", message[1:4]))

    async def _keyboard_worker(self) -> None:
        while True:
            message = await self.keyboard_queue.get()
            kind = message[0]
            if kind == 0x01:
                self.hid_channel.send(bytes([0x01, 0x01]))
            elif kind == 0x02:
                self.device.keyboard.append((message[1], tuple(k for k in message[2:8] if k)))
            elif kind == 0x07:
                await self._run_macro(message[1:])

    async def _run_macro(self, payload: bytes) -> None:
        (count,) = struct.unpack(">I", payload[1:5])
        assert len(payload) == count * 9 + 5, "macro length mismatch"
        self.macro_cancel.clear()
        self.hid_channel.send(bytes([0x34, 1, 1]))
        for i in range(count):
            step = payload[5 + i * 9 : 14 + i * 9]
            self.device.keyboard.append((step[0], tuple(k for k in step[1:7] if k)))
            (delay,) = struct.unpack(">H", step[7:9])
            try:
                await asyncio.wait_for(self.macro_cancel.wait(), delay / 1000)
                self.device.keyboard.append((0, ()))
                break
            except TimeoutError:
                pass
        self.hid_channel.send(bytes([0x34, 0, 1]))

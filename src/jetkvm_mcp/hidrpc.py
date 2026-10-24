"""Encoder/decoder for JetKVM's binary "hidrpc" data channel protocol.

Mirrors internal/hidrpc in the JetKVM firmware: every message is one type byte
followed by a type-specific payload. Multi-byte integers are big-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

from .keyboard import MAX_KEYS_PER_REPORT, MacroStep

VERSION = 0x01

TYPE_HANDSHAKE = 0x01
TYPE_KEYBOARD_REPORT = 0x02
TYPE_POINTER_REPORT = 0x03
TYPE_WHEEL_REPORT = 0x04  # defined by the firmware but not handled; wheel goes over JSON-RPC
TYPE_KEYPRESS_REPORT = 0x05
TYPE_MOUSE_REPORT = 0x06
TYPE_KEYBOARD_MACRO_REPORT = 0x07
TYPE_CANCEL_KEYBOARD_MACRO_REPORT = 0x08
TYPE_KEYPRESS_KEEPALIVE_REPORT = 0x09
TYPE_KEYBOARD_LED_STATE = 0x32
TYPE_KEYDOWN_STATE = 0x33
TYPE_KEYBOARD_MACRO_STATE = 0x34

# Matches the web UI: a macro is sent in chunks of this many steps so a single
# message stays well under the data channel's size limit. Even, so a chunk never
# splits a key press from its release.
MACRO_CHUNK_STEPS = 128


def handshake() -> bytes:
    return bytes([TYPE_HANDSHAKE, VERSION])


def _pad_keys(keys: Iterable[int]) -> bytes:
    keys = bytes(keys)
    if len(keys) > MAX_KEYS_PER_REPORT:
        raise ValueError(f"at most {MAX_KEYS_PER_REPORT} keys per report")
    return keys + bytes(MAX_KEYS_PER_REPORT - len(keys))


def keyboard_report(modifier: int, keys: Iterable[int]) -> bytes:
    return bytes([TYPE_KEYBOARD_REPORT, modifier]) + _pad_keys(keys)


def pointer_report(x: int, y: int, buttons: int) -> bytes:
    """Absolute pointer position in the HID digitizer range 0..32767."""
    return struct.pack(">BiiB", TYPE_POINTER_REPORT, x, y, buttons)


def mouse_report(dx: int, dy: int, buttons: int) -> bytes:
    """Relative mouse movement; dx and dy are clamped to a signed byte by the caller."""
    return struct.pack(">BbbB", TYPE_MOUSE_REPORT, dx, dy, buttons)


def keyboard_macro(steps: list[MacroStep]) -> bytes:
    out = bytearray(struct.pack(">BBI", TYPE_KEYBOARD_MACRO_REPORT, 1, len(steps)))
    for step in steps:
        if not 0 <= step.delay_ms <= 0xFFFF:
            raise ValueError("macro step delay must be 0..65535 ms")
        out.append(step.modifier)
        out += _pad_keys(step.keys)
        out += struct.pack(">H", step.delay_ms)
    return bytes(out)


def cancel_keyboard_macro() -> bytes:
    return bytes([TYPE_CANCEL_KEYBOARD_MACRO_REPORT])


def chunk_steps(steps: list[MacroStep], size: int = MACRO_CHUNK_STEPS) -> list[list[MacroStep]]:
    return [steps[i : i + size] for i in range(0, len(steps), size)]


@dataclass(frozen=True)
class IncomingMessage:
    type: int
    payload: bytes


def parse(data: bytes) -> IncomingMessage:
    if not data:
        raise ValueError("empty hidrpc message")
    return IncomingMessage(data[0], bytes(data[1:]))

import struct

import pytest

from jetkvm_mcp import hidrpc
from jetkvm_mcp.keyboard import MacroStep


def test_handshake():
    assert hidrpc.handshake() == b"\x01\x01"


def test_keyboard_report_pads_to_six_keys():
    assert hidrpc.keyboard_report(0x02, [0x04]) == bytes([0x02, 0x02, 0x04, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        hidrpc.keyboard_report(0, range(1, 8))


def test_pointer_report_is_big_endian_int32():
    assert hidrpc.pointer_report(1, 32767, 1) == bytes([0x03, 0, 0, 0, 1, 0, 0, 0x7F, 0xFF, 1])


def test_mouse_report_signed_bytes():
    assert hidrpc.mouse_report(-1, 5, 2) == bytes([0x06, 0xFF, 0x05, 0x02])


def test_keyboard_macro_layout_matches_firmware_parser():
    steps = [MacroStep(0x02, (0x04,), 20), MacroStep(0, (), 300)]
    data = hidrpc.keyboard_macro(steps)
    # Firmware: payload (after the type byte) = isPaste, u32 BE count, then 9 bytes per step.
    payload = data[1:]
    assert data[0] == hidrpc.TYPE_KEYBOARD_MACRO_REPORT
    assert len(payload) == len(steps) * 9 + 5
    assert payload[0] == 1
    assert struct.unpack(">I", payload[1:5]) == (2,)
    assert payload[5:14] == bytes([0x02, 0x04, 0, 0, 0, 0, 0]) + struct.pack(">H", 20)
    assert payload[14:23] == bytes(7) + struct.pack(">H", 300)


def test_chunks_keep_press_and_release_together():
    steps = [MacroStep(0, (4,), 1), MacroStep(0, (), 1)] * 100
    chunks = hidrpc.chunk_steps(steps)
    assert [len(c) for c in chunks] == [128, 72]
    assert all(c[0].keys for c in chunks)

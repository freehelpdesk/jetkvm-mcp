"""USB HID keyboard tables and helpers for turning key names and text into reports.

Text typing assumes the target computer uses a US keyboard layout: the JetKVM
emulates a physical keyboard, so it sends key positions, not characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Modifier byte bits in a boot-protocol keyboard report.
MOD_LCTRL = 0x01
MOD_LSHIFT = 0x02
MOD_LALT = 0x04
MOD_LMETA = 0x08
MOD_RCTRL = 0x10
MOD_RSHIFT = 0x20
MOD_RALT = 0x40
MOD_RMETA = 0x80

# Keyboard LED bits reported by the host (see usbgadget.KeyboardLedMask*).
LED_NUM_LOCK = 0x01
LED_CAPS_LOCK = 0x02
LED_SCROLL_LOCK = 0x04

MAX_KEYS_PER_REPORT = 6

MODIFIERS: dict[str, int] = {
    "ctrl": MOD_LCTRL,
    "control": MOD_LCTRL,
    "lctrl": MOD_LCTRL,
    "leftctrl": MOD_LCTRL,
    "shift": MOD_LSHIFT,
    "lshift": MOD_LSHIFT,
    "leftshift": MOD_LSHIFT,
    "alt": MOD_LALT,
    "lalt": MOD_LALT,
    "leftalt": MOD_LALT,
    "option": MOD_LALT,
    "opt": MOD_LALT,
    "meta": MOD_LMETA,
    "lmeta": MOD_LMETA,
    "cmd": MOD_LMETA,
    "command": MOD_LMETA,
    "win": MOD_LMETA,
    "windows": MOD_LMETA,
    "super": MOD_LMETA,
    "gui": MOD_LMETA,
    "rctrl": MOD_RCTRL,
    "rightctrl": MOD_RCTRL,
    "rshift": MOD_RSHIFT,
    "rightshift": MOD_RSHIFT,
    "ralt": MOD_RALT,
    "rightalt": MOD_RALT,
    "altgr": MOD_RALT,
    "rmeta": MOD_RMETA,
    "rcmd": MOD_RMETA,
    "rwin": MOD_RMETA,
    "rsuper": MOD_RMETA,
}

# Named (non-character) keys -> HID usage IDs (Keyboard/Keypad page 0x07).
NAMED_KEYS: dict[str, int] = {
    "enter": 0x28,
    "return": 0x28,
    "escape": 0x29,
    "esc": 0x29,
    "backspace": 0x2A,
    "tab": 0x2B,
    "space": 0x2C,
    "spacebar": 0x2C,
    "minus": 0x2D,
    "equal": 0x2E,
    "equals": 0x2E,
    "bracketleft": 0x2F,
    "leftbracket": 0x2F,
    "bracketright": 0x30,
    "rightbracket": 0x30,
    "backslash": 0x31,
    "semicolon": 0x33,
    "quote": 0x34,
    "apostrophe": 0x34,
    "grave": 0x35,
    "backtick": 0x35,
    "comma": 0x36,
    "period": 0x37,
    "dot": 0x37,
    "slash": 0x38,
    "capslock": 0x39,
    "printscreen": 0x46,
    "prtsc": 0x46,
    "sysrq": 0x46,
    "scrolllock": 0x47,
    "pause": 0x48,
    "break": 0x48,
    "insert": 0x49,
    "ins": 0x49,
    "home": 0x4A,
    "pageup": 0x4B,
    "pgup": 0x4B,
    "delete": 0x4C,
    "del": 0x4C,
    "end": 0x4D,
    "pagedown": 0x4E,
    "pgdn": 0x4E,
    "right": 0x4F,
    "arrowright": 0x4F,
    "left": 0x50,
    "arrowleft": 0x50,
    "down": 0x51,
    "arrowdown": 0x51,
    "up": 0x52,
    "arrowup": 0x52,
    "numlock": 0x53,
    "kpdivide": 0x54,
    "kpmultiply": 0x55,
    "kpminus": 0x56,
    "kpplus": 0x57,
    "kpenter": 0x58,
    "kpdecimal": 0x63,
    "kpequal": 0x67,
    "intlbackslash": 0x64,
    "menu": 0x65,
    "application": 0x65,
    "contextmenu": 0x65,
    "power": 0x66,
    "mute": 0x7F,
    "volumeup": 0x80,
    "volumedown": 0x81,
    "plus": 0x2E,  # the "=" key; _SHIFTED_NAMES adds shift
}
for _i in range(12):
    NAMED_KEYS[f"f{_i + 1}"] = 0x3A + _i
for _i in range(12):
    NAMED_KEYS[f"f{_i + 13}"] = 0x68 + _i
for _i in range(1, 10):
    NAMED_KEYS[f"kp{_i}"] = 0x59 + _i - 1
NAMED_KEYS["kp0"] = 0x62

# Named keys that need shift to produce their symbol.
_SHIFTED_NAMES = {"plus"}

# US layout: character -> (modifier, usage ID).
US_LAYOUT: dict[str, tuple[int, int]] = {}
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    US_LAYOUT[_c] = (0, 0x04 + _i)
    US_LAYOUT[_c.upper()] = (MOD_LSHIFT, 0x04 + _i)
for _i, _c in enumerate("1234567890"):
    US_LAYOUT[_c] = (0, 0x1E + _i)
for _i, _c in enumerate("!@#$%^&*()"):
    US_LAYOUT[_c] = (MOD_LSHIFT, 0x1E + _i)
for _plain, _shifted, _code in [
    ("-", "_", 0x2D),
    ("=", "+", 0x2E),
    ("[", "{", 0x2F),
    ("]", "}", 0x30),
    ("\\", "|", 0x31),
    (";", ":", 0x33),
    ("'", '"', 0x34),
    ("`", "~", 0x35),
    (",", "<", 0x36),
    (".", ">", 0x37),
    ("/", "?", 0x38),
]:
    US_LAYOUT[_plain] = (0, _code)
    US_LAYOUT[_shifted] = (MOD_LSHIFT, _code)
US_LAYOUT[" "] = (0, 0x2C)
US_LAYOUT["\n"] = (0, 0x28)
US_LAYOUT["\t"] = (0, 0x2B)


@dataclass(frozen=True)
class KeyCombo:
    """One chord: modifiers plus up to six simultaneously pressed keys."""

    modifier: int
    keys: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        return self.modifier == 0 and not self.keys


@dataclass(frozen=True)
class MacroStep:
    """A full keyboard report, held for `delay_ms` before the next step."""

    modifier: int
    keys: tuple[int, ...]
    delay_ms: int


def _normalize_name(token: str) -> str:
    return re.sub(r"[\s_\-]", "", token.lower())


def parse_key(token: str) -> KeyCombo:
    """Parse a single key token such as "a", "enter", "f5", "?" or "ctrl"."""
    if not token:
        raise ValueError("empty key name")
    if len(token) == 1:
        char = token.lower() if token.isalpha() else token
        if char in US_LAYOUT:
            modifier, code = US_LAYOUT[char]
            return KeyCombo(modifier, (code,))
        raise ValueError(f"unsupported key character {token!r}")
    name = _normalize_name(token)
    if name in MODIFIERS:
        return KeyCombo(MODIFIERS[name], ())
    if name in NAMED_KEYS:
        modifier = MOD_LSHIFT if name in _SHIFTED_NAMES else 0
        return KeyCombo(modifier, (NAMED_KEYS[name],))
    raise ValueError(f"unknown key name {token!r}")


def parse_combo(combo: str) -> KeyCombo:
    """Parse a chord like "ctrl+alt+delete", "cmd+shift+4", "shift+F10" or "ctrl++".

    Letters are case-insensitive here (use "shift+a" for a capital A); symbols
    that need shift on a US keyboard (e.g. "?") add shift automatically.
    """
    combo = combo.strip()
    if not combo:
        raise ValueError("empty key combination")
    # Split on "+" except a trailing one, so "ctrl++" means ctrl and the plus key.
    tokens = re.split(r"(?<!^)\+(?=.)", combo)
    modifier = 0
    keys: list[int] = []
    for token in tokens:
        part = parse_key(token.strip() or "+")
        modifier |= part.modifier
        for code in part.keys:
            if code not in keys:
                keys.append(code)
    if len(keys) > MAX_KEYS_PER_REPORT:
        raise ValueError(f"at most {MAX_KEYS_PER_REPORT} non-modifier keys can be pressed at once")
    return KeyCombo(modifier, tuple(keys))


def parse_sequence(sequence: str) -> list[KeyCombo]:
    """Parse whitespace-separated chords, e.g. "ctrl+a ctrl+c" or "down down enter"."""
    combos = [parse_combo(part) for part in sequence.split()]
    if not combos:
        raise ValueError("no keys given")
    return combos


def unsupported_characters(text: str) -> list[str]:
    """Characters in `text` that cannot be typed with the US layout."""
    seen: list[str] = []
    for char in _normalize_newlines(text):
        if char not in US_LAYOUT and char not in seen:
            seen.append(char)
    return seen


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def text_to_steps(text: str, delay_ms: int, caps_lock: bool = False) -> list[MacroStep]:
    """Turn text into press/release macro steps for the US layout.

    With caps lock on, shift is inverted for letters so the typed case matches.
    """
    bad = unsupported_characters(text)
    if bad:
        raise ValueError(
            "cannot type these characters with a US keyboard layout: "
            + ", ".join(repr(c) for c in bad)
        )
    steps: list[MacroStep] = []
    for char in _normalize_newlines(text):
        modifier, code = US_LAYOUT[char]
        if caps_lock and char.isalpha():
            modifier ^= MOD_LSHIFT
        steps.append(MacroStep(modifier, (code,), delay_ms))
        steps.append(MacroStep(0, (), delay_ms))
    return steps


def combo_to_steps(combo: KeyCombo, hold_ms: int, release_ms: int) -> list[MacroStep]:
    """Press a chord like a person would: modifiers first, then keys, then release."""
    steps: list[MacroStep] = []
    if combo.modifier and combo.keys:
        steps.append(MacroStep(combo.modifier, (), 20))
    steps.append(MacroStep(combo.modifier, combo.keys, hold_ms))
    if combo.modifier and combo.keys:
        steps.append(MacroStep(combo.modifier, (), 20))
    steps.append(MacroStep(0, (), release_ms))
    return steps

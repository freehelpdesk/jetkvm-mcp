import pytest

from jetkvm_mcp.keyboard import (
    MOD_LALT,
    MOD_LCTRL,
    MOD_LMETA,
    MOD_LSHIFT,
    MOD_RALT,
    KeyCombo,
    MacroStep,
    combo_to_steps,
    parse_combo,
    parse_sequence,
    text_to_steps,
    unsupported_characters,
)


@pytest.mark.parametrize(
    "combo, expected",
    [
        ("ctrl+alt+delete", KeyCombo(MOD_LCTRL | MOD_LALT, (0x4C,))),
        ("cmd+space", KeyCombo(MOD_LMETA, (0x2C,))),
        ("ctrl+C", KeyCombo(MOD_LCTRL, (0x06,))),  # letters are case-insensitive in chords
        ("shift+F10", KeyCombo(MOD_LSHIFT, (0x43,))),
        ("ctrl++", KeyCombo(MOD_LCTRL | MOD_LSHIFT, (0x2E,))),
        ("+", KeyCombo(MOD_LSHIFT, (0x2E,))),
        ("?", KeyCombo(MOD_LSHIFT, (0x38,))),
        ("super", KeyCombo(MOD_LMETA, ())),
        ("altgr+e", KeyCombo(MOD_RALT, (0x08,))),
        ("Page_Down", KeyCombo(0, (0x4E,))),
        ("f24", KeyCombo(0, (0x73,))),
        ("kp0", KeyCombo(0, (0x62,))),
        ("enter", KeyCombo(0, (0x28,))),
        ("a+b", KeyCombo(0, (0x04, 0x05))),
    ],
)
def test_parse_combo(combo, expected):
    assert parse_combo(combo) == expected


@pytest.mark.parametrize("combo", ["", "ctrl+banana", "ctrl+", "é", "a+b+c+d+e+f+g"])
def test_parse_combo_rejects(combo):
    with pytest.raises(ValueError):
        parse_combo(combo)


def test_parse_sequence():
    assert parse_sequence("down  down enter") == [
        KeyCombo(0, (0x51,)),
        KeyCombo(0, (0x51,)),
        KeyCombo(0, (0x28,)),
    ]


def test_text_to_steps_presses_and_releases_each_char():
    steps = text_to_steps("Hi!\r\n", delay_ms=7)
    assert steps == [
        MacroStep(MOD_LSHIFT, (0x0B,), 7),
        MacroStep(0, (), 7),
        MacroStep(0, (0x0C,), 7),
        MacroStep(0, (), 7),
        MacroStep(MOD_LSHIFT, (0x1E,), 7),
        MacroStep(0, (), 7),
        MacroStep(0, (0x28,), 7),  # \r\n becomes a single Enter
        MacroStep(0, (), 7),
    ]


def test_text_to_steps_compensates_for_caps_lock():
    pressed = [s for s in text_to_steps("aB1", delay_ms=1, caps_lock=True) if s.keys]
    assert [s.modifier for s in pressed] == [MOD_LSHIFT, 0, 0]


def test_unsupported_characters_rejected_before_typing():
    assert unsupported_characters("héllo “x”") == ["é", "“", "”"]
    with pytest.raises(ValueError, match="'é'"):
        text_to_steps("héllo", delay_ms=1)


def test_combo_to_steps_presses_modifiers_first():
    assert combo_to_steps(parse_combo("ctrl+c"), hold_ms=50, release_ms=30) == [
        MacroStep(MOD_LCTRL, (), 20),
        MacroStep(MOD_LCTRL, (0x06,), 50),
        MacroStep(MOD_LCTRL, (), 20),
        MacroStep(0, (), 30),
    ]
    assert combo_to_steps(parse_combo("enter"), hold_ms=50, release_ms=30) == [
        MacroStep(0, (0x28,), 50),
        MacroStep(0, (), 30),
    ]

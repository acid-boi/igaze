"""Maps macOS keys (as pynput reports them) to Linux evdev key names.

Design decisions baked in here:

  * POSITION-BASED. We map by physical key position, not by the character
    produced. pynput on the Mac gives us either a printable `.char` or a
    named special key; we translate that to the Linux `KEY_*` that sits in
    the same spot on a standard keyboard. The Linux injector then presses
    that physical key, and the Linux system applies its OWN layout/shift
    logic on top. So a Shift chord is sent as two physical key events
    (Shift down, letter down, ...), NOT as a pre-shifted character -- which
    is exactly what makes real chords like Super+Enter work.

  * Cmd -> Super. macOS's Command key maps to Linux KEY_LEFTMETA, i.e. the
    Super key Omarchy/Hyprland binds shortcuts to. This is the whole point
    of the redesign: pressing Cmd+Enter on the Mac fires Super+Enter on
    Linux.

  * MINIMAL first pass, by request: letters, digits, the common editing
    keys (Enter/Space/Tab/Backspace), and the four modifier families
    (Cmd/Shift/Ctrl/Alt). Punctuation, function keys, and arrows are
    intentionally left out for now; unmapped keys are reported by
    `to_linux_key` returning None so callers can drop them cleanly instead
    of guessing.

The values here are evdev KEY_* NAMES (strings), validated against the
evdev library at import of the injector. Keeping them as names rather than
raw integers keeps this file readable and layout-documentation-friendly;
the injector resolves them to integers via ecodes.
"""

from __future__ import annotations

# Printable characters (pynput key.char) -> Linux KEY_ name, by position on
# a US physical keyboard. Letters are stored lowercase; we lowercase the
# incoming char before lookup so a Shift-produced 'A' still resolves to
# KEY_A (the Shift event is carried separately as its own modifier key).
_CHAR_TO_KEY = {
    # letters
    **{c: f"KEY_{c.upper()}" for c in "abcdefghijklmnopqrstuvwxyz"},
    # digits
    **{d: f"KEY_{d}" for d in "0123456789"},
}

# pynput special-key names (key.name) -> Linux KEY_ name.
# pynput exposes left/right variants for modifiers (e.g. cmd, cmd_r); both
# fold onto the same Linux key for this minimal pass.
_SPECIAL_TO_KEY = {
    "enter": "KEY_ENTER",
    "space": "KEY_SPACE",
    "tab": "KEY_TAB",
    "backspace": "KEY_BACKSPACE",
    # Cmd -> Super (LEFTMETA). This is the mapping that makes Omarchy
    # Super+<key> shortcuts fire.
    "cmd": "KEY_LEFTMETA",
    "cmd_r": "KEY_LEFTMETA",
    # other modifiers, by position
    "shift": "KEY_LEFTSHIFT",
    "shift_r": "KEY_RIGHTSHIFT",
    "ctrl": "KEY_LEFTCTRL",
    "ctrl_r": "KEY_RIGHTCTRL",
    "alt": "KEY_LEFTALT",
    "alt_r": "KEY_RIGHTALT",
}


def to_linux_key(char: str | None, name: str | None) -> str | None:
    """Resolve a pynput key to a Linux KEY_ name.

    Exactly one of `char` (printable key) or `name` (special key) is
    expected to be set, matching how pynput reports events. Returns the
    KEY_ name, or None if this key isn't in the minimal map (caller should
    drop it).
    """
    if char is not None:
        return _CHAR_TO_KEY.get(char.lower())
    if name is not None:
        return _SPECIAL_TO_KEY.get(name)
    return None


def all_mapped_key_names() -> set[str]:
    """Every Linux KEY_ name this map can emit -- used by the injector to
    pre-declare the virtual device's capabilities and to validate the names
    against evdev at startup."""
    return set(_CHAR_TO_KEY.values()) | set(_SPECIAL_TO_KEY.values())

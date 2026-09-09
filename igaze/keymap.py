"""Maps macOS keys (as pynput reports them) to Linux evdev key names.

Design decisions baked in here:

  * POSITION-BASED. We map by physical key position, not by the character
    produced. pynput on the Mac gives us either a printable `.char` or a
    named special key; we translate that to the Linux `KEY_*` that sits in
    the same spot on a standard US keyboard. The Linux injector presses that
    physical key, and the Linux system applies its OWN layout/shift logic on
    top. So a shifted symbol is sent as Shift + the base key, NOT as a
    pre-shifted character -- which is what makes real chords like Super+Enter
    (and Shift-based symbols) work.

  * Cmd -> Super. macOS's Command key maps to Linux KEY_LEFTMETA, i.e. the
    Super key Omarchy/Hyprland binds shortcuts to. Pressing Cmd+Enter on the
    Mac fires Super+Enter on Linux.

  * SHIFTED CHARACTERS. When you press e.g. Shift+'/', macOS delivers the
    already-shifted char '?'. Position-based mapping means we translate '?'
    back to its BASE physical key (KEY_SLASH); the Shift key press itself
    arrives as its own separate event, so Linux re-derives '?' from
    Shift+SLASH. Both a symbol and its unshifted twin therefore map to the
    same KEY_ (e.g. '1' and '!' -> KEY_1).

The values here are evdev KEY_* NAMES (strings), validated against the evdev
library at injector startup. Names (not raw ints) keep this file readable;
the injector resolves them to integers via ecodes.
"""

from __future__ import annotations

# Printable characters (pynput key.char) -> Linux KEY_ name, by US physical
# position. Letters are lowercased before lookup so a Shift-produced 'A'
# still resolves to KEY_A (Shift travels as its own event).
_CHAR_TO_KEY: dict[str, str] = {}

# letters
for _c in "abcdefghijklmnopqrstuvwxyz":
    _CHAR_TO_KEY[_c] = f"KEY_{_c.upper()}"

# digit row -- both the digit and its shifted symbol land on the same key
_DIGIT_ROW = {
    "1": "KEY_1", "!": "KEY_1",
    "2": "KEY_2", "@": "KEY_2",
    "3": "KEY_3", "#": "KEY_3",
    "4": "KEY_4", "$": "KEY_4",
    "5": "KEY_5", "%": "KEY_5",
    "6": "KEY_6", "^": "KEY_6",
    "7": "KEY_7", "&": "KEY_7",
    "8": "KEY_8", "*": "KEY_8",
    "9": "KEY_9", "(": "KEY_9",
    "0": "KEY_0", ")": "KEY_0",
}
_CHAR_TO_KEY.update(_DIGIT_ROW)

# punctuation / symbol keys, each with its unshifted and shifted char
_PUNCTUATION = {
    "-": "KEY_MINUS", "_": "KEY_MINUS",
    "=": "KEY_EQUAL", "+": "KEY_EQUAL",
    "[": "KEY_LEFTBRACE", "{": "KEY_LEFTBRACE",
    "]": "KEY_RIGHTBRACE", "}": "KEY_RIGHTBRACE",
    "\\": "KEY_BACKSLASH", "|": "KEY_BACKSLASH",
    ";": "KEY_SEMICOLON", ":": "KEY_SEMICOLON",
    "'": "KEY_APOSTROPHE", "\"": "KEY_APOSTROPHE",
    "`": "KEY_GRAVE", "~": "KEY_GRAVE",
    ",": "KEY_COMMA", "<": "KEY_COMMA",
    ".": "KEY_DOT", ">": "KEY_DOT",
    "/": "KEY_SLASH", "?": "KEY_SLASH",
}
_CHAR_TO_KEY.update(_PUNCTUATION)

# pynput special-key names (key.name) -> Linux KEY_ name.
# pynput exposes left/right variants for modifiers (e.g. cmd, cmd_r); both
# fold onto the correct positional Linux key.
_SPECIAL_TO_KEY = {
    # editing / whitespace
    "enter": "KEY_ENTER",
    "space": "KEY_SPACE",
    "tab": "KEY_TAB",
    "backspace": "KEY_BACKSPACE",
    "delete": "KEY_DELETE",
    "esc": "KEY_ESC",
    "caps_lock": "KEY_CAPSLOCK",
    # navigation
    "up": "KEY_UP",
    "down": "KEY_DOWN",
    "left": "KEY_LEFT",
    "right": "KEY_RIGHT",
    "home": "KEY_HOME",
    "end": "KEY_END",
    "page_up": "KEY_PAGEUP",
    "page_down": "KEY_PAGEDOWN",
    "insert": "KEY_INSERT",
    # function row
    "f1": "KEY_F1", "f2": "KEY_F2", "f3": "KEY_F3", "f4": "KEY_F4",
    "f5": "KEY_F5", "f6": "KEY_F6", "f7": "KEY_F7", "f8": "KEY_F8",
    "f9": "KEY_F9", "f10": "KEY_F10", "f11": "KEY_F11", "f12": "KEY_F12",
    # Cmd -> Super (LEFTMETA): makes Omarchy Super+<key> shortcuts fire
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

    Exactly one of `char` (printable key) or `name` (special key) is expected
    to be set, matching how pynput reports events. Returns the KEY_ name, or
    None if this key isn't mapped (caller should drop it).
    """
    if char is not None:
        # Lowercase letters so shifted letters resolve to the base key; the
        # dict lookups for symbols are already exact.
        return _CHAR_TO_KEY.get(char) or _CHAR_TO_KEY.get(char.lower())
    if name is not None:
        return _SPECIAL_TO_KEY.get(name)
    return None


def all_mapped_key_names() -> set[str]:
    """Every Linux KEY_ name this map can emit -- used by the injector to
    pre-declare the virtual device's capabilities and validate names against
    evdev at startup."""
    return set(_CHAR_TO_KEY.values()) | set(_SPECIAL_TO_KEY.values())

"""Wire format for forwarded key EVENTS (not characters).

Each line is one key event: a Linux KEY_ name and whether it went down or
up. This is what makes real chords work -- the receiver replays the exact
down/up timeline, so Super can still be held while Enter fires:

    KEY_LEFTMETA D      <- Super down
    KEY_ENTER    D      <- Enter down  (terminal opens here on Omarchy)
    KEY_ENTER    U
    KEY_LEFTMETA U

Format, one event per line, newline-terminated:

    <KEY_NAME> <D|U>

It stays human-readable on purpose: run recv_test.py (or even nc) on the
Linux side and you can watch the event stream directly. It's also trivial
to parse on the injector side -- split on whitespace, look up the KEY_ name,
map D/U to evdev's 1/0.

Newline is the frame delimiter. KEY_ names never contain whitespace, so a
line is always exactly "<name> <D|U>".
"""

from __future__ import annotations

from dataclasses import dataclass

DOWN = "D"
UP = "U"


@dataclass(frozen=True)
class KeyEvent:
    key_name: str  # a Linux KEY_ name, e.g. "KEY_ENTER"
    pressed: bool  # True = down, False = up

    def encode(self) -> str:
        """Serialize to a single wire line (without the trailing newline)."""
        return f"{self.key_name} {DOWN if self.pressed else UP}"


def decode(line: str) -> KeyEvent | None:
    """Parse one wire line back into a KeyEvent, or None if malformed.

    Tolerant by design: the injector should skip a garbled line rather than
    crash the whole stream on it.
    """
    parts = line.strip().split()
    if len(parts) != 2:
        return None
    key_name, state = parts
    if not key_name.startswith("KEY_"):
        return None
    if state == DOWN:
        return KeyEvent(key_name, True)
    if state == UP:
        return KeyEvent(key_name, False)
    return None

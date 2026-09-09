"""Passive keyboard capture on macOS, emitting key EVENTS, gaze-gated.

Changed from the character-based version: we now listen for BOTH press and
release, translate each to a Linux KEY_ name via keymap, and emit a
KeyEvent (down/up). That's what lets the far side replay real chords like
Cmd(Super)+Enter.

Still passive: pynput on macOS does not swallow events, so everything you
type still reaches the Mac app you're using. We only ALSO forward, and only
while the gaze target is LINUX.

Gating subtlety for chords: the gate is checked on the DOWN event. If a key
was forwarded down while looking at Linux, its UP is forwarded too even if
your gaze drifts in between -- otherwise a released-look mid-chord would
strand a key in the "held" state on the Linux side (a stuck Super is
exactly the kind of bug that makes a machine unusable). We track the set of
keys we've forwarded-down and always deliver their matching up.
"""

from __future__ import annotations

from typing import Callable

from pynput import keyboard

from igaze.keymap import to_linux_key
from igaze.protocol import KeyEvent


class KeyboardCapture:
    def __init__(
        self,
        on_event: Callable[[KeyEvent], None],
        should_forward: Callable[[], bool],
    ) -> None:
        """
        on_event: called with each KeyEvent to forward.
        should_forward: returns True when gaze target is LINUX (forward),
            False when looking at the Mac (drop).
        """
        self._on_event = on_event
        self._should_forward = should_forward
        self._listener: keyboard.Listener | None = None
        # KEY_ names we've sent a DOWN for and not yet a matching UP. Their
        # UP must always be delivered to avoid a stuck key on the far side.
        self._held: set[str] = set()

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        # Best-effort: release anything still held so we never leave the far
        # side with a stuck key when we shut down.
        for key_name in list(self._held):
            self._on_event(KeyEvent(key_name, False))
        self._held.clear()

    def _on_press(self, key) -> None:
        key_name = self._resolve(key)
        if key_name is None:
            return
        # Gate on the down event.
        if not self._should_forward():
            return
        self._held.add(key_name)
        self._on_event(KeyEvent(key_name, True))

    def _on_release(self, key) -> None:
        key_name = self._resolve(key)
        if key_name is None:
            return
        # Only forward the up if we forwarded its down -- and always do so in
        # that case, regardless of current gaze, to avoid a stuck key.
        if key_name in self._held:
            self._held.discard(key_name)
            self._on_event(KeyEvent(key_name, False))

    @staticmethod
    def _resolve(key) -> str | None:
        char = getattr(key, "char", None)
        name = getattr(key, "name", None)
        return to_linux_key(char, name)

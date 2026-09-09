"""macOS keyboard capture that can SUPPRESS local keys (true KVM behaviour).

When gaze is on Linux, keystrokes are forwarded to the Linux box AND blocked
from the Mac, so only Linux receives them. When gaze is on the Mac, keys pass
through untouched and nothing is forwarded.

Suppression uses pynput's macOS `darwin_intercept` hook: it receives each raw
CGEvent before the system does, and returning None drops the event
system-wide while returning it lets it through. This requires the
**Accessibility** permission (interception is more privileged than passive
listening).

SAFETY -- this is the dangerous part of the whole project. A bug that
suppresses while you think you're on the Mac makes your local keyboard feel
dead. Three guards:

  1. Fail OPEN. Any exception in the interceptor lets the key through. A lost
     forward is annoying; a dead local keyboard is a lockout.
  2. A panic switch. A configurable hotkey (default: Esc pressed 3x quickly)
     force-disables suppression regardless of gaze, so you can always get
     your Mac keyboard back without reaching for the process.
  3. Suppression only ever happens when should_forward() is True AND the
     master enable is on. Default state is pass-through.

Chords: the forward gate is checked on the DOWN event; a key forwarded down
is tracked so its UP is always forwarded too, even if gaze drifts mid-chord
(otherwise a modifier could stick on the Linux side). Suppression mirrors
this: if we suppressed a key's down locally, we suppress its up locally too.
"""

from __future__ import annotations

import time
from typing import Callable

from pynput import keyboard

from igaze.keymap import to_linux_key
from igaze.protocol import KeyEvent


class KeyboardCapture:
    def __init__(
        self,
        on_event: Callable[[KeyEvent], None],
        should_forward: Callable[[], bool],
        suppress_local: bool = True,
        panic_key: str = "esc",
        panic_count: int = 3,
        panic_window_s: float = 1.0,
    ) -> None:
        """
        on_event: called with each KeyEvent to forward.
        should_forward: True when gaze target is LINUX.
        suppress_local: if True, block from the Mac the keys we forward, so
            only Linux receives them (true KVM). If False, keys also type on
            the Mac (passive behaviour).
        panic_key / panic_count / panic_window_s: pressing panic_key
            panic_count times within panic_window_s seconds flips suppression
            OFF for the rest of the session -- a guaranteed way to get the
            local keyboard back.
        """
        self._on_event = on_event
        self._should_forward = should_forward
        self._suppress_local = suppress_local
        self._listener: keyboard.Listener | None = None

        # KEY_ names we've forwarded a DOWN for and not yet an UP.
        self._held: set[str] = set()
        # KEY_ names whose DOWN we suppressed locally; their UP must also be
        # suppressed locally so the Mac doesn't see a dangling key-up.
        self._suppressed_down: set[str] = set()

        self._panic_key = panic_key
        self._panic_count = panic_count
        self._panic_window_s = panic_window_s
        self._panic_times: list[float] = []
        self._panic_tripped = False

    def start(self) -> None:
        # darwin_intercept is only consulted on macOS; on other platforms the
        # listener just ignores it, so this stays import-safe elsewhere.
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            darwin_intercept=self._darwin_intercept,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        for key_name in list(self._held):
            self._on_event(KeyEvent(key_name, False))
        self._held.clear()
        self._suppressed_down.clear()

    # --- forwarding (runs from the listener callbacks) ---------------------

    def _on_press(self, key) -> None:
        key_name = self._resolve(key)
        if key_name is None:
            return
        if not self._should_forward():
            return
        self._held.add(key_name)
        self._on_event(KeyEvent(key_name, True))

    def _on_release(self, key) -> None:
        key_name = self._resolve(key)
        if key_name is None:
            return
        if key_name in self._held:
            self._held.discard(key_name)
            self._on_event(KeyEvent(key_name, False))

    # --- local suppression (macOS interceptor) ----------------------------

    def _darwin_intercept(self, event_type, event):
        """Return the event to let it reach the Mac, or None to drop it.

        Must be extremely robust: any error path returns the event (fail
        open), so a bug here can never make the local keyboard go dead.
        """
        try:
            # Track the panic hotkey regardless of everything else, so it
            # works even mid-suppression.
            self._maybe_trip_panic(event_type, event)

            if self._panic_tripped or not self._suppress_local:
                return event  # pass-through

            # Resolve this raw event to a key name using the same keymap.
            key_name = self._resolve_raw(event)
            if key_name is None:
                return event  # unmapped keys always pass locally

            is_down = self._is_key_down(event_type)

            if is_down:
                if self._should_forward():
                    # Going to Linux: block locally.
                    self._suppressed_down.add(key_name)
                    return None
                return event
            else:
                # Key up: suppress locally iff we suppressed its down, so the
                # Mac never sees a half key press.
                if key_name in self._suppressed_down:
                    self._suppressed_down.discard(key_name)
                    return None
                return event
        except Exception:
            # Fail OPEN: never let a bug kill the local keyboard.
            return event

    def _maybe_trip_panic(self, event_type, event) -> None:
        if self._panic_tripped:
            return
        if not self._is_key_down(event_type):
            return
        if self._resolve_raw(event) != to_linux_key(None, self._panic_key):
            return
        now = time.monotonic()
        self._panic_times = [t for t in self._panic_times if now - t <= self._panic_window_s]
        self._panic_times.append(now)
        if len(self._panic_times) >= self._panic_count:
            self._panic_tripped = True

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _resolve(key) -> str | None:
        char = getattr(key, "char", None)
        name = getattr(key, "name", None)
        return to_linux_key(char, name)

    def _resolve_raw(self, event) -> str | None:
        """Resolve a raw CGEvent to a Linux KEY_ name via pynput's own
        keycode tables, so the interceptor and the callbacks agree."""
        try:
            key = self._listener._event_to_key(event)  # pynput internal
        except Exception:
            return None
        char = getattr(key, "char", None)
        name = getattr(key, "name", None)
        return to_linux_key(char, name)

    @staticmethod
    def _is_key_down(event_type) -> bool:
        try:
            import Quartz
            return event_type == Quartz.kCGEventKeyDown
        except Exception:
            return False

#!/usr/bin/env python3
"""igaze injector -- Linux side.

Listens for the key-event stream from the Mac and replays it into a virtual
keyboard via /dev/uinput, so keystrokes (and real chords like Super+Enter)
land in whatever Omarchy/Hyprland window is focused.

Why uinput: Wayland deliberately blocks apps from synthesizing input at the
compositor level. uinput sidesteps that cleanly by creating a virtual device
at the KERNEL level -- the compositor sees a normal keyboard, so this works
under Hyprland, GNOME, KDE, etc. without per-compositor hacks.

Run:
    sudo python3 -m igaze.injector --port 5005

...or set up the udev rule in the README once, add yourself to the input
group, and then run it without sudo.

This is the only part of igaze that needs Linux input hardware, so it's kept
deliberately thin: parse events (protocol.decode), map D/U to evdev 1/0,
write, and syn. All the tricky logic (gating, stuck-key protection) already
happened on the Mac side.
"""

from __future__ import annotations

import argparse
import socket
import sys

try:
    from evdev import UInput, ecodes
except ImportError:
    sys.exit(
        "python-evdev is required on the Linux side:\n"
        "    pip install evdev\n"
    )

from igaze.protocol import decode


def _build_uinput() -> UInput:
    """Create a virtual keyboard that can emit EVERY standard keyboard key.

    Critically, a uinput device can only ever emit keys it declared at
    creation time -- writing an undeclared keycode silently does nothing. An
    earlier version declared only the keys in this machine's copy of the
    keymap, which meant that if the Mac's keymap and the injector's keymap
    drifted (e.g. the injector wasn't restarted after a keymap update), the
    extra keys -- punctuation, symbols, function keys -- were received and
    then silently dropped by the kernel.

    To make injection immune to that entirely, we declare the whole standard
    keyboard here, independent of igaze's keymap. The Mac decides which keys
    to send; the virtual device can always emit whatever arrives.
    """
    # Every KEY_* that evdev knows about, restricted to real single keycodes
    # (skip the KEY_MAX/KEY_CNT sentinels and any non-int entries).
    key_codes = sorted(
        code
        for name, code in ecodes.ecodes.items()
        if isinstance(name, str)
        and name.startswith("KEY_")
        and name not in ("KEY_MAX", "KEY_CNT")
        and isinstance(code, int)
    )

    capabilities = {ecodes.EV_KEY: key_codes}
    try:
        return UInput(capabilities, name="igaze-virtual-keyboard")
    except PermissionError:
        sys.exit(
            "Permission denied opening /dev/uinput.\n"
            "Run with sudo, or set up the udev rule (see README) to run "
            "without sudo."
        )
    except OSError as e:
        sys.exit(f"Could not open /dev/uinput: {e}")


def _serve(ui: UInput, port: int, verbose: bool = False) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"igaze injector listening on 0.0.0.0:{port}  (Ctrl-C to stop)")

    while True:
        conn, addr = srv.accept()
        print(f"Mac connected from {addr[0]}:{addr[1]}")
        # Keys this connection pressed and hasn't released -- so we can flush
        # them if the connection drops mid-chord (no stuck Super on the desk).
        held: set[int] = set()
        buffer = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    _apply(ui, raw.decode("utf-8", "replace"), held, verbose)
        except (ConnectionResetError, OSError):
            pass
        finally:
            _flush_held(ui, held)
            conn.close()
            print("Mac disconnected; released any held keys. Waiting again.")


def _apply(ui: UInput, line: str, held: set[int], verbose: bool = False) -> None:
    event = decode(line)
    if event is None:
        if verbose and line.strip():
            print(f"  ? garbled line ignored: {line!r}")
        return
    code = ecodes.ecodes.get(event.key_name)
    if code is None:
        # Shouldn't happen now that we declare the full keyboard, but if the
        # Mac ever sends a name evdev doesn't know, say so loudly rather than
        # dropping it in silence.
        print(f"  ! unknown key name from Mac, dropped: {event.key_name}")
        return
    ui.write(ecodes.EV_KEY, code, 1 if event.pressed else 0)
    ui.syn()
    if event.pressed:
        held.add(code)
    else:
        held.discard(code)
    if verbose:
        print(f"  {'v' if event.pressed else '^'} {event.key_name}")


def _flush_held(ui: UInput, held: set[int]) -> None:
    for code in list(held):
        ui.write(ecodes.EV_KEY, code, 0)
    if held:
        ui.syn()
    held.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="igaze Linux-side key injector")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each key as it's injected (v = down, ^ = up). Use this "
        "to confirm punctuation etc. is actually arriving.",
    )
    args = parser.parse_args()

    ui = _build_uinput()
    try:
        _serve(ui, args.port, verbose=args.verbose)
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()


if __name__ == "__main__":
    main()

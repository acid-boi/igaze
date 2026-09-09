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

from igaze.keymap import all_mapped_key_names
from igaze.protocol import decode


def _build_uinput() -> UInput:
    """Create a virtual keyboard advertising exactly the keys our keymap can
    emit. Validates every KEY_ name against evdev up front, so a typo in the
    keymap fails loudly here instead of silently dropping keys later."""
    key_codes = []
    for name in all_mapped_key_names():
        code = ecodes.ecodes.get(name)
        if code is None:
            sys.exit(f"keymap has an unknown evdev key name: {name!r}")
        key_codes.append(code)

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


def _serve(ui: UInput, port: int) -> None:
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
                    _apply(ui, raw.decode("utf-8", "replace"), held)
        except (ConnectionResetError, OSError):
            pass
        finally:
            _flush_held(ui, held)
            conn.close()
            print("Mac disconnected; released any held keys. Waiting again.")


def _apply(ui: UInput, line: str, held: set[int]) -> None:
    event = decode(line)
    if event is None:
        return  # skip garbled line rather than crash the stream
    code = ecodes.ecodes.get(event.key_name)
    if code is None:
        return
    ui.write(ecodes.EV_KEY, code, 1 if event.pressed else 0)
    ui.syn()
    if event.pressed:
        held.add(code)
    else:
        held.discard(code)


def _flush_held(ui: UInput, held: set[int]) -> None:
    for code in list(held):
        ui.write(ecodes.EV_KEY, code, 0)
    if held:
        ui.syn()
    held.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="igaze Linux-side key injector")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    ui = _build_uinput()
    try:
        _serve(ui, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()


if __name__ == "__main__":
    main()

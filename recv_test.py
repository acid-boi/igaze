#!/usr/bin/env python3
"""Proper receiver for testing igaze keystroke forwarding.

Unlike `nc -l`, this reassembles the TCP byte stream by newline and prints
one numbered token per line. That makes it timing-independent: if a
keystroke is ever duplicated or dropped, you'll see it in the count and the
sequence, instead of chasing netcat's terminal-rendering artifacts under a
fast burst.

Run this on the LINUX box in place of netcat:

    python3 recv_test.py 5005

Then on the Mac:

    python -m igaze.main --send-to <linux-ip>:5005

Type at full speed and watch the numbered stream. Ctrl-C to stop; it prints
a total count so you can compare against how many keys you actually pressed.
"""

import socket
import sys


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"Listening on 0.0.0.0:{port} ... (Ctrl-C to stop)")

    conn, addr = srv.accept()
    print(f"Connected from {addr[0]}:{addr[1]}\n")

    buffer = b""
    count = 0
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("\n(peer closed the connection)")
                break
            buffer += data
            # Emit every COMPLETE line; keep any trailing partial in buffer.
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                count += 1
                print(f"{count:5d}: {raw.decode('utf-8', 'replace')}")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nTotal tokens received: {count}")
        conn.close()
        srv.close()


if __name__ == "__main__":
    main()

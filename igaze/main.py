"""igaze gaze-detection harness.

First run, calibrate (about 15 seconds):

    python -m igaze.main --calibrate

You'll be asked to look at each screen in turn and press SPACE. This
records what each screen actually measures, which is why nothing here has
a "which side is the Linux laptop" setting -- calibration determines the
direction, so it can't be inverted.

Then run it:

    python -m igaze.main

The window shows a progress bar: 0% = looking at the Mac, 100% = looking at
the Linux laptop, with the switch point marked. The terminal prints a line
on each actual switch. Press 'q' to quit.

Pose is measured on the RAW frame; the mirror is applied only to the
display (and overlay coordinates are mirrored to match). Measuring a
mirrored frame would flip the sign of the signal.
"""

import argparse
from pathlib import Path

import cv2

from igaze.calibration import DEFAULT_PATH, Calibration
from igaze.face_tracker import FaceTracker
from igaze.gaze_state import GazeStateMachine, GazeTarget

_SAMPLES_PER_SCREEN = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Run calibration, then exit.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--switch-at", type=float, default=0.5)
    parser.add_argument("--hysteresis", type=float, default=0.15)
    parser.add_argument("--debounce-frames", type=int, default=3)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    parser.add_argument(
        "--no-adapt-distance",
        action="store_true",
        help="Disable rescaling the threshold as you move nearer/further.",
    )
    parser.add_argument(
        "--no-face-fallback-frames",
        type=int,
        default=0,
        help="Revert to MAC after N no-face frames. 0 (default) holds.",
    )
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--model-path", default="face_landmarker.task")
    parser.add_argument(
        "--send-to",
        metavar="HOST:PORT",
        help=(
            "Forward keystrokes to this TCP endpoint while gaze is LINUX. "
            "Test with `nc -l <port>` on the Linux box. Omit to run "
            "detection only, with no capture or networking."
        ),
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")

    tracker = FaceTracker(model_path=args.model_path)
    try:
        if args.calibrate:
            run_calibration(cap, tracker, args.calibration_path)
            return

        calibration = Calibration.load(args.calibration_path)
        if calibration is None:
            raise SystemExit(
                f"No calibration found at {args.calibration_path}.\n"
                "Run:  python -m igaze.main --calibrate"
            )
        run_live(cap, tracker, calibration, args)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


def run_calibration(cap, tracker: FaceTracker, path: Path) -> None:
    print("Calibration. Sit the way you normally would while working.\n")
    ratio_mac, scale_mac = _collect(cap, tracker, "MAC (this screen)")
    if ratio_mac is None:
        print("Calibration cancelled.")
        return
    ratio_linux, _ = _collect(cap, tracker, "LINUX laptop")
    if ratio_linux is None:
        print("Calibration cancelled.")
        return

    calibration = Calibration(
        ratio_mac=ratio_mac, ratio_linux=ratio_linux, face_scale=scale_mac
    )
    if abs(calibration.span) < 0.05:
        print(
            f"\nThe two readings are nearly identical "
            f"({ratio_mac:+.3f} vs {ratio_linux:+.3f}). That usually means "
            "the head turn was too small to distinguish. Nothing saved -- "
            "please re-run and turn fully toward each screen."
        )
        return

    calibration.save(path)
    side = "right" if calibration.span > 0 else "left"
    print(f"\nSaved to {path}")
    print(f"  looking at Mac:   {ratio_mac:+.3f}")
    print(f"  looking at Linux: {ratio_linux:+.3f}  (detected: on your {side})")
    print("\nNow run:  python -m igaze.main")


def _collect(cap, tracker: FaceTracker, label: str) -> tuple[float | None, float | None]:
    """Show a live preview until SPACE, then average a few frames."""
    print(f"Look at the {label}, then press SPACE.  (ESC cancels)")
    samples: list[tuple[float, float]] = []
    capturing = False

    while True:
        ok, frame = cap.read()
        if not ok:
            return None, None

        signal = tracker.measure(frame)
        display = cv2.flip(frame, 1)

        if capturing:
            if signal is not None:
                samples.append((signal.yaw_ratio, signal.face_scale))
            _text(display, f"capturing... {len(samples)}/{_SAMPLES_PER_SCREEN}", 40, (0, 200, 255))
            if len(samples) >= _SAMPLES_PER_SCREEN:
                break
        else:
            _text(display, f"Look at {label}", 40, (0, 200, 0))
            _text(
                display,
                "SPACE to capture, ESC to cancel" if signal else "no face detected",
                70,
                (255, 255, 255) if signal else (0, 0, 255),
                scale=0.6,
            )

        cv2.imshow("igaze - calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return None, None
        if key == 32 and signal is not None:  # SPACE
            capturing = True

    ratios = [r for r, _ in samples]
    scales = [s for _, s in samples]
    return sum(ratios) / len(ratios), sum(scales) / len(scales)


def run_live(cap, tracker: FaceTracker, calibration: Calibration, args) -> None:
    machine = GazeStateMachine(
        calibration=calibration,
        switch_at=args.switch_at,
        hysteresis=args.hysteresis,
        debounce_frames=args.debounce_frames,
        smoothing_alpha=args.smoothing_alpha,
        adapt_to_distance=not args.no_adapt_distance,
        no_face_fallback_frames=args.no_face_fallback_frames,
    )

    side = "right" if calibration.span > 0 else "left"
    print(f"Calibrated: Linux laptop is on your {side}.")

    sender = None
    capture = None
    if args.send_to:
        # Imported lazily so detection-only runs don't need pynput installed.
        from igaze.gaze_state import GazeTarget
        from igaze.keyboard_capture import KeyboardCapture
        from igaze.sender import KeystrokeSender

        host, port = _parse_endpoint(args.send_to)
        sender = KeystrokeSender(host, port)
        sender.start()
        # The capture asks this each keypress: forward only when gaze == LINUX.
        capture = KeyboardCapture(
            on_event=lambda ev: sender.send_line(ev.encode()),
            should_forward=lambda: machine.current == GazeTarget.LINUX,
        )
        capture.start()
        print(f"Forwarding keystrokes to {host}:{port} while gaze is LINUX.")
        print(f"On the Linux box, run:  nc -l {port}")
        print("(First run will prompt for macOS Input Monitoring permission.)")

    print("Press 'q' in the video window to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            signal = tracker.measure(frame)  # raw frame, never mirrored
            target, changed = machine.update(
                signal.yaw_ratio if signal else None,
                signal.face_scale if signal else None,
            )
            if changed:
                print(f"[SWITCH] -> {target.value.upper()}")

            display = cv2.flip(frame, 1)
            _draw_overlay(display, signal, machine, target, sender)
            cv2.imshow("igaze", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if capture is not None:
            capture.stop()
        if sender is not None:
            sender.stop()


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if ":" not in endpoint:
        raise SystemExit(f"--send-to must be HOST:PORT, got {endpoint!r}")
    host, _, port_str = endpoint.rpartition(":")
    try:
        return host, int(port_str)
    except ValueError:
        raise SystemExit(f"Invalid port in --send-to: {port_str!r}")


def _draw_overlay(display, signal, machine: GazeStateMachine, target: GazeTarget, sender=None) -> None:
    width = display.shape[1]
    color = (0, 200, 0) if target == GazeTarget.MAC else (0, 140, 255)
    _text(display, f"TARGET: {target.value.upper()}", 40, color)

    if sender is not None:
        if sender.is_connected:
            fwd = target == GazeTarget.LINUX
            msg = "link up - FORWARDING" if fwd else "link up - idle (looking at Mac)"
            col = (0, 140, 255) if fwd else (150, 150, 150)
        else:
            msg = "link down - waiting for listener"
            col = (0, 0, 255)
        _text(display, msg, 120, col, scale=0.55)

    if signal is None:
        _text(display, "no face - holding target", 70, (0, 0, 255), scale=0.6)
        return

    progress = machine.progress if machine.progress is not None else 0.0
    _text(
        display,
        f"progress {progress * 100:5.1f}%   ratio {signal.yaw_ratio:+.3f}",
        70,
        (255, 255, 255),
        scale=0.6,
    )

    # Progress bar: 0% = Mac, 100% = Linux, tick marks the switch point.
    x0, x1, y = 20, min(320, width - 20), 95
    cv2.rectangle(display, (x0, y), (x1, y + 14), (70, 70, 70), -1)
    filled = int(x0 + max(0.0, min(1.0, progress)) * (x1 - x0))
    cv2.rectangle(display, (x0, y), (filled, y + 14), color, -1)
    tick = int(x0 + machine.switch_at * (x1 - x0))
    cv2.line(display, (tick, y - 4), (tick, y + 18), (255, 255, 255), 2)

    for x, py in signal.landmarks_2d:
        cv2.circle(display, (int(width - x), int(py)), 3, (0, 255, 255), -1)


def _text(img, message: str, y: int, color, scale: float = 0.9) -> None:
    cv2.putText(
        img, message, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale > 0.7 else 1
    )


if __name__ == "__main__":
    main()

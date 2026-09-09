# igaze — gaze detection (phase 1)

Detects which of two screens you're looking at, using the Mac's webcam.
No networking or keystroke forwarding yet — this is the detection layer,
built and tuned in isolation first.

## Setup

```bash
python3.12 -m venv venv       # mediapipe has no wheels for 3.14 yet
source venv/bin/activate
pip install -r requirements.txt

python3 -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')"
```

## Calibrate, then run

```bash
python -m igaze.main --calibrate   # ~15s, once
python -m igaze.main
```

Calibration asks you to look at each screen and press SPACE. Sit the way
you normally would while working.

The live window shows a progress bar: 0% = looking at the Mac, 100% =
looking at the Linux laptop, with a tick at the switch point. The terminal
prints on each switch. Press `q` to quit.

## How it works

**Direct geometry, not 3D pose.** When you turn your head, your nose shifts
toward the eye corners on that side. That's measured directly: where the
nose sits between the two outer eye corners, projected onto the eye axis so
head tilt doesn't corrupt it. No `solvePnP`, no camera intrinsics, no 3D
face model, no Euler angles — all of which are fragile overhead for what is
ultimately a binary decision.

**Calibration removes the direction problem.** Rather than configuring
which side the Linux laptop is on and hoping the sign convention is right,
calibration measures both screens. Readings are remapped onto a 0..1
"progress toward Linux" axis; because the mapping is signed, a laptop on
your left behaves identically to one on your right. There is no sign left
to get backwards.

**Distance is compensated automatically.** Two screens a fixed distance
apart subtend a wider angle when you sit close and a narrower one when you
sit back, so a fixed threshold is only correct at one seating distance. The
nose-offset ratio tracks `tan(yaw)`, and `tan(yaw) = offset / distance`, so
the threshold scales as `1/distance`. Face size on screen is itself
proportional to `1/distance`, so scaling the threshold by the ratio of
current to calibrated face size is exactly the right correction — and the
physical separation between the laptops cancels out, so you never measure
it. Disable with `--no-adapt-distance`.

Face scale is measured from the eye line down to the chin, deliberately.
The obvious choice — distance between the eye corners — shrinks as you turn
your head, which would confuse "turned away" with "sitting further back".
A vertical extent is unaffected by yaw.

**Face loss holds the current target.** Turning far enough to look at the
other laptop often means the webcam can no longer see your face, so
treating "no face" as "go back to the Mac" fires exactly when the feature
is working. Pass `--no-face-fallback-frames 300` (~10s) if you want a
revert after you actually walk away.

## Forwarding keystrokes + chords (phase 2)

Keystrokes are forwarded to the Linux box as a **key-event timeline** (each
key's down and up), not as characters. That's what makes real shortcuts
work: `Cmd+Enter` on the Mac is replayed as `Super+Enter` on Omarchy and
opens a terminal, because Super is genuinely held down while Enter fires.

**macOS Cmd maps to Linux Super** (`KEY_LEFTMETA`). The current keymap is a
deliberately minimal set: letters, digits, Enter/Space/Tab/Backspace, and
the Cmd/Shift/Ctrl/Alt modifier families. Keys outside that set are dropped
for now (punctuation, function keys, arrows come later).

### On the Linux laptop: run the injector

The injector writes into a virtual keyboard via `/dev/uinput`, which is why
it works under Wayland/Hyprland (the kernel-level device sidesteps the
compositor's synthetic-input block).

```bash
pip install evdev
sudo python3 -m igaze.injector --port 5005
```

`sudo` is the zero-setup path. To run without it, add the udev rule below
once.

### On the Mac: forward to it

```bash
python -m igaze.main --send-to <linux-ip>:5005
```

Look at the Mac -> nothing is sent. Look at the Linux laptop -> keystrokes
and chords land in the focused Omarchy window. The first run triggers the
macOS **Input Monitoring** prompt (System Settings -> Privacy & Security);
capture is passive, so your Mac keeps working normally.

### Running the injector without sudo (one-time)

```bash
# As root, once:
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' \
  | sudo tee /etc/udev/rules.d/99-igaze-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG input $USER
# then log out and back in so the group membership takes effect
```

After that, `python3 -m igaze.injector --port 5005` runs without sudo.

### Safety notes built in

* If your gaze drifts back to the Mac *mid-chord*, the key's release is
  still forwarded, so you never get a stuck Super/Ctrl on the Linux side.
* If the connection drops while keys are held, the injector releases them.
* `recv_test.py` is still included as a no-hardware receiver if you want to
  eyeball the raw event stream (`python3 recv_test.py 5005`) instead of
  actually injecting.

Without `--send-to`, igaze runs detection-only with no capture or
networking.

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--switch-at` | 0.5 | Progress fraction where the switch fires. Raise to require a fuller turn. |
| `--hysteresis` | 0.15 | Gap between switch-in and fall-back points. Raise if it flickers. |
| `--debounce-frames` | 3 | Frames a candidate must hold. |
| `--smoothing-alpha` | 0.5 | EMA weight; 1.0 disables smoothing, lower is smoother but laggier. |

If detection feels off, re-run `--calibrate` — most tuning problems are
really calibration problems.

## Files

- `igaze/face_tracker.py` — landmarks → `FaceSignal(yaw_ratio, face_scale)`
- `igaze/calibration.py` — persisted per-screen readings (`igaze_calibration.json`)
- `igaze/gaze_state.py` — calibrated thresholding, distance adaptation, smoothing/hysteresis/debounce
- `igaze/keymap.py` — macOS keys -> Linux KEY_ names, position-based, Cmd->Super
- `igaze/keyboard_capture.py` — passive pynput listener (press+release), gaze-gated, stuck-key protection
- `igaze/protocol.py` — key-event wire format (KEY_ name + down/up)
- `igaze/sender.py` — reconnecting TCP client, batch-drains the queue under fast typing
- `igaze/injector.py` — **Linux side**: replays the event timeline into /dev/uinput
- `igaze/main.py` — calibration flow + live harness + forwarding
- `recv_test.py` — Linux-side debug receiver (no hardware; prints the raw stream)

## macOS notes

`face_tracker.py` forces the CPU delegate. The GPU/Metal path can hard-crash
with `Check failed: service_ Service is unavailable.` outside a proper
GPU-context app. CPU is plenty fast for one face.

`requirements.txt` pins `mediapipe==0.10.14`, which needs Python ≤3.13.
Newer releases added macOS GPU crashes and broke the legacy `mp.solutions`
API.

## Next steps

Capture, forwarding, and uinput injection with chord support are wired
in. Next: expand the keymap (punctuation, arrows, function keys), then
encryption/authentication on the socket.

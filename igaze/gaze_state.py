"""Turns the per-frame face signal into a stable MAC/LINUX target.

Everything here works in *calibrated* units. The raw yaw_ratio is remapped
onto a "progress toward the Linux laptop" axis:

    progress = (yaw_ratio - ratio_mac) / (ratio_linux - ratio_mac)

So 0.0 means "looking where you were when you calibrated the Mac" and 1.0
means "looking where you were when you calibrated the Linux laptop". The
switch fires around the midpoint. Because the divisor is signed, this works
identically whether the Linux laptop is to your left or your right -- there
is no sign to configure and therefore no direction to get inverted.


Distance compensation
---------------------
Two screens a fixed distance apart subtend a LARGER angle when you sit
close and a SMALLER one when you sit far back. So a threshold that is right
at one seating distance is wrong at every other one.

The correction is simpler than it looks. The nose-offset ratio tracks
tan(yaw), and for a screen a lateral distance `offset` away at viewing
distance `d`:

    tan(yaw_to_screen) = offset / d

so the ratio corresponding to "looking at the Linux laptop" scales as 1/d.
Face scale on screen is itself proportional to 1/d, so:

    span_now = span_at_calibration * (face_scale_now / face_scale_at_cal)

The physical separation between the laptops cancels out entirely -- you
never have to measure it, and no camera calibration is needed. Only the
RATIO of face sizes matters, which is why face_scale is measured on a
yaw-invariant axis (see face_tracker.py).

Stability comes from three layers: EMA smoothing on the raw ratio,
hysteresis between the switch-in and fall-back points, and a debounce
count.

Face loss holds the current target. Turning far enough to look at the other
laptop frequently means the webcam can no longer see your face -- treating
that as "go back to the Mac" would fire exactly when the feature is working.
"""

from enum import Enum

from igaze.calibration import Calibration


class GazeTarget(Enum):
    MAC = "mac"
    LINUX = "linux"


class GazeStateMachine:
    def __init__(
        self,
        calibration: Calibration,
        switch_at: float = 0.5,
        hysteresis: float = 0.15,
        debounce_frames: int = 3,
        smoothing_alpha: float = 0.5,
        adapt_to_distance: bool = True,
        no_face_fallback_frames: int = 0,
    ) -> None:
        """
        calibration: measured readings for each screen.
        switch_at: progress fraction at which MAC -> LINUX fires. 0.5 is the
            midpoint between the two calibrated readings.
        hysteresis: how much lower the fall-back point sits, in progress
            units, to stop flicker at the boundary.
        debounce_frames: consecutive frames a candidate must hold.
        smoothing_alpha: EMA weight per new sample (0 < a <= 1); lower is
            smoother but laggier.
        adapt_to_distance: rescale the threshold as you move nearer/further.
        no_face_fallback_frames: if > 0, revert to MAC after this many
            consecutive no-face frames. 0 (default) holds indefinitely.
        """
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if abs(calibration.span) < 1e-6:
            raise ValueError(
                "Calibration span is ~0: the two screens measured the same. "
                "Re-run calibration, turning fully toward each screen."
            )

        self.calibration = calibration
        self.switch_at = switch_at
        self.hysteresis = hysteresis
        self.debounce_frames = debounce_frames
        self.smoothing_alpha = smoothing_alpha
        self.adapt_to_distance = adapt_to_distance
        self.no_face_fallback_frames = no_face_fallback_frames

        self.current = GazeTarget.MAC
        self.smoothed_ratio: float | None = None
        self.progress: float | None = None
        self._pending: GazeTarget | None = None
        self._pending_count = 0
        self._no_face_count = 0

    def update(
        self, yaw_ratio: float | None, face_scale: float | None = None
    ) -> tuple[GazeTarget, bool]:
        """Feed one frame. Pass None for yaw_ratio when no face was found.
        Returns (current_target, changed_this_call)."""
        if yaw_ratio is None:
            return self._handle_no_face()

        self._no_face_count = 0

        if self.smoothed_ratio is None:
            self.smoothed_ratio = yaw_ratio
        else:
            a = self.smoothing_alpha
            self.smoothed_ratio = a * yaw_ratio + (1.0 - a) * self.smoothed_ratio

        span = self.calibration.span
        if self.adapt_to_distance and face_scale and self.calibration.face_scale > 0:
            # Sitting closer => face_scale larger => the other screen sits at
            # a wider angle => a bigger ratio is needed to count as looking
            # at it. See the module docstring.
            span *= face_scale / self.calibration.face_scale

        self.progress = (self.smoothed_ratio - self.calibration.ratio_mac) / span
        candidate = self._candidate_target(self.progress)

        if candidate == self.current:
            self._pending = None
            self._pending_count = 0
            return self.current, False

        if candidate == self._pending:
            self._pending_count += 1
        else:
            self._pending = candidate
            self._pending_count = 1

        if self._pending_count >= self.debounce_frames:
            self.current = candidate
            self._pending = None
            self._pending_count = 0
            return self.current, True

        return self.current, False

    def _candidate_target(self, progress: float) -> GazeTarget:
        if self.current == GazeTarget.MAC:
            return GazeTarget.LINUX if progress > self.switch_at else GazeTarget.MAC
        fall_back_at = self.switch_at - self.hysteresis
        return GazeTarget.MAC if progress < fall_back_at else GazeTarget.LINUX

    def _handle_no_face(self) -> tuple[GazeTarget, bool]:
        # `_pending` is deliberately NOT cleared. If the face was lost
        # mid-turn, the last frames already voted for the new target;
        # discarding that would make turning away fail to register exactly
        # when the turn is big enough to lose tracking.
        self._no_face_count += 1

        if (
            self.no_face_fallback_frames > 0
            and self._no_face_count >= self.no_face_fallback_frames
            and self.current != GazeTarget.MAC
        ):
            self.current = GazeTarget.MAC
            self._pending = None
            self._pending_count = 0
            self._no_face_count = 0
            return self.current, True

        return self.current, False

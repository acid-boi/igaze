"""Direct geometric gaze signal from MediaPipe face landmarks.

This deliberately does NOT use solvePnP. Recovering a full 3D head pose
needs camera intrinsics we don't have, a generic 3D face model that isn't
the user's face, and a Euler decomposition whose sign conventions are easy
to get backwards. For a binary "which screen am I looking at?" decision,
all of that machinery is fragile overhead.

Instead we measure the thing directly: when you turn your head, your nose
shifts toward the eye/mouth corners on that side. Two measurements per
frame:

  yaw_ratio   Where the nose sits between the two outer eye corners,
              projected onto the eye axis so head TILT doesn't corrupt it.
              0.0 = nose centered between the eyes. Positive = nose has
              moved toward the image-LEFT side, which happens when the head
              turns toward the person's own right. Normalized by eye
              separation, so it's scale-invariant: moving nearer or further
              from the camera doesn't change it by itself.

              Geometrically this tracks tan(yaw), not yaw. That turns out
              to be exactly what we want -- see gaze_state.py, where it
              makes the distance compensation fall out for free.

  face_scale  Vertical distance from the eye-line midpoint down to the
              chin, in pixels. Used as an inverse proxy for how far the
              user is sitting from the camera.

              Note this is measured VERTICALLY on purpose. The obvious
              scale proxy -- distance between the eye corners -- shrinks as
              you turn your head (foreshortening), so it would conflate
              "turned away" with "sitting further back". A vertical extent
              is unaffected by yaw.

No absolute units and no camera calibration are needed anywhere: everything
downstream uses these two numbers only in ratios against their own
calibrated values.
"""

from dataclasses import dataclass
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# MediaPipe FaceLandmarker landmark indices.
# We never hardcode which of a symmetric pair is "left" vs "right" --
# MediaPipe's anatomical naming (the person's left) is the mirror of
# image-space left, and conflating them silently inverts everything. Pairs
# are sorted by x at runtime instead.
_NOSE_TIP = 1
_CHIN = 152
_EYE_OUTER_CORNERS = (33, 263)


@dataclass
class FaceSignal:
    yaw_ratio: float  # ~tan(yaw); + = head turned toward person's own right
    face_scale: float  # px, eye-line to chin; larger = sitting closer
    landmarks_2d: np.ndarray  # points used, for the debug overlay


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker and turns frames into a FaceSignal."""

    def __init__(self, model_path: str = "face_landmarker.task") -> None:
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            # Force CPU: on macOS the GPU/Metal delegate can hard-crash with
            # "Check failed: service_ Service is unavailable." outside a
            # proper GPU-context app. CPU is plenty fast for one face.
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()

    def close(self) -> None:
        self._detector.close()

    def measure(self, frame_bgr: np.ndarray) -> FaceSignal | None:
        """Measure the RAW (un-mirrored) BGR frame. Returns None if no face.

        Must be given the un-mirrored frame: mirroring the image mirrors the
        geometry and flips the sign of yaw_ratio.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # detect_for_video needs strictly increasing timestamps.
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        def px(idx: int) -> np.ndarray:
            return np.array(
                [landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float64
            )

        nose = px(_NOSE_TIP)
        chin = px(_CHIN)
        eye_a, eye_b = sorted((px(i) for i in _EYE_OUTER_CORNERS), key=lambda p: p[0])

        eye_vector = eye_b - eye_a
        eye_separation = float(np.linalg.norm(eye_vector))
        if eye_separation < 1e-6:
            return None

        # Project the nose onto the eye axis. Using a projection rather than
        # raw x-difference means head roll (tilting your head sideways)
        # doesn't leak into the measurement.
        axis = eye_vector / eye_separation
        t = float(np.dot(nose - eye_a, axis)) / eye_separation  # 0..1, .5 = centered

        # t < 0.5 means the nose sits toward the image-left eye, which is
        # what happens when the head turns toward the person's own right.
        # Flip so that "turned to own right" reads positive.
        yaw_ratio = 1.0 - 2.0 * t

        # Vertical extent: yaw-invariant, so it tracks viewing distance
        # without being polluted by head turn.
        eye_midpoint = (eye_a + eye_b) / 2.0
        face_scale = float(abs(chin[1] - eye_midpoint[1]))
        if face_scale < 1e-6:
            return None

        points = np.array([nose, chin, eye_a, eye_b], dtype=np.float64)
        return FaceSignal(yaw_ratio=yaw_ratio, face_scale=face_scale, landmarks_2d=points)

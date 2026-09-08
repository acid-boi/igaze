"""Persisted calibration: what "looking at each screen" actually measures.

Rather than hardcoding a threshold angle and a sign for which side the
Linux laptop is on, we measure both once. This removes the entire class of
inverted-direction bugs: the direction is whatever the calibration recorded,
so it cannot be backwards.

Stored values:
    ratio_mac     yaw_ratio while looking at the Mac (your neutral -- rarely
                  exactly 0, since nobody sits perfectly square to the
                  camera, and faces aren't symmetric)
    ratio_linux   yaw_ratio while looking at the Linux laptop
    face_scale    face scale at calibration time, so distance changes can be
                  compensated later
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

DEFAULT_PATH = Path("igaze_calibration.json")


@dataclass
class Calibration:
    ratio_mac: float
    ratio_linux: float
    face_scale: float

    @property
    def span(self) -> float:
        """Signed distance from the Mac reading to the Linux reading."""
        return self.ratio_linux - self.ratio_mac

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Calibration | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(
                ratio_mac=float(data["ratio_mac"]),
                ratio_linux=float(data["ratio_linux"]),
                face_scale=float(data["face_scale"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

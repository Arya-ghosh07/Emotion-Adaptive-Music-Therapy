"""Heart rate and HRV from a Bluetooth heart-rate strap (standard 0x180D profile).

The browser reads the strap via Web Bluetooth and forwards beats-per-minute
plus RR intervals (ms between beats). RR intervals give HRV, the best simple
marker of stress: lower RMSSD = more stressed.
"""
from collections import deque
from typing import Optional

import numpy as np

WINDOW_SECONDS = 60.0


def rmssd(rr_ms) -> Optional[float]:
    """Root mean square of successive RR differences, in ms."""
    rr = np.asarray([x for x in rr_ms if 300 <= x <= 2000], dtype=float)  # drop artefacts
    if len(rr) < 3:
        return None
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


class BLEHeartRate:
    def __init__(self):
        self.hr = deque()   # (t, bpm)
        self.rr = deque()   # (t, rr_ms)

    def add(self, t_s: float, bpm: Optional[float], rr_ms=()):
        if bpm:
            self.hr.append((t_s, float(bpm)))
        for x in rr_ms or ():
            self.rr.append((t_s, float(x)))
        cutoff = t_s - WINDOW_SECONDS
        for q in (self.hr, self.rr):
            while q and q[0][0] < cutoff:
                q.popleft()

    def latest(self) -> Optional[dict]:
        if not self.hr:
            return None
        recent = [b for t, b in self.hr if t >= self.hr[-1][0] - 5]
        return {
            "bpm": round(float(np.mean(recent)), 1),
            "rmssd": None if (v := rmssd([r for _, r in self.rr])) is None else round(v, 1),
            "quality": 1.0,
            "reliable": True,
            "source": "ble",
        }

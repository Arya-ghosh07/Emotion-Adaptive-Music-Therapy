"""How well did a music segment work? Measured from the body, not a guess.

For each ~90 s segment we compare heart rate in its first and last 30 s.
A drop means the music helped the user settle. HRV (RMSSD) rising also counts
when a Bluetooth strap provides it. An optional thumbs up/down is blended in.
Segments with poor webcam signal quality are marked invalid so noise never
trains the model.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

EDGE_SECONDS = 30.0
BPM_FOR_FULL_REWARD = 8.0     # an 8 BPM drop -> reward 1.0, an 8 BPM rise -> 0.0
RMSSD_FOR_FULL_REWARD = 15.0  # ms
MIN_RELIABLE_SHARE = 0.5
MIN_SAMPLES = 3
THUMBS_WEIGHT = 0.3

# (t_seconds, bpm, quality_ok, rmssd_or_None)
Sample = Tuple[float, float, bool, Optional[float]]


def _window(samples: Sequence[Sample], t0: float, t1: float):
    win = [s for s in samples if t0 <= s[0] <= t1]
    good = [s for s in win if s[2]]
    return win, good


def segment_reward(samples: Sequence[Sample], start: float, end: float,
                   thumbs: Optional[int] = None) -> dict:
    """thumbs: +1, -1 or None."""
    first, first_ok = _window(samples, start, start + EDGE_SECONDS)
    last, last_ok = _window(samples, end - EDGE_SECONDS, end)

    def enough(win, ok):
        return len(ok) >= MIN_SAMPLES and len(ok) >= MIN_RELIABLE_SHARE * max(1, len(win))

    result = {"hr_start": None, "hr_end": None, "rmssd_start": None, "rmssd_end": None}
    phys = None
    if enough(first, first_ok) and enough(last, last_ok):
        hr_start = float(np.median([s[1] for s in first_ok]))
        hr_end = float(np.median([s[1] for s in last_ok]))
        phys = float(np.clip(0.5 + (hr_start - hr_end) / (2 * BPM_FOR_FULL_REWARD), 0, 1))
        result.update(hr_start=round(hr_start, 1), hr_end=round(hr_end, 1))

        r0 = [s[3] for s in first_ok if s[3] is not None]
        r1 = [s[3] for s in last_ok if s[3] is not None]
        if r0 and r1:
            d = float(np.median(r1) - np.median(r0))
            hrv = float(np.clip(0.5 + d / (2 * RMSSD_FOR_FULL_REWARD), 0, 1))
            phys = 0.5 * phys + 0.5 * hrv
            result.update(rmssd_start=round(float(np.median(r0)), 1),
                          rmssd_end=round(float(np.median(r1)), 1))

    thumb_r = None if thumbs is None else (1.0 if thumbs > 0 else 0.0)

    if phys is not None and thumb_r is not None:
        reward, basis = (1 - THUMBS_WEIGHT) * phys + THUMBS_WEIGHT * thumb_r, "physiology+thumbs"
    elif phys is not None:
        reward, basis = phys, "physiology"
    elif thumb_r is not None:
        reward, basis = thumb_r, "thumbs"
    else:
        return {**result, "reward": None, "valid": False, "basis": "low_signal_quality"}
    return {**result, "reward": round(reward, 4), "valid": True, "basis": basis}

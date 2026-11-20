"""Fuse sensors into one valence/arousal estimate, trusting each one as much as
it has earned *for this user*.

Every time the user self-reports how they feel (two sliders), each sensor's
latest reading is compared with the report. A running error is kept per
sensor and per dimension, and weights are proportional to 1 / (error + eps).
So if your face is a poor guide to how you feel but your heart rate is a good
guide to how activated you are, the system learns exactly that.

Heart rate only says how activated someone is (arousal), so it is never used
for valence.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

MODALITIES = ("face", "text", "heart")
DIMS = ("valence", "arousal")
PRIOR_ERROR = {  # starting assumptions before any self-report, in V-A units (range 2)
    ("face", "valence"): 0.45, ("face", "arousal"): 0.55,
    ("text", "valence"): 0.40, ("text", "arousal"): 0.55,
    ("heart", "arousal"): 0.40,
}
EPS = 0.1
ERROR_LEARNING_RATE = 0.3
STALE_SECONDS = {"face": 15.0, "heart": 15.0, "text": 600.0}
SMOOTHING = 0.4


@dataclass
class ModalityReading:
    source: str
    valence: Optional[float]
    arousal: Optional[float]
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


def va_to_label(v: float, a: float):
    if v >= 0 and a >= 0:
        quadrant = "excited"
        label = "elated" if v > 0.5 and a > 0.5 else "happy" if v > 0.3 else "alert"
    elif v < 0 and a >= 0:
        quadrant = "stressed"
        label = "angry" if a > 0.6 and v < -0.4 else "anxious" if a > 0.4 else "tense"
    elif v < 0 and a < 0:
        quadrant = "low"
        label = "depressed" if v < -0.5 else "fatigued" if a < -0.5 else "sad"
    else:
        quadrant = "calm"
        label = "content" if v > 0.5 else "relaxed" if v > 0.2 else "neutral"
    return label, quadrant


def sam_to_va(value_1_to_9: float) -> float:
    """Self-Assessment Manikin slider (1..9) -> [-1, 1]."""
    return float(np.clip((float(value_1_to_9) - 5.0) / 4.0, -1, 1))


class ReliabilityModel:
    """Per-user running error for each (sensor, dimension)."""

    def __init__(self, errors: Optional[Dict] = None):
        self.errors = dict(PRIOR_ERROR)
        self.counts = {k: 0 for k in PRIOR_ERROR}
        for (m, d), (err, n) in (errors or {}).items():
            if (m, d) in self.errors:
                self.errors[(m, d)] = err
                self.counts[(m, d)] = n

    def weight(self, modality: str, dim: str) -> float:
        err = self.errors.get((modality, dim))
        return 0.0 if err is None else 1.0 / (err + EPS)

    def update(self, readings: Dict[str, ModalityReading], reported: Dict[str, float]):
        changed = []
        for m, r in readings.items():
            for d in DIMS:
                key = (m, d)
                val = getattr(r, d)
                if key not in self.errors or val is None:
                    continue
                e = abs(val - reported[d])
                self.errors[key] = (1 - ERROR_LEARNING_RATE) * self.errors[key] + ERROR_LEARNING_RATE * e
                self.counts[key] += 1
                changed.append((m, d, round(self.errors[key], 4), self.counts[key]))
        return changed

    def normalised_weights(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for d in DIMS:
            ws = {m: self.weight(m, d) for m in MODALITIES if (m, d) in self.errors}
            total = sum(ws.values()) or 1.0
            out[d] = {m: round(w / total, 4) for m, w in ws.items()}
        return out


class FusionEngine:
    def __init__(self, reliability: Optional[ReliabilityModel] = None):
        self.reliability = reliability or ReliabilityModel()
        self.readings: Dict[str, ModalityReading] = {}
        self._smoothed = None

    def add_reading(self, reading: ModalityReading):
        self.readings[reading.source] = reading

    def fresh_readings(self, now: Optional[float] = None) -> Dict[str, ModalityReading]:
        now = now or time.time()
        return {m: r for m, r in self.readings.items()
                if now - r.timestamp <= STALE_SECONDS.get(m, 30.0)}

    def fuse(self, now: Optional[float] = None) -> Optional[dict]:
        fresh = self.fresh_readings(now)
        if not fresh:
            return None
        fused, contributions = {}, {}
        for d in DIMS:
            num = den = 0.0
            contrib = {}
            for m, r in fresh.items():
                val = getattr(r, d)
                if val is None:
                    continue
                w = self.reliability.weight(m, d) * max(0.0, r.confidence)
                num += w * val
                den += w
                contrib[m] = w
            fused[d] = num / den if den > 0 else None
            contributions[d] = {m: round(w / den, 3) for m, w in contrib.items()} if den > 0 else {}
        if fused["valence"] is None and fused["arousal"] is None:
            return None
        v = fused["valence"] if fused["valence"] is not None else 0.0
        a = fused["arousal"] if fused["arousal"] is not None else 0.0
        if self._smoothed is None:
            self._smoothed = (v, a)
        else:
            pv, pa = self._smoothed
            self._smoothed = (SMOOTHING * v + (1 - SMOOTHING) * pv, SMOOTHING * a + (1 - SMOOTHING) * pa)
        sv, sa = (round(float(np.clip(x, -1, 1)), 4) for x in self._smoothed)
        label, quadrant = va_to_label(sv, sa)
        return {
            "valence": sv, "arousal": sa,
            "emotion_label": label, "quadrant": quadrant,
            "contributions": contributions,
            "valence_known": fused["valence"] is not None,
        }

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def synthetic_skin_samples(bpm, seconds, t0_ms=0.0, fps=30.0, jitter_ms=6.0,
                           pulse_amp=0.6, noise=0.4, seed=0):
    """Mean skin RGB per frame with a pulse at `bpm`, uneven frame timing,
    sensor noise and slow lighting drift - roughly what a webcam sees."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = t0_ms + np.cumsum(np.full(n, 1000.0 / fps) + rng.normal(0, jitter_ms, n))
    ts = t / 1000.0
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * ts)
    drift = 3.0 * np.sin(2 * np.pi * 0.05 * ts)
    base = np.array([160.0, 120.0, 100.0])
    weights = np.array([0.35, 1.0, 0.25])  # pulse is strongest in green
    rgb = base + drift[:, None] + pulse_amp * pulse[:, None] * weights + rng.normal(0, noise, (n, 3))
    return [[float(ti), *map(float, c)] for ti, c in zip(t, rgb)]


class PulseStream:
    """Continuous-phase synthetic webcam stream whose heart rate can change
    over time (like a real pulse), emitted in 1 s batches."""

    def __init__(self, fps=30.0, pulse_amp=0.6, noise=0.4, seed=0):
        self.fps, self.amp, self.noise = fps, pulse_amp, noise
        self.rng = np.random.default_rng(seed)
        self.phase = 0.0

    def batch(self, bpm, t0_ms, seconds=1.0):
        n = int(seconds * self.fps)
        dt = 1.0 / self.fps
        t = t0_ms + np.arange(n) * dt * 1000 + self.rng.normal(0, 4.0, n)
        out = []
        for ti in t:
            self.phase += 2 * np.pi * (bpm / 60.0) * dt
            ts = ti / 1000.0
            drift = 3.0 * np.sin(2 * np.pi * 0.05 * ts)
            pulse = self.amp * np.sin(self.phase)
            rgb = np.array([160.0, 120.0, 100.0]) + drift + pulse * np.array([0.35, 1.0, 0.25]) \
                + self.rng.normal(0, self.noise, 3)
            out.append([float(ti), *map(float, rgb)])
        return out

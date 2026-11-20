"""Webcam heart rate (remote photoplethysmography, CHROM method).

The browser averages the RGB colour of the skin region (forehead and cheeks)
on every video frame and sends `[t, r, g, b]` samples here. Only these numbers
leave the browser - never the video itself.

Pipeline: resample to a uniform rate -> CHROM chrominance projection ->
band-pass 0.7-3 Hz -> FFT peak = heart rate. A signal-quality score (share of
spectral power near the peak) tells the rest of the system how much to trust
each estimate.
"""
from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

TARGET_FPS = 30.0
WINDOW_SECONDS = 10.0
MIN_SECONDS = 6.0
BPM_MIN = 42
BPM_MAX = 180
MIN_QUALITY = 0.35          # below this an estimate is reported but flagged unreliable
DEFAULT_RESTING_BPM = 70.0


def chrom_signal(rgb: np.ndarray) -> np.ndarray:
    """CHROM projection (de Haan & Jeanne, 2013) of an (N, 3) RGB trace."""
    rgb_norm = rgb / (rgb.mean(axis=0) + 1e-9)
    xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    alpha = xs.std() / (ys.std() + 1e-9)
    return xs - alpha * ys


def bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    nyq = fps / 2.0
    low = (BPM_MIN / 60.0) / nyq
    high = min(0.99, (BPM_MAX / 60.0) / nyq)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


def spectrum_bpm(signal: np.ndarray, fps: float):
    """Return (bpm, quality) from the dominant frequency of `signal`.

    quality = fraction of in-band power within +-6 BPM of the peak and its
    first harmonic. Clean pulse signals score ~0.5-0.9, noise ~0.1-0.3.
    """
    n = len(signal)
    n_fft = max(4096, 1 << (n - 1).bit_length())  # zero-pad for finer BPM resolution
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal * np.hanning(n), n=n_fft)) ** 2
    band = (freqs >= BPM_MIN / 60.0) & (freqs <= BPM_MAX / 60.0)
    if not band.any() or power[band].sum() == 0:
        return None, 0.0
    f_band, p_band = freqs[band], power[band]
    peak_f = f_band[np.argmax(p_band)]
    tol = 6.0 / 60.0
    near = (np.abs(f_band - peak_f) <= tol) | (np.abs(f_band - 2 * peak_f) <= tol)
    quality = float(p_band[near].sum() / p_band.sum())
    return float(peak_f * 60.0), quality


def bpm_to_va(bpm: float, resting_bpm: float):
    """Map heart rate relative to the user's own resting rate onto valence/arousal.

    Heart rate mainly tells us arousal. The valence nudge is deliberately small:
    a raised heart rate can be excitement or anxiety.
    """
    delta = (bpm - resting_bpm) / 25.0
    arousal = float(np.tanh(delta))
    valence = float(np.tanh(-delta * 0.3))
    return round(valence, 4), round(arousal, 4)


class RPPGDetector:
    def __init__(self, fps: float = TARGET_FPS):
        self.fps = fps
        self.samples = deque()  # (t, r, g, b), t in seconds

    def reset(self):
        self.samples.clear()

    def add_samples(self, samples):
        """samples: iterable of [t_ms, r, g, b] from the browser."""
        for s in samples:
            if len(s) != 4:
                continue
            t, r, g, b = (float(x) for x in s)
            t /= 1000.0
            if self.samples and t <= self.samples[-1][0]:
                continue  # drop out-of-order / duplicate frames
            self.samples.append((t, r, g, b))
        if self.samples:
            cutoff = self.samples[-1][0] - WINDOW_SECONDS
            while self.samples and self.samples[0][0] < cutoff:
                self.samples.popleft()

    def _uniform_trace(self) -> Optional[np.ndarray]:
        if len(self.samples) < 10:
            return None
        arr = np.array(self.samples, dtype=np.float64)
        t = arr[:, 0]
        if t[-1] - t[0] < MIN_SECONDS:
            return None
        grid = np.arange(t[0], t[-1], 1.0 / self.fps)
        return np.stack([np.interp(grid, t, arr[:, c]) for c in (1, 2, 3)], axis=1)

    def estimate(self, resting_bpm: float = DEFAULT_RESTING_BPM) -> Optional[dict]:
        rgb = self._uniform_trace()
        if rgb is None:
            return None
        pulse = bandpass(chrom_signal(rgb), self.fps)
        bpm, quality = spectrum_bpm(pulse, self.fps)
        if bpm is None:
            return None
        valence, arousal = bpm_to_va(bpm, resting_bpm)
        return {
            "bpm": round(bpm, 1),
            "quality": round(quality, 3),
            "reliable": quality >= MIN_QUALITY,
            "valence": valence,
            "arousal": arousal,
            "window_s": round(len(rgb) / self.fps, 1),
            "source": "rppg",
        }

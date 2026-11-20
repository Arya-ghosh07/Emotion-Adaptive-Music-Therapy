"""ISO principle as music parameters: meet the listener where they are, then
gently lead them somewhere calmer.

Match: tempo starts near the user's current heart rate and the mode reflects
their mood. Guide: tempo, loudness and note density ease down (smoothstep).
Target: settle at a slow, consonant, quiet texture.

The browser's music engine (Tone.js) turns these parameters into sound; the
chosen preset decides the instrument.
"""
from dataclasses import asdict, dataclass

import numpy as np

MATCH_SHARE = 0.2     # first 20% of the music phase
TARGET_SHARE = 0.25   # last 25%
MIN_TEMPO, MAX_TEMPO = 54.0, 110.0
FIXED_TEMPO = 70.0


@dataclass
class MusicParams:
    tempo_bpm: float
    scale: str
    velocity: float        # 0..1 loudness
    density: float         # 0..1 how many notes per beat
    phase: str             # match | guide | target | fixed
    preset: str

    def to_dict(self):
        return asdict(self)


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


def scale_for(valence: float, progress: float) -> str:
    """Start in a mode that matches the mood, resolve to bright pentatonic."""
    if progress >= 0.6:
        return "pentatonic_major"
    if valence < -0.3:
        return "minor" if progress < 0.3 else "dorian"
    if valence < 0.1:
        return "dorian"
    return "pentatonic_major"


def start_tempo(bpm, resting_bpm):
    ref = bpm if bpm else resting_bpm
    return float(np.clip(ref, MIN_TEMPO + 6, MAX_TEMPO))


def target_tempo(resting_bpm):
    return float(np.clip(resting_bpm - 8, MIN_TEMPO, 66))


def iso_params(progress: float, tempo_start: float, tempo_target: float,
               valence: float, preset: str) -> MusicParams:
    progress = float(np.clip(progress, 0, 1))
    if progress < MATCH_SHARE:
        phase, t = "match", 0.0
    elif progress < 1 - TARGET_SHARE:
        phase = "guide"
        t = (progress - MATCH_SHARE) / (1 - MATCH_SHARE - TARGET_SHARE)
    else:
        phase, t = "target", 1.0
    s = _smoothstep(t)
    return MusicParams(
        tempo_bpm=round(tempo_start + s * (tempo_target - tempo_start), 1),
        scale=scale_for(valence, progress),
        velocity=round(0.7 - 0.3 * s, 3),
        density=round(0.7 - 0.45 * s, 3),
        phase=phase,
        preset=preset,
    )


def fixed_params(preset: str) -> MusicParams:
    """Control condition: one style, constant tempo, no adaptation."""
    return MusicParams(FIXED_TEMPO, "pentatonic_major", 0.55, 0.45, "fixed", preset)

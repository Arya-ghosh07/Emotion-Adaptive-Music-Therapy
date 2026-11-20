"""SIMULATION (synthetic users, not real data).

Checks that the learning algorithm can find each person's best calming style
under realistic noise before trusting it with real sessions. Each synthetic
user has hidden preferences: how many BPM their heart rate drops per segment
for each style and state. Measurements get webcam-level noise (+-4 BPM), and
~20% of segments are lost to poor signal, as in real use.

Compares the contextual bandit with random choice and with always playing
the same style (the 'fixed' control).

    venv/bin/python scripts/simulate.py   ->  docs/results/simulation.png
"""
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from learning.bandit import CONTEXTS, ContextualUCB  # noqa: E402
from learning.presets import DEFAULT_FIXED_PRESET, PRESET_IDS  # noqa: E402
from learning.reward import BPM_FOR_FULL_REWARD  # noqa: E402

USERS = 200
SESSIONS = 20
SEGMENTS_PER_SESSION = 6
NOISE_BPM = 4.0
LOST_SEGMENT_RATE = 0.2
CONTEXT_WEIGHTS = [0.5, 0.4, 0.1]  # how often users start elevated / near rest / below rest


def make_user(rng):
    """True mean BPM drop per (context, preset). One or two styles really work."""
    prefs = {}
    for ctx in CONTEXTS:
        base = {p: rng.uniform(-0.5, 1.5) for p in PRESET_IDS}
        base[rng.choice(PRESET_IDS)] += rng.uniform(2.5, 4.5)   # this person's best style
        prefs[ctx] = base
    return prefs


def reward_from_drop(drop):
    return float(np.clip(0.5 + drop / (2 * BPM_FOR_FULL_REWARD), 0, 1))


def run(policy, rng, users):
    """Returns array [sessions] of mean true BPM drop per segment, averaged over users."""
    out = np.zeros(SESSIONS)
    for prefs in users:
        bandit = ContextualUCB(rng=random.Random(rng.random()))
        for s in range(SESSIONS):
            total = 0.0
            for _ in range(SEGMENTS_PER_SESSION):
                ctx = rng.choices(CONTEXTS, weights=CONTEXT_WEIGHTS)[0]
                if policy == "adaptive":
                    arm = bandit.select(ctx)
                elif policy == "random":
                    arm = rng.choice(PRESET_IDS)
                else:
                    arm = DEFAULT_FIXED_PRESET
                true_drop = prefs[ctx][arm]
                total += true_drop
                if rng.random() > LOST_SEGMENT_RATE:
                    measured = true_drop + rng.gauss(0, NOISE_BPM)
                    bandit.update(ctx, arm, reward_from_drop(measured))
            out[s] += total / SEGMENTS_PER_SESSION
    return out / len(users)


def main():
    rng = random.Random(42)
    users = [make_user(rng) for _ in range(USERS)]
    oracle = np.mean([sum(w * max(p[c].values()) for w, c in zip(CONTEXT_WEIGHTS, CONTEXTS)) for p in users])
    results = {name: run(name, random.Random(i), users) for i, name in enumerate(("adaptive", "random", "fixed"))}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(ROOT, "docs", "results"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "results", "simulation.png")
    x = np.arange(1, SESSIONS + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, results["adaptive"], label="Adaptive (contextual bandit)", lw=2.2, color="#3b6fd8")
    ax.plot(x, results["random"], label="Random style", lw=1.6, color="#999999")
    ax.plot(x, results["fixed"], label="Fixed style (control)", lw=1.6, color="#d8873b", ls="--")
    ax.axhline(oracle, color="#2a9d5b", lw=1, ls=":", label="Best possible (knows the user)")
    ax.set_xlabel("Session number")
    ax.set_ylabel("Mean heart-rate drop per segment (BPM)")
    ax.set_title(f"SIMULATION: {USERS} synthetic users, ±{NOISE_BPM:.0f} BPM measurement noise", fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)

    print("SIMULATION - synthetic users, not real data")
    print(f"  best possible           : {oracle:.2f} BPM/segment")
    for name, r in results.items():
        print(f"  {name:8s} session 1 : {r[0]:.2f}   session {SESSIONS}: {r[-1]:.2f}")
    print(f"  chart -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()

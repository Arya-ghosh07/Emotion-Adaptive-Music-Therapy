"""Analyse REAL sessions from data/eamta.db.

    venv/bin/python scripts/evaluate.py [--user NAME] [--db PATH]

Writes docs/results/real_<user>_*.png and prints a summary:
  * heart rate during the music phase, by mode (adaptive / fixed / silence),
    relative to where it started
  * heart-rate change per session, by mode
  * what the system has learned: mean reward per music style and state

Only sessions with reliable heart-rate data are used. With few sessions the
numbers are anecdotal - the script says so.
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from learning.bandit import CONTEXTS, ContextualUCB  # noqa: E402
from learning.presets import PRESETS  # noqa: E402

COLORS = {"adaptive": "#3b6fd8", "fixed": "#d8873b", "silence": "#888888"}
GRID_S = np.arange(0, 541, 15)  # 0-9 min of music, 15 s steps


def load(db_path, user):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    q = lambda sql, p=(): [dict(r) for r in con.execute(sql, p).fetchall()]  # noqa: E731
    sessions = q("SELECT * FROM sessions WHERE user_id = ? AND music_started_at IS NOT NULL "
                 "ORDER BY started_at", (user,))
    for s in sessions:
        s["heart"] = q("SELECT t, bpm FROM readings WHERE session_id = ? AND modality = 'heart' "
                       "AND reliable = 1 AND phase = 'music' ORDER BY t", (s["id"],))
        s["segments"] = q("SELECT * FROM segments WHERE session_id = ? ORDER BY idx", (s["id"],))
        s["reports"] = {r["moment"]: r for r in q("SELECT * FROM self_reports WHERE session_id = ?", (s["id"],))}
    return sessions


def relative_curve(s):
    """Heart rate on GRID_S relative to the first 30 s of music; None if too sparse."""
    if len(s["heart"]) < 30:
        return None
    t = np.array([h["t"] - s["music_started_at"] for h in s["heart"]])
    b = np.array([h["bpm"] for h in s["heart"]])
    start = b[t <= 30]
    if len(start) < 5:
        return None
    curve = np.full(len(GRID_S), np.nan)
    for i, g in enumerate(GRID_S):
        win = b[(t >= g - 15) & (t <= g + 15)]
        if len(win) >= 5:
            curve[i] = np.median(win) - np.median(start)
    return curve


def session_change(s):
    valid = [g for g in s["segments"] if g["valid"] and g["hr_start"] is not None]
    return valid[-1]["hr_end"] - valid[0]["hr_start"] if valid else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "eamta.db"))
    args = ap.parse_args()
    if not os.path.exists(args.db):
        sys.exit(f"No database at {args.db}. Run some sessions first.")

    sessions = load(args.db, args.user)
    if not sessions:
        sys.exit(f"No sessions with a music phase for user '{args.user}' yet.")

    by_mode = defaultdict(list)
    for s in sessions:
        by_mode[s["mode"]].append(s)

    print(f"REAL DATA - user '{args.user}', {len(sessions)} sessions")
    print(f"{'mode':9s} {'n':>3s} {'median HR change':>17s} {'self-report arousal change':>27s}")
    for mode in ("adaptive", "fixed", "silence"):
        ss = by_mode.get(mode, [])
        ch = [c for c in (session_change(s) for s in ss) if c is not None]
        ar = [s["reports"]["end"]["arousal"] - s["reports"]["start"]["arousal"]
              for s in ss if "start" in s["reports"] and "end" in s["reports"]]
        print(f"{mode:9s} {len(ss):3d} {(f'{np.median(ch):+.1f} BPM' if ch else '-'):>17s} "
              f"{(f'{np.median(ar):+.2f}' if ar else '-'):>27s}")
    if min((len(by_mode.get(m, [])) for m in ("adaptive", "fixed", "silence")), default=0) < 5:
        print("\nNote: fewer than 5 sessions in some modes - treat differences as anecdotal.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(ROOT, "docs", "results")
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for mode, ss in by_mode.items():
        curves = [c for c in (relative_curve(s) for s in ss) if c is not None]
        if not curves:
            continue
        m = np.nanmean(np.vstack(curves), axis=0)
        ax1.plot(GRID_S / 60, m, color=COLORS.get(mode), lw=2, label=f"{mode} (n={len(curves)})")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("Minutes into music phase")
    ax1.set_ylabel("Heart rate vs. start (BPM)")
    ax1.set_title("Heart rate during the session, by mode")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    modes = [m for m in ("adaptive", "fixed", "silence") if m in by_mode]
    data = [[c for c in (session_change(s) for s in by_mode[m]) if c is not None] for m in modes]
    for i, (m, d) in enumerate(zip(modes, data)):
        ax2.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)), d, color=COLORS[m], alpha=0.7)
        if d:
            ax2.hlines(np.median(d), i - 0.25, i + 0.25, color="k", lw=2)
    ax2.set_xticks(range(len(modes)), modes)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("HR change, first to last segment (BPM)")
    ax2.set_title("Per-session change (bar = median)")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle(f"REAL DATA - {args.user}", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"real_{args.user}_sessions.png")
    fig.savefig(p1, dpi=150)

    # what has been learned
    history = [(g["context"], g["preset"], g["reward"]) for s in by_mode.get("adaptive", [])
               for g in s["segments"] if g["valid"] and g["preset"]]
    if history:
        b = ContextualUCB.from_history(history)
        print("\nLearned calming score per style (0.5 = no effect, higher = HR dropped more):")
        print(f"{'':18s}" + "".join(f"{c:>13s}" for c in CONTEXTS))
        for p, meta in PRESETS.items():
            cells = []
            for c in CONTEXTS:
                n = b.n[(c, p)]
                cells.append(f"{b.mean(c, p):.2f} (n={n})" if n else "-")
            print(f"{meta['name']:18s}" + "".join(f"{x:>13s}" for x in cells))
    print(f"\nchart -> {os.path.relpath(p1, ROOT)}")


if __name__ == "__main__":
    main()

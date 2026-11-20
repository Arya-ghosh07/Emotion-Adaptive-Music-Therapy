"""Per-user contextual bandit: which music style calms *this* person, in *this* state?

Tabular contextual UCB1. The context is how activated the user is right now
(heart rate relative to their own resting rate). Each (context, preset) keeps a
running mean reward. Sparse contexts borrow strength from the preset's overall
mean (shrinkage), so the system is sensible after only a few sessions.

State is rebuilt from the logged segments, so the database is the single
source of truth and the model can always be audited or replayed.
"""
import math
import random
from collections import defaultdict
from typing import Iterable, Optional, Tuple

from .presets import PRESET_IDS

CONTEXTS = ("elevated", "near_rest", "below_rest")
EXPLORATION = 0.1     # tuned with scripts/simulate.py (0.1 beat 0.2-0.5 under webcam-level noise)
SHRINKAGE = 2.0       # pseudo-observations pulling a context's mean toward the global mean
PRIOR_MEAN = 0.5


def context_for(bpm: Optional[float], resting_bpm: float) -> str:
    if bpm is None:
        return "near_rest"
    if bpm > resting_bpm + 8:
        return "elevated"
    if bpm < resting_bpm - 5:
        return "below_rest"
    return "near_rest"


class ContextualUCB:
    def __init__(self, arms=PRESET_IDS, exploration=EXPLORATION, rng=None):
        self.arms = list(arms)
        self.c = exploration
        self.rng = rng or random.Random()
        self.n = defaultdict(int)       # (ctx, arm) -> pulls
        self.s = defaultdict(float)     # (ctx, arm) -> reward sum
        self.gn = defaultdict(int)      # arm -> pulls (all contexts)
        self.gs = defaultdict(float)

    @classmethod
    def from_history(cls, history: Iterable[Tuple[str, str, float]], **kw):
        b = cls(**kw)
        for ctx, arm, reward in history:
            b.update(ctx, arm, reward)
        return b

    def update(self, ctx: str, arm: str, reward: float):
        if arm not in self.arms:
            return
        r = min(1.0, max(0.0, float(reward)))
        self.n[(ctx, arm)] += 1
        self.s[(ctx, arm)] += r
        self.gn[arm] += 1
        self.gs[arm] += r

    def global_mean(self, arm: str) -> float:
        return self.gs[arm] / self.gn[arm] if self.gn[arm] else PRIOR_MEAN

    def mean(self, ctx: str, arm: str) -> float:
        g = self.global_mean(arm)
        n = self.n[(ctx, arm)]
        return (self.s[(ctx, arm)] + SHRINKAGE * g) / (n + SHRINKAGE)

    def scores(self, ctx: str) -> dict:
        total = sum(self.n[(ctx, a)] for a in self.arms)
        out = {}
        for a in self.arms:
            n = self.n[(ctx, a)]
            bonus = self.c * math.sqrt(2 * math.log(total + 1) / n) if n else float("inf")
            out[a] = {"mean": round(self.mean(ctx, a), 4), "pulls": n,
                      "ucb": None if math.isinf(bonus) else round(self.mean(ctx, a) + bonus, 4)}
        return out

    def select(self, ctx: str, exclude: Optional[str] = None) -> str:
        """Pick the next preset. `exclude` avoids repeating the same style back-to-back
        while still exploring (only applied when there is an alternative)."""
        candidates = [a for a in self.arms if a != exclude] or self.arms
        untried = [a for a in candidates if self.n[(ctx, a)] == 0]
        if untried:
            # try styles the user responded well to in other states first
            best_g = max(self.global_mean(a) for a in untried)
            top = [a for a in untried if math.isclose(self.global_mean(a), best_g)]
            return self.rng.choice(top)
        sc = self.scores(ctx)
        return max(candidates, key=lambda a: sc[a]["ucb"])

    def best(self, ctx: str) -> Optional[str]:
        tried = [a for a in self.arms if self.n[(ctx, a)] > 0]
        return max(tried, key=lambda a: self.mean(ctx, a)) if tried else None

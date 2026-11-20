import random

from fusion.fusion_engine import FusionEngine, ModalityReading, ReliabilityModel, sam_to_va
from learning.bandit import ContextualUCB, context_for
from learning.reward import segment_reward


def heart(start, end, bpm_start, bpm_end, ok=True, step=1.0):
    out, t = [], start
    while t <= end:
        frac = (t - start) / (end - start)
        out.append((t, bpm_start + frac * (bpm_end - bpm_start), ok, None))
        t += step
    return out


def test_reward_rewards_heart_rate_drop():
    r = segment_reward(heart(0, 90, 84, 76), 0, 90)
    assert r["valid"] and r["reward"] > 0.8 and r["hr_start"] > r["hr_end"]
    r = segment_reward(heart(0, 90, 76, 84), 0, 90)
    assert r["reward"] < 0.2
    assert 0.45 < segment_reward(heart(0, 90, 75, 75), 0, 90)["reward"] < 0.55


def test_reward_ignores_poor_signal_but_accepts_thumbs():
    bad = heart(0, 90, 84, 70, ok=False)
    assert segment_reward(bad, 0, 90)["valid"] is False
    r = segment_reward(bad, 0, 90, thumbs=1)
    assert r["valid"] and r["basis"] == "thumbs" and r["reward"] == 1.0


def test_reward_uses_hrv_when_present():
    s = [(t, 75.0, True, 30.0 if t < 45 else 45.0) for t in range(91)]
    r = segment_reward(s, 0, 90)
    assert r["reward"] > 0.7 and r["rmssd_end"] > r["rmssd_start"]


def test_context_buckets():
    assert context_for(85, 70) == "elevated"
    assert context_for(71, 70) == "near_rest"
    assert context_for(60, 70) == "below_rest"
    assert context_for(None, 70) == "near_rest"


def test_bandit_learns_best_style_per_context():
    rng = random.Random(7)
    truth = {"elevated": "ambient_pad", "near_rest": "warm_piano"}
    b = ContextualUCB(rng=random.Random(1))
    for i in range(240):
        ctx = "elevated" if i % 2 else "near_rest"
        arm = b.select(ctx)
        p = 0.8 if arm == truth[ctx] else 0.4
        b.update(ctx, arm, min(1, max(0, rng.gauss(p, 0.1))))
    assert b.best("elevated") == "ambient_pad"
    assert b.best("near_rest") == "warm_piano"
    pulls = {a: b.n[("elevated", a)] for a in b.arms}
    assert max(pulls, key=pulls.get) == "ambient_pad"


def test_bandit_avoids_repeat_and_rebuilds_from_history():
    b = ContextualUCB.from_history([("near_rest", "warm_piano", 0.9)] * 5)
    assert b.best("near_rest") == "warm_piano"
    assert b.select("near_rest", exclude="warm_piano") != "warm_piano"


def test_fusion_learns_to_distrust_a_misleading_sensor():
    rel = ReliabilityModel()
    before = rel.normalised_weights()["valence"]["face"]
    for _ in range(5):
        readings = {"face": ModalityReading("face", 0.9, 0.0), "text": ModalityReading("text", -0.5, 0.1)}
        rel.update(readings, {"valence": -0.5, "arousal": 0.0})  # user says: unpleasant
    after = rel.normalised_weights()["valence"]
    assert after["face"] < before and after["text"] > after["face"]


def test_fusion_never_uses_heart_for_valence():
    f = FusionEngine()
    f.add_reading(ModalityReading("heart", None, 0.8))
    out = f.fuse()
    assert out["arousal"] > 0.7 and out["valence_known"] is False
    assert "heart" in ReliabilityModel().normalised_weights()["arousal"]
    assert "heart" not in ReliabilityModel().normalised_weights()["valence"]


def test_sam_scale():
    assert sam_to_va(1) == -1 and sam_to_va(5) == 0 and sam_to_va(9) == 1

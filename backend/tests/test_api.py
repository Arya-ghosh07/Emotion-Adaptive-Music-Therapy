"""End-to-end session through the HTTP API with a fake clock and fake face model."""
import pytest

from app import create_app
from conftest import PulseStream
from inputs.text_nlp import TextEmotionDetector
from session.manager import BASELINE_SECONDS, SEGMENT_SECONDS


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class FakeFace:
    available = True

    def analyze_image(self, image):
        return {"face_found": True, "valence": 0.2, "arousal": 0.1, "confidence": 0.9,
                "dominant_emotion": "neutral", "box": [0.3, 0.2, 0.4, 0.5], "source": "facial"}


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    app = create_app(db_path=str(tmp_path / "t.db"), facial=FakeFace(),
                     text=TextEmotionDetector(use_transformer=False), clock=clock)
    return app.test_client(), clock, app.extensions["eamta"]["db"]


def feed_heart(client, sid, clock, seconds, bpm_from, bpm_to, stream=None):
    """Send 1 s batches of synthetic webcam samples, heart rate drifting linearly."""
    stream = stream or PulseStream(seed=int(clock.t))
    for i in range(int(seconds)):
        bpm = bpm_from + (bpm_to - bpm_from) * i / max(1, seconds - 1)
        batch = stream.batch(bpm, t0_ms=clock.t * 1000)
        clock.t += 1.0
        assert client.post(f"/api/session/{sid}/rppg", json={"samples": batch}).status_code == 200


def test_rejects_bad_input(env):
    client, _, _ = env
    assert client.post("/api/session/start", json={"user_id": "../../etc"}).status_code == 400
    assert client.post("/api/session/start", json={"user_id": "a", "mode": "x"}).status_code == 400
    assert client.get("/api/session/nope/status").status_code == 404
    sid = client.post("/api/session/start", json={"user_id": "a"}).json["id"]
    assert client.post(f"/api/session/{sid}/self_report", json={"valence": 12, "arousal": 3}).status_code == 400
    assert client.post(f"/api/session/{sid}/rppg", json={"samples": "x"}).status_code == 400


def test_full_adaptive_session_learns_and_logs(env):
    client, clock, db = env
    s = client.post("/api/session/start", json={"user_id": "tester", "mode": "adaptive", "music_minutes": 3}).json
    sid = s["id"]
    assert s["phase"] == "checkin"

    r = client.post(f"/api/session/{sid}/text", json={"text": "I feel stressed and anxious"}).json
    assert r["valence"] < 0
    client.post(f"/api/session/{sid}/face", json={"image": "data:image/jpeg;base64,xx"})
    client.post(f"/api/session/{sid}/self_report", json={"valence": 3, "arousal": 7, "moment": "start"})
    assert client.post(f"/api/session/{sid}/begin").json["phase"] == "baseline"

    feed_heart(client, sid, clock, BASELINE_SECONDS + 1, 84, 84)
    st = client.get(f"/api/session/{sid}/status").json
    assert st["phase"] == "music"
    assert st["heart"]["reliable"] and abs(st["heart"]["bpm"] - 84) <= 4
    assert st["music"]["preset"] and st["music"]["phase"] == "match"
    assert 70 <= st["music"]["tempo_bpm"] <= 95  # ISO: tempo starts near heart rate

    client.post(f"/api/session/{sid}/feedback", json={"thumbs": 1})
    feed_heart(client, sid, clock, 2 * SEGMENT_SECONDS + 2, 84, 70)  # user calms down
    st = client.get(f"/api/session/{sid}/status").json
    assert st["phase"] == "report"
    assert st["last_segment"]["valid"]

    client.post(f"/api/session/{sid}/self_report", json={"valence": 6, "arousal": 3, "moment": "end"})
    summary = client.post(f"/api/session/{sid}/end").json
    assert -18 < summary["hr_change"] < -8  # true change over the music phase is -14
    segs = db.query("SELECT * FROM segments WHERE session_id = ?", (sid,))
    assert len(segs) == 2 and all(g["valid"] for g in segs) and segs[0]["reward"] > 0.5
    assert db.one("SELECT resting_bpm FROM users WHERE id = 'tester'")["resting_bpm"] == pytest.approx(84, abs=4)

    ins = client.get("/api/users/tester/insights").json
    assert ins["learned_segments"] == 2
    assert ins["sessions"][0]["hr_change"] < 0
    assert client.get(f"/api/session/{sid}/status").status_code == 404


def test_fixed_and_silence_modes_do_not_train_bandit(env):
    client, clock, db = env
    for mode in ("fixed", "silence"):
        sid = client.post("/api/session/start", json={"user_id": "ctl", "mode": mode, "music_minutes": 3}).json["id"]
        client.post(f"/api/session/{sid}/begin")
        feed_heart(client, sid, clock, BASELINE_SECONDS + SEGMENT_SECONDS + 2, 80, 75)
        st = client.get(f"/api/session/{sid}/status").json
        if mode == "fixed":
            assert st["music"]["phase"] == "fixed" and st["music"]["preset"] == "warm_piano"
        else:
            assert st["music"] is None and st["segment"]["name"] == "Silence"
        client.post(f"/api/session/{sid}/end")
    assert db.bandit_history("ctl") == []
    assert len(db.query("SELECT * FROM segments WHERE user_id = 'ctl' AND valid = 1")) >= 2

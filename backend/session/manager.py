"""One listening session as a state machine.

checkin  -> user says how they feel (spoken check-in) and sets two sliders
baseline -> 60 s of silence: sensors warm up, resting heart rate is measured
music    -> ~90 s segments; at every boundary the reward for the segment just
            played is computed and (adaptive mode) the bandit picks the next style
report   -> user sets the sliders again
done

Modes make a fair self-experiment possible:
  adaptive - bandit picks styles, ISO tempo curve starts at your heart rate
  fixed    - one style, constant tempo (control)
  silence  - no music (baseline)
"""
import threading
import time
import uuid
from statistics import median
from typing import Callable, Dict, Optional

from fusion.fusion_engine import FusionEngine, ModalityReading, ReliabilityModel, sam_to_va
from inputs.heart_ble import BLEHeartRate
from inputs.rppg import DEFAULT_RESTING_BPM, RPPGDetector, bpm_to_va
from learning.bandit import ContextualUCB, context_for
from learning.presets import DEFAULT_FIXED_PRESET, PRESETS
from learning.reward import segment_reward
from music.therapy_engine import fixed_params, iso_params, start_tempo, target_tempo

MODES = ("adaptive", "fixed", "silence")
BASELINE_SECONDS = 60.0
SEGMENT_SECONDS = 90.0
DEFAULT_MUSIC_MINUTES = 9.0
HEART_SERIES_SECONDS = 180.0


class Session:
    def __init__(self, db, user_id: str, mode: str, music_minutes: float,
                 clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock
        self.id = uuid.uuid4().hex[:12]
        self.user_id = user_id
        self.mode = mode
        self.music_seconds = max(3.0, min(30.0, music_minutes)) * 60.0
        self.lock = threading.RLock()

        user = db.ensure_user(user_id)
        self.resting_bpm = user.get("resting_bpm") or DEFAULT_RESTING_BPM
        self.fusion = FusionEngine(ReliabilityModel(db.modality_errors(user_id)))
        self.bandit = ContextualUCB.from_history(db.bandit_history(user_id))

        self.rppg = RPPGDetector()
        self.ble = BLEHeartRate()
        self.heart_source = "webcam"
        self.heart = []            # (t, bpm, reliable, rmssd)
        self.latest_heart = None
        self.latest_face = None
        self.latest_text = None

        self.phase = "checkin"
        self.started_at = clock()
        self.baseline_ends = None
        self.music_started = None
        self.music_ends = None
        self.tempo_start = self.tempo_target = None
        self.segment = None        # dict for the segment in progress
        self.segments = []         # finished segment results
        self.start_valence = 0.0

        db.execute("INSERT INTO sessions (id, user_id, mode, started_at) VALUES (?, ?, ?, ?)",
                   (self.id, user_id, mode, self.started_at))

    # ----------------------------------------------------------------- inputs
    def _log_reading(self, modality, valence=None, arousal=None, bpm=None,
                     quality=None, reliable=None, rmssd=None):
        self.db.execute(
            "INSERT INTO readings (session_id, t, modality, valence, arousal, bpm, quality, reliable, rmssd, phase) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.id, self.clock(), modality, valence, arousal, bpm, quality,
             None if reliable is None else int(reliable), rmssd, self.phase))

    def _record_heart(self, est: dict):
        now = self.clock()
        self.latest_heart = {**est, "t": now}
        self.heart.append((now, est["bpm"], bool(est["reliable"]), est.get("rmssd")))
        self._log_reading("heart", est.get("valence"), est.get("arousal"), est["bpm"],
                          est.get("quality"), est["reliable"], est.get("rmssd"))
        if est["reliable"]:
            self.fusion.add_reading(ModalityReading(
                "heart", None, est["arousal"], confidence=min(1.0, est.get("quality", 1.0) + 0.3),
                timestamp=now))

    def add_rppg_samples(self, samples):
        with self.lock:
            if self.heart_source == "ble":
                return self.latest_heart
            self.rppg.add_samples(samples)
            est = self.rppg.estimate(self.resting_bpm)
            if est:
                self._record_heart(est)
            self.tick()
            return est

    def add_ble(self, bpm, rr_ms):
        with self.lock:
            self.heart_source = "ble"
            self.ble.add(self.clock(), bpm, rr_ms)
            est = self.ble.latest()
            if est:
                est["valence"], est["arousal"] = bpm_to_va(est["bpm"], self.resting_bpm)
                self._record_heart(est)
            self.tick()
            return est

    def add_face(self, result: dict):
        with self.lock:
            if result.get("face_found"):
                self.latest_face = {**result, "t": self.clock()}
                self.fusion.add_reading(ModalityReading(
                    "face", result["valence"], result["arousal"],
                    confidence=result.get("confidence", 0.8), timestamp=self.clock()))
                self._log_reading("face", result["valence"], result["arousal"],
                                  quality=result.get("confidence"))
            return result

    def add_text(self, text: str, result: dict):
        with self.lock:
            self.latest_text = {**result, "t": self.clock()}
            self.fusion.add_reading(ModalityReading(
                "text", result["valence"], result["arousal"], 1.0, timestamp=self.clock()))
            self._log_reading("text", result["valence"], result["arousal"])
            if self.phase == "checkin":
                self.db.execute("UPDATE sessions SET checkin_text = ?, checkin_emotion = ? WHERE id = ?",
                                (text[:500], result.get("dominant_emotion"), self.id))
            return result

    def add_self_report(self, valence_raw: int, arousal_raw: int, moment: str):
        """Sliders are 1..9. Also teaches the fusion engine how reliable each sensor is."""
        with self.lock:
            reported = {"valence": sam_to_va(valence_raw), "arousal": sam_to_va(arousal_raw)}
            self.db.execute(
                "INSERT INTO self_reports (session_id, user_id, t, moment, valence_raw, arousal_raw, valence, arousal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.id, self.user_id, self.clock(), moment, int(valence_raw), int(arousal_raw),
                 reported["valence"], reported["arousal"]))
            changed = self.fusion.reliability.update(self.fusion.fresh_readings(self.clock()), reported)
            for m, d, err, n in changed:
                self.db.save_modality_error(self.user_id, m, d, err, n)
            if moment == "start":
                self.start_valence = reported["valence"]
            return {"reported": reported, "updated": [f"{m}.{d}" for m, d, _, _ in changed],
                    "weights": self.fusion.reliability.normalised_weights()}

    def set_thumbs(self, thumbs: int):
        with self.lock:
            if self.segment:
                self.segment["thumbs"] = 1 if thumbs > 0 else -1
                return True
            return False

    # ------------------------------------------------------------ transitions
    def begin(self):
        """Check-in done: start the silent baseline."""
        with self.lock:
            if self.phase == "checkin":
                self.phase = "baseline"
                self.baseline_ends = self.clock() + BASELINE_SECONDS
            return self.phase

    def current_bpm(self, seconds: float = 10.0) -> Optional[float]:
        now = self.clock()
        recent = [b for t, b, ok, _ in self.heart if ok and now - t <= seconds]
        return median(recent) if recent else None

    def _start_music(self, now):
        baseline = [b for t, b, ok, _ in self.heart if ok and t >= now - BASELINE_SECONDS]
        baseline_bpm = round(median(baseline), 1) if len(baseline) >= 5 else None
        self.db.execute("UPDATE sessions SET baseline_bpm = ?, music_started_at = ?, heart_source = ? WHERE id = ?",
                        (baseline_bpm, now, self.heart_source, self.id))
        bpm = self.current_bpm() or baseline_bpm
        self.tempo_start = start_tempo(bpm, self.resting_bpm)
        self.tempo_target = target_tempo(self.resting_bpm)
        self.phase = "music"
        self.music_started = now
        self.music_ends = now + self.music_seconds
        self._start_segment(now)

    def _pick_preset(self, context):
        if self.mode == "silence":
            return None
        if self.mode == "fixed":
            return DEFAULT_FIXED_PRESET
        return self.bandit.select(context)

    def _start_segment(self, now):
        context = context_for(self.current_bpm(), self.resting_bpm)
        self.segment = {"idx": len(self.segments), "preset": self._pick_preset(context),
                        "context": context, "started_at": now, "thumbs": None}

    def _finish_segment(self, now, complete=True):
        seg = self.segment
        self.segment = None
        if seg is None:
            return None
        res = segment_reward(self.heart, seg["started_at"], now, seg["thumbs"])
        if not complete:  # cut short by ending the session: log it, don't learn from it
            res = {**res, "valid": False, "basis": "incomplete"}
        if res["valid"] and self.mode == "adaptive" and seg["preset"]:
            self.bandit.update(seg["context"], seg["preset"], res["reward"])
        row = {**seg, **res, "ended_at": now}
        self.db.execute(
            "INSERT INTO segments (session_id, user_id, mode, idx, preset, context, started_at, ended_at, "
            "hr_start, hr_end, rmssd_start, rmssd_end, thumbs, reward, valid, basis) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self.id, self.user_id, self.mode, seg["idx"], seg["preset"], seg["context"],
             seg["started_at"], now, res["hr_start"], res["hr_end"], res["rmssd_start"],
             res["rmssd_end"], seg["thumbs"], res["reward"], int(res["valid"]), res["basis"]))
        self.segments.append(row)
        return row

    def tick(self):
        """Advance phases based on the clock. Called on every input and status poll."""
        with self.lock:
            now = self.clock()
            if self.phase == "baseline" and now >= self.baseline_ends:
                self._start_music(now)
            if self.phase == "music":
                while self.segment and now >= self.segment["started_at"] + SEGMENT_SECONDS:
                    boundary = self.segment["started_at"] + SEGMENT_SECONDS
                    self._finish_segment(boundary)
                    if boundary + SEGMENT_SECONDS * 0.5 <= self.music_ends:
                        self._start_segment(boundary)
                if self.segment is None or now >= self.music_ends:
                    if self.segment:
                        self._finish_segment(now, complete=now - self.segment["started_at"] >= SEGMENT_SECONDS * 0.8)
                    self.phase = "report"

            # keep memory bounded; full history is in the DB
            cutoff = now - max(HEART_SERIES_SECONDS, SEGMENT_SECONDS + 10)
            if self.heart and self.heart[0][0] < cutoff:
                self.heart = [h for h in self.heart if h[0] >= cutoff]

    def end(self):
        with self.lock:
            now = self.clock()
            if self.phase == "music" and self.segment:
                self._finish_segment(now, complete=now - self.segment["started_at"] >= SEGMENT_SECONDS * 0.8)
            completed = self.phase in ("report", "music")
            self.phase = "done"
            self.db.execute("UPDATE sessions SET ended_at = ?, completed = ? WHERE id = ?",
                            (now, int(completed), self.id))
            # resting heart rate = median of recent baselines
            baselines = self.db.recent_baselines(self.user_id)
            if baselines:
                self.resting_bpm = round(median(baselines), 1)
                self.db.set_resting_bpm(self.user_id, self.resting_bpm)
            return self.summary()

    # ---------------------------------------------------------------- outputs
    def music_params(self):
        if self.phase != "music" or not self.segment or not self.segment["preset"]:
            return None
        if self.mode == "fixed":
            return fixed_params(self.segment["preset"]).to_dict()
        progress = (self.clock() - self.music_started) / self.music_seconds
        valence = self.fusion.fuse(self.clock()) or {}
        v = valence.get("valence", self.start_valence) if valence.get("valence_known") else self.start_valence
        return iso_params(progress, self.tempo_start, self.tempo_target, v, self.segment["preset"]).to_dict()

    def status(self):
        with self.lock:
            self.tick()
            now = self.clock()
            phase_ends = {"baseline": self.baseline_ends, "music": self.music_ends}.get(self.phase)
            seg = None
            if self.segment:
                p = self.segment["preset"]
                seg = {"idx": self.segment["idx"], "preset": p,
                       "name": PRESETS[p]["name"] if p else "Silence",
                       "context": self.segment["context"], "thumbs": self.segment["thumbs"],
                       "ends_in": round(self.segment["started_at"] + SEGMENT_SECONDS - now, 1)}
            ctx = self.segment["context"] if self.segment else context_for(self.current_bpm(), self.resting_bpm)
            return {
                "id": self.id, "user_id": self.user_id, "mode": self.mode, "phase": self.phase,
                "elapsed_s": round(now - self.started_at, 1),
                "phase_remaining_s": None if phase_ends is None else round(max(0.0, phase_ends - now), 1),
                "resting_bpm": self.resting_bpm,
                "heart": self.latest_heart,
                "heart_source": self.heart_source,
                "heart_series": [[round(t - self.started_at, 1), b, ok] for t, b, ok, _ in self.heart
                                 if now - t <= HEART_SERIES_SECONDS],
                "face": self.latest_face,
                "text": self.latest_text,
                "fused": self.fusion.fuse(now),
                "weights": self.fusion.reliability.normalised_weights(),
                "music": self.music_params(),
                "segment": seg,
                "last_segment": self._public_segment(self.segments[-1]) if self.segments else None,
                "learning": {"context": ctx, "scores": self.bandit.scores(ctx)} if self.mode == "adaptive" else None,
            }

    @staticmethod
    def _public_segment(s):
        keys = ("idx", "preset", "context", "hr_start", "hr_end", "rmssd_start", "rmssd_end",
                "thumbs", "reward", "valid", "basis")
        return {k: s.get(k) for k in keys}

    def summary(self):
        valid = [s for s in self.segments if s["valid"] and s["hr_start"] is not None]
        return {
            "id": self.id, "mode": self.mode,
            "segments": [self._public_segment(s) for s in self.segments],
            "hr_change": round(valid[-1]["hr_end"] - valid[0]["hr_start"], 1) if valid else None,
            "resting_bpm": self.resting_bpm,
        }


class SessionManager:
    def __init__(self, db, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def start(self, user_id, mode, music_minutes=DEFAULT_MUSIC_MINUTES) -> Session:
        s = Session(self.db, user_id, mode, music_minutes, clock=self.clock)
        with self._lock:
            # one live session per user; abandon any older one
            for sid, old in list(self.sessions.items()):
                if old.user_id == user_id:
                    old.end()
                    del self.sessions[sid]
            self.sessions[s.id] = s
        return s

    def get(self, session_id) -> Optional[Session]:
        return self.sessions.get(session_id)

    def finish(self, session_id):
        with self._lock:
            s = self.sessions.pop(session_id, None)
        return s.end() if s else None

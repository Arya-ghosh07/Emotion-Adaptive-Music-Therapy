"""EAMTA API server.

The browser does the capturing (webcam, mic, optional Bluetooth strap) and the
music playback; this server estimates heart rate and emotion, runs the session
state machine, learns per user, and logs everything to SQLite.
"""
import os
import re
import sys
from statistics import median

from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inputs.facial import FacialEmotionDetector  # noqa: E402
from inputs.text_nlp import TextEmotionDetector  # noqa: E402
from learning.bandit import CONTEXTS, ContextualUCB  # noqa: E402
from learning.presets import PRESETS  # noqa: E402
from session.manager import MODES, SessionManager  # noqa: E402
from storage.db import Database  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(ROOT, "data", "eamta.db")
USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_RPPG_SAMPLES = 300
MAX_IMAGE_BYTES = 2_000_000


def create_app(db_path=None, facial=None, text=None, clock=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    CORS(app, origins=os.environ.get("EAMTA_CORS_ORIGINS", "http://localhost:3000").split(","))

    db = Database(db_path or os.environ.get("EAMTA_DB", DEFAULT_DB))
    manager = SessionManager(db, **({"clock": clock} if clock else {}))
    facial = facial or FacialEmotionDetector()
    text = text or TextEmotionDetector(use_transformer=os.environ.get("EAMTA_TEXT_MODEL", "1") != "0")
    app.extensions["eamta"] = {"db": db, "manager": manager}

    def bad(msg, code=400):
        return jsonify({"error": msg}), code

    def body():
        return request.get_json(silent=True) or {}

    def session_or_404(sid):
        s = manager.get(sid)
        if s is None:
            return None, bad("unknown or finished session", 404)
        return s, None

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "face_model": facial.available,
                        "presets": {k: v["name"] for k, v in PRESETS.items()}, "modes": MODES})

    @app.post("/api/session/start")
    def start():
        data = body()
        user_id = str(data.get("user_id", "")).strip()
        if not USER_ID_RE.match(user_id):
            return bad("user_id must be 1-32 letters, digits, '-' or '_'")
        mode = data.get("mode", "adaptive")
        if mode not in MODES:
            return bad(f"mode must be one of {MODES}")
        try:
            minutes = float(data.get("music_minutes", 9))
        except (TypeError, ValueError):
            return bad("music_minutes must be a number")
        s = manager.start(user_id, mode, minutes)
        return jsonify(s.status())

    @app.post("/api/session/<sid>/begin")
    def begin(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        s.begin()
        return jsonify(s.status())

    @app.get("/api/session/<sid>/status")
    def status(sid):
        s, err = session_or_404(sid)
        return err or jsonify(s.status())

    @app.post("/api/session/<sid>/rppg")
    def rppg(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        samples = body().get("samples")
        if not isinstance(samples, list) or len(samples) > MAX_RPPG_SAMPLES:
            return bad(f"samples must be a list of at most {MAX_RPPG_SAMPLES} [t_ms, r, g, b]")
        try:
            est = s.add_rppg_samples(samples)
        except (TypeError, ValueError):
            return bad("samples must be numeric [t_ms, r, g, b]")
        return jsonify({"heart": est})

    @app.post("/api/session/<sid>/hr")
    def hr(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        data = body()
        try:
            bpm = float(data["bpm"]) if data.get("bpm") is not None else None
            rr = [float(x) for x in (data.get("rr_ms") or [])][:20]
        except (TypeError, ValueError):
            return bad("bpm and rr_ms must be numbers")
        return jsonify({"heart": s.add_ble(bpm, rr)})

    @app.post("/api/session/<sid>/face")
    def face(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        image = body().get("image")
        if not isinstance(image, str) or len(image) > MAX_IMAGE_BYTES:
            return bad("image must be a JPEG data URL under 2 MB")
        return jsonify(s.add_face(facial.analyze_image(image)))

    @app.post("/api/session/<sid>/text")
    def analyze_text(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        t = str(body().get("text", "")).strip()
        if not t:
            return bad("text is required")
        return jsonify(s.add_text(t, text.analyze_text(t)))

    @app.post("/api/session/<sid>/self_report")
    def self_report(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        data = body()
        try:
            v, a = int(data["valence"]), int(data["arousal"])
        except (KeyError, TypeError, ValueError):
            return bad("valence and arousal (1-9) are required")
        if not (1 <= v <= 9 and 1 <= a <= 9):
            return bad("valence and arousal must be between 1 and 9")
        moment = data.get("moment", "mid")
        if moment not in ("start", "mid", "end"):
            return bad("moment must be start, mid or end")
        return jsonify(s.add_self_report(v, a, moment))

    @app.post("/api/session/<sid>/feedback")
    def feedback(sid):
        s, err = session_or_404(sid)
        if err:
            return err
        thumbs = body().get("thumbs")
        if thumbs not in (1, -1):
            return bad("thumbs must be 1 or -1")
        return jsonify({"recorded": s.set_thumbs(thumbs)})

    @app.post("/api/session/<sid>/end")
    def end(sid):
        summary = manager.finish(sid)
        return jsonify(summary) if summary else bad("unknown or finished session", 404)

    @app.get("/api/users/<user_id>/insights")
    def insights(user_id):
        if not USER_ID_RE.match(user_id):
            return bad("invalid user_id")
        user = db.one("SELECT * FROM users WHERE id = ?", (user_id,))
        if not user:
            return jsonify({"user_id": user_id, "sessions": [], "learning": None})
        bandit = ContextualUCB.from_history(db.bandit_history(user_id))
        learning = {ctx: {"best": bandit.best(ctx), "scores": bandit.scores(ctx)} for ctx in CONTEXTS}
        sessions = db.query(
            "SELECT s.id, s.mode, s.started_at, s.baseline_bpm, s.heart_source, s.checkin_emotion, s.completed, "
            "  (SELECT hr_start FROM segments g WHERE g.session_id = s.id AND g.valid = 1 AND g.hr_start IS NOT NULL ORDER BY idx LIMIT 1) AS hr_first, "
            "  (SELECT hr_end FROM segments g WHERE g.session_id = s.id AND g.valid = 1 AND g.hr_end IS NOT NULL ORDER BY idx DESC LIMIT 1) AS hr_last, "
            "  (SELECT valence FROM self_reports r WHERE r.session_id = s.id AND moment = 'start' ORDER BY t LIMIT 1) AS v_start, "
            "  (SELECT valence FROM self_reports r WHERE r.session_id = s.id AND moment = 'end' ORDER BY t DESC LIMIT 1) AS v_end, "
            "  (SELECT arousal FROM self_reports r WHERE r.session_id = s.id AND moment = 'start' ORDER BY t LIMIT 1) AS a_start, "
            "  (SELECT arousal FROM self_reports r WHERE r.session_id = s.id AND moment = 'end' ORDER BY t DESC LIMIT 1) AS a_end "
            "FROM sessions s WHERE s.user_id = ? ORDER BY s.started_at DESC LIMIT 30", (user_id,))
        by_mode = {}
        for s in sessions:
            if s["hr_first"] is not None and s["hr_last"] is not None:
                s["hr_change"] = round(s["hr_last"] - s["hr_first"], 1)
                by_mode.setdefault(s["mode"], []).append(s["hr_change"])
        return jsonify({
            "user_id": user_id,
            "resting_bpm": user["resting_bpm"],
            "presets": {k: v["name"] for k, v in PRESETS.items()},
            "learning": learning,
            "learned_segments": len(db.bandit_history(user_id)),
            "hr_change_by_mode": {m: {"median": round(median(v), 1), "n": len(v)} for m, v in by_mode.items()},
            "sessions": sessions,
        })

    return app


if __name__ == "__main__":
    port = int(os.environ.get("EAMTA_PORT", "5001"))
    debug = os.environ.get("EAMTA_DEBUG") == "1"
    print(f"[EAMTA] http://127.0.0.1:{port}/api/health  (debug={debug})")
    create_app().run(host="127.0.0.1", port=port, debug=debug, threaded=True)

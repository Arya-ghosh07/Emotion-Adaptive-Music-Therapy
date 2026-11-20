"""Facial expression -> valence/arousal, from frames the browser sends.

Frames are decoded and analysed in memory only; nothing is written to disk.
Also returns the face box so the browser can place the heart-rate ROI on the
forehead and cheeks.
"""
import base64
from typing import Optional

import cv2
import numpy as np

try:
    try:
        from fer.fer import FER  # fer >= 25 moved the class here
    except ImportError:
        from fer import FER
    FER_AVAILABLE = True
except Exception:  # fer pulls in TensorFlow; keep the server usable without it
    FER_AVAILABLE = False

# Approximate circumplex positions of the basic emotions (Russell, 1980).
EMOTION_VA_MAP = {
    "angry":    (-0.70,  0.85),
    "disgust":  (-0.60,  0.40),
    "fear":     (-0.65,  0.80),
    "happy":    ( 0.85,  0.60),
    "sad":      (-0.70, -0.50),
    "surprise": ( 0.10,  0.75),
    "neutral":  ( 0.00,  0.00),
}


def decode_image(data_url: str) -> Optional[np.ndarray]:
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(data_url), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def scores_to_va(scores: dict):
    v = a = total = 0.0
    for emotion, score in scores.items():
        if emotion in EMOTION_VA_MAP:
            ev, ea = EMOTION_VA_MAP[emotion]
            v += score * ev
            a += score * ea
            total += score
    if total > 0:
        v, a = v / total, a / total
    return round(float(v), 4), round(float(a), 4)


class FacialEmotionDetector:
    def __init__(self):
        self.available = FER_AVAILABLE
        self._detector = None

    def _get_detector(self):
        if self._detector is None and self.available:
            self._detector = FER(mtcnn=False)
        return self._detector

    def analyze_image(self, data_url: str) -> dict:
        frame = decode_image(data_url)
        if frame is None:
            return {"face_found": False, "error": "invalid_image", "source": "facial"}
        detector = self._get_detector()
        if detector is None:
            return {"face_found": False, "error": "fer_not_installed", "source": "facial"}
        faces = detector.detect_emotions(frame)
        if not faces:
            return {"face_found": False, "source": "facial"}
        face = max(faces, key=lambda f: f["box"][2] * f["box"][3])  # largest face
        emotions = face["emotions"]
        valence, arousal = scores_to_va(emotions)
        h, w = frame.shape[:2]
        x, y, bw, bh = (int(v) for v in face["box"])
        return {
            "face_found": True,
            "valence": valence,
            "arousal": arousal,
            "dominant_emotion": max(emotions, key=emotions.get),
            "confidence": round(float(max(emotions.values())), 3),
            # normalised so the browser can map it onto any video size
            "box": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)],
            "source": "facial",
        }

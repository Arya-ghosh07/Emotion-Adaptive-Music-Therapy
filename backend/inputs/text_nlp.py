"""Text (spoken check-in transcript) -> valence/arousal.

Uses a local DistilRoBERTa emotion classifier when available (downloaded once,
~330 MB, runs on CPU). Falls back to a small keyword lexicon if the model
cannot be loaded; the response always says which method produced it.
"""
import re
import threading

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_VA_MAP = {
    "anger":    (-0.70,  0.85),
    "disgust":  (-0.60,  0.40),
    "fear":     (-0.65,  0.80),
    "joy":      ( 0.85,  0.60),
    "neutral":  ( 0.00,  0.00),
    "sadness":  (-0.70, -0.50),
    "surprise": ( 0.10,  0.75),
}

KEYWORD_LEXICON = {
    "anger":    ["angry", "furious", "rage", "mad", "irritated", "hate", "annoyed"],
    "disgust":  ["disgusting", "gross", "revolting", "nasty", "repulsed"],
    "fear":     ["scared", "afraid", "terrified", "anxious", "nervous", "worried", "stressed", "panic"],
    "joy":      ["happy", "joyful", "excited", "great", "wonderful", "love", "amazing", "good"],
    "sadness":  ["sad", "depressed", "unhappy", "cry", "miserable", "hopeless", "lonely", "tired"],
    "surprise": ["surprised", "shocked", "unexpected", "wow", "astonished"],
    "neutral":  ["okay", "fine", "alright", "normal", "nothing"],
}


def keyword_scores(text: str) -> dict:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    scores = {e: sum(1 for kw in kws if kw in tokens) for e, kws in KEYWORD_LEXICON.items()}
    total = sum(scores.values())
    if total == 0:
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTION_VA_MAP}
    return {e: s / total for e, s in scores.items()}


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


class TextEmotionDetector:
    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()

    def _get_model(self):
        if not self.use_transformer or self._load_failed:
            return None
        with self._lock:
            if self._model is None:
                try:
                    self._model = hf_pipeline("text-classification", model=MODEL_NAME, top_k=None, device=-1)
                except Exception as e:
                    print(f"[text] transformer unavailable ({e}); using keyword fallback")
                    self._load_failed = True
        return self._model

    def analyze_text(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"valence": 0.0, "arousal": 0.0, "dominant_emotion": "neutral",
                    "method": "empty", "source": "text"}
        model = self._get_model()
        if model is not None:
            out = model(text[:512])
            out = out[0] if out and isinstance(out[0], list) else out
            scores = {r["label"].lower(): float(r["score"]) for r in out}
            method = "distilroberta"
        else:
            scores = keyword_scores(text)
            method = "keywords"
        valence, arousal = scores_to_va(scores)
        return {
            "valence": valence,
            "arousal": arousal,
            "dominant_emotion": max(scores, key=scores.get),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "method": method,
            "source": "text",
        }

"""Music styles the system can choose between (the bandit's arms).

Each preset fixes the *timbre* and texture; the music engine controls tempo,
scale and loudness on top of it. The ids must match frontend/src/audio/musicEngine.js.
"""
PRESETS = {
    "warm_piano":       {"name": "Warm piano",       "instrument": "piano"},
    "ambient_pad":      {"name": "Ambient pad",      "instrument": "ambient_pad"},
    "soft_strings":     {"name": "Soft strings",     "instrument": "strings"},
    "pentatonic_flute": {"name": "Pentatonic flute", "instrument": "flute"},
    "mellow_keys":      {"name": "Mellow keys",      "instrument": "vibraphone"},
}
PRESET_IDS = list(PRESETS)
DEFAULT_FIXED_PRESET = "warm_piano"

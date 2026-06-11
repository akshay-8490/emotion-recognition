"""
config.py — Central configuration for Dreamweaver.
All constants, paths, and thresholds live here.
Edit this file to adapt the project to your environment.
"""

import os
import tempfile

import torch

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Paths ──────────────────────────────────────────────────────────────
# Use env var if set (good for deployment), otherwise fall back to a local path.
# Set DREAMWEAVER_MODEL_ROOT in your shell or .env before running.
BASE_MODEL_ROOT = os.getenv(
    "DREAMWEAVER_MODEL_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights"),
)

FACIAL_MODEL_PATH = os.path.join(BASE_MODEL_ROOT, "facial")
AUDIO_MODEL_PATH  = os.path.join(BASE_MODEL_ROOT, "audio")
FUSION_MODEL_PATH = os.path.join(BASE_MODEL_ROOT, "fusion")

# ── Audio Processing ──────────────────────────────────────────────────────────
SR           = 22050        # Sample rate
N_MELS       = 128          # Mel spectrogram bins
FIXED_LENGTH = 3            # Clip length in seconds
SAMPLES      = SR * FIXED_LENGTH
EPSILON      = 1e-6         # Numerical stability

# ── Emotion Classes ───────────────────────────────────────────────────────────
COMMON_EMOTIONS = ["Happy", "Neutral", "Sad"]
NUM_CLASSES     = len(COMMON_EMOTIONS)

# ── Fusion / Confidence Thresholds ────────────────────────────────────────────
MIN_RELIABILITY       = 0.15
CONFLICT_THRESHOLD    = 0.60
FACE_TEMPERATURE      = 1.2   # Higher = softer face predictions
AUDIO_TEMPERATURE     = 1.0
UNCERTAINTY_THRESHOLD = 1.10  # Entropy threshold for "uncertain" flag

# ── Temporal Stabilisation & Context ─────────────────────────
BUFFER_SIZE        = 15
EMA_ALPHA          = 0.35
TRANSITION_PENALTY = 0.12
CONTEXT_WINDOW     = 5
CONTEXT_WEIGHT     = 0.15

# ── Story Generation ──────────────────────────────────────────────────────────
GEMINI_MODEL          = "gemini-2.5-flash"       # Default, overridable from sidebar
GEMINI_MODEL_OPTIONS  = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
STORY_WORD_TARGET     = 200   # ~4 paragraphs
SCENE_COUNT           = 4

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_VOICES = {
    "Happy":   {"voice": "en-US-GuyNeural",   "rate": "+12%", "pitch": "+4Hz"},
    "Sad":     {"voice": "en-US-JennyNeural",  "rate": "-18%", "pitch": "-4Hz"},
    "Neutral": {"voice": "en-US-AriaNeural",   "rate": "+0%",  "pitch": "+0Hz"},
}
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

# ── Image Generation ──────────────────────────────────────────────────────────
POLLINATIONS_BASE_URL  = "https://image.pollinations.ai/prompt"
POLLINATIONS_PRO_URL   = "https://gen.pollinations.ai/image"
IMAGE_STYLE_SUFFIX     = "cinematic fairytale cartoon illustration, emotional lighting, high resolution, cozy"
IMAGE_FETCH_RETRIES    = 3
IMAGE_RETRY_DELAY      = 2    # seconds

# ── Video ─────────────────────────────────────────────────────────────────────
VIDEO_FPS              = 24
VIDEO_HEIGHT           = 720
SUBTITLE_CHARS_PER_LINE = 55
SUBTITLE_FONT_SIZE     = 20
SUBTITLE_LINE_HEIGHT   = 26
SUBTITLE_PADDING       = 30
SUBTITLE_BG_ALPHA      = 160  # 0-255

# ── Temp File Directory ───────────────────────────────────────────────────────
TMP_DIR = tempfile.gettempdir()  # Always writable, works on Streamlit Cloud too

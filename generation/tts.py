"""
generation/tts.py — Text-to-Speech narration synthesis.
Supports ElevenLabs (premium) with automatic fallback to Microsoft Edge TTS (free).

Critical fix from monolithic app.py:
  asyncio.run() crashes inside Streamlit's existing event loop.
  We use nest_asyncio to patch the loop and run coroutines safely.
"""

import os
import asyncio
import tempfile
from typing import Tuple

import requests
import nest_asyncio

from config import TTS_VOICES, ELEVENLABS_VOICE_ID, TMP_DIR

# Patch the event loop once at import time — safe to call multiple times
nest_asyncio.apply()


def generate_narration(
    story_text: str,
    emotion: str,
    elevenlabs_api_key: str = "",
) -> Tuple[str, str]:
    """
    Synthesizes speech from story_text and saves to a temp MP3 file.

    Args:
        story_text:          Full story string.
        emotion:             Detected emotion ("Happy", "Neutral", "Sad").
        elevenlabs_api_key:  Optional. Empty string → skip to Edge TTS.

    Returns:
        (narration_file_path, engine_label)
        narration_file_path: Absolute path to the generated MP3.
        engine_label:        Human-readable label for the UI.
    """
    narration_path = os.path.join(TMP_DIR, "dw_narration.mp3")

    # ── Try ElevenLabs first ───────────────────────────────────────────────────
    if elevenlabs_api_key.strip():
        success, path = _elevenlabs_tts(story_text, narration_path, elevenlabs_api_key.strip())
        if success:
            return path, "ElevenLabs Neural Voice"

    # ── Fallback: Microsoft Edge TTS (free, no key needed) ────────────────────
    path = _edge_tts(story_text, emotion, narration_path)
    return path, "Microsoft Edge TTS (Free Neural Voices)"


def _elevenlabs_tts(text: str, output_path: str, api_key: str) -> Tuple[bool, str]:
    """Calls ElevenLabs API. Returns (success, path)."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.85,
            "style": 0.6,
            "use_speaker_boost": True,
        },
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True, output_path
        return False, ""
    except Exception:
        return False, ""


def _edge_tts(text: str, emotion: str, output_path: str) -> str:
    """
    Uses Microsoft Edge TTS (edge-tts package) with emotion-matched voice settings.
    Runs async coroutine safely inside Streamlit via nest_asyncio.
    """
    import edge_tts

    voice_cfg = TTS_VOICES.get(emotion, TTS_VOICES["Neutral"])

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_cfg["voice"],
        rate=voice_cfg["rate"],
        pitch=voice_cfg["pitch"],
    )

    # nest_asyncio.apply() at module load makes this safe inside Streamlit
    loop = asyncio.get_event_loop()
    loop.run_until_complete(communicate.save(output_path))

    return output_path

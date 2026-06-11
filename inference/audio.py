"""
inference/audio.py — Speech Emotion Recognition inference.
Pure functions, no Streamlit dependency.
"""

import os
import tempfile
from typing import Dict, Any, List

import numpy as np
import torch
import librosa

from config import (
    SR, N_MELS, SAMPLES, EPSILON,
    COMMON_EMOTIONS, AUDIO_TEMPERATURE,
    device,
)
from inference.face import _temperature_scale, _prediction_entropy


def run_audio_prediction(audio_bytes: bytes, audio_models_list: List) -> Dict[str, Any]:
    """
    Processes raw WAV bytes through the AudioCNN ensemble and returns
    a prediction result dict.

    Args:
        audio_bytes:       Raw bytes of a WAV audio file.
        audio_models_list: List of loaded AudioCNN models (eval mode, on device).

    Returns:
        dict with keys: valid, emotion, probabilities, confidence, quality,
                        reliability, entropy
        Returns {"valid": False} on any processing error.
    """
    try:
        # Write bytes to a temp file so librosa can load it
        tmp_wav = os.path.join(tempfile.gettempdir(), "dw_tmp_audio.wav")
        with open(tmp_wav, "wb") as f:
            f.write(audio_bytes)

        y, sr = librosa.load(tmp_wav, sr=SR)

        # Clean up temp file immediately after loading
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

        # Quality estimate from RMS energy before processing
        rms     = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
        quality = min(rms / 0.1, 1.0)

        # Pre-processing: trim silence → normalize → fixed-length
        y, _  = librosa.effects.trim(y, top_db=20)
        y     = librosa.util.normalize(y)
        y     = np.pad(y, (0, max(0, SAMPLES - len(y))))[:SAMPLES]

        # Feature extraction: mel + delta + delta-delta (3-channel spectrogram)
        mel    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, win_length=2048, n_mels=N_MELS
        )
        mel_db  = librosa.power_to_db(mel, ref=np.max)
        delta   = librosa.feature.delta(mel_db)
        delta2  = librosa.feature.delta(mel_db, order=2)

        stacked = np.stack([mel_db, delta, delta2], axis=0)
        for i in range(3):
            stacked[i] = (stacked[i] - stacked[i].mean()) / (stacked[i].std() + EPSILON)

        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)

        # Ensemble inference
        with torch.inference_mode():
            ensemble_probs = None
            for m in audio_models_list:
                p = torch.softmax(m(tensor), dim=1)
                ensemble_probs = p if ensemble_probs is None else ensemble_probs + p
            probs = (ensemble_probs / len(audio_models_list)).cpu().numpy()[0]

        audio_probs = _temperature_scale(probs, AUDIO_TEMPERATURE)
        pred_idx    = int(np.argmax(audio_probs))
        confidence  = float(np.max(audio_probs))
        reliability = 0.6 * confidence + 0.4 * quality
        entropy     = _prediction_entropy(audio_probs)

        return {
            "valid":         True,
            "emotion":       COMMON_EMOTIONS[pred_idx],
            "probabilities": audio_probs,
            "confidence":    confidence,
            "quality":       quality,
            "reliability":   reliability,
            "entropy":       entropy,
        }

    except Exception as e:
        # Silently return invalid — the fusion layer handles missing modalities
        return {"valid": False, "error": str(e)}

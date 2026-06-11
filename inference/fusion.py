"""
inference/fusion.py — Late-fusion of face and audio modality predictions.
Pure functions, no Streamlit dependency.
"""

from typing import Dict, Any, Tuple
from collections import deque

import numpy as np
import torch

from config import (
    COMMON_EMOTIONS, NUM_CLASSES,
    MIN_RELIABILITY, CONFLICT_THRESHOLD, UNCERTAINTY_THRESHOLD,
    device,
    BUFFER_SIZE, EMA_ALPHA, TRANSITION_PENALTY, CONTEXT_WINDOW, CONTEXT_WEIGHT,
)
from inference.face import _prediction_entropy


class ContextAwareness:
    """
    Maintains a sliding window of recent emotions.
    Returns a context prior (probability vector) derived
    from that history.
    """

    def __init__(self, window: int = CONTEXT_WINDOW, num_classes: int = NUM_CLASSES):
        self.window      = window
        self.num_classes = num_classes
        self.history     = deque(maxlen=window)

    def update(self, emotion: str):
        if emotion in COMMON_EMOTIONS:
            idx = COMMON_EMOTIONS.index(emotion)
            self.history.append(idx)

    def get_prior(self) -> np.ndarray:
        """
        Returns a np array [p_Happy, p_Neutral, p_Sad] based
        on recent history. Returns uniform if history empty.
        """
        if len(self.history) == 0:
            return np.ones(self.num_classes) / self.num_classes

        counts = np.zeros(self.num_classes)
        # more recent frames get higher weight
        for rank, idx in enumerate(self.history):
            weight = (rank + 1) / len(self.history)   # linear ramp
            counts[idx] += weight

        prior = counts / counts.sum()
        return prior

    def dominant_emotion(self) -> str:
        """Most common emotion in window."""
        if len(self.history) == 0:
            return "Unknown"
        counts = np.bincount(list(self.history), minlength=self.num_classes)
        return COMMON_EMOTIONS[int(np.argmax(counts))]

    def reset(self):
        self.history.clear()


class TemporalStabilizer:
    """
    Applies Exponential Moving Average (EMA) smoothing and inertia penalties
    to prevent rapid flickering between predicted emotions.
    """

    def __init__(self, alpha: float = EMA_ALPHA, penalty: float = TRANSITION_PENALTY, buffer_size: int = BUFFER_SIZE):
        self.alpha = alpha
        self.penalty = penalty
        self.ema_probs = None
        self.prev_emotion = None
        self.history = deque(maxlen=buffer_size)

    def update(self, fusion_result: Dict[str, Any]) -> Dict[str, Any]:
        probs = fusion_result["probabilities"]
        curr_emo = fusion_result["emotion"]
        self.history.append(probs)

        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs

        smoothed = self.ema_probs.copy()
        if self.prev_emotion is not None and curr_emo != self.prev_emotion:
            curr_idx = COMMON_EMOTIONS.index(curr_emo)
            prev_idx = COMMON_EMOTIONS.index(self.prev_emotion)
            smoothed[curr_idx] -= self.penalty
            smoothed[prev_idx] += self.penalty

        smoothed = np.clip(smoothed, 1e-6, None)
        smoothed /= smoothed.sum()
        stable_idx = int(np.argmax(smoothed))
        stable_emo = COMMON_EMOTIONS[stable_idx]
        self.prev_emotion = stable_emo

        return {
            "emotion": stable_emo,
            "probabilities": smoothed,
            "confidence": float(np.max(smoothed))
        }

    def reset(self):
        self.ema_probs = None
        self.prev_emotion = None
        self.history.clear()


def fuse_emotions(
    face_res: Dict[str, Any],
    audio_res: Dict[str, Any],
    fusion_model,
    context: ContextAwareness = None,
) -> Tuple[str, bool, bool, np.ndarray]:
    """
    Combines face and audio predictions using the learned FusionMLP.
    Handles gracefully when one or both modalities are unavailable.

    Args:
        face_res:      Output dict from run_face_prediction().
        audio_res:     Output dict from run_audio_prediction().
        fusion_model:  Loaded FusionMLP (eval mode, on device).
        context:       Optional ContextAwareness instance.

    Returns:
        (final_emotion, conflict_flag, uncertain_flag, fused_probs)
        - conflict_flag:  True when face & audio strongly disagree.
        - uncertain_flag: True when entropy is high (unfamiliar input).
    """
    face_valid  = face_res.get("valid", False)
    audio_valid = audio_res.get("valid", False)

    # ── Graceful fallbacks when modality is missing ────────────────────────────
    if not face_valid and not audio_valid:
        uniform = np.array([1 / NUM_CLASSES] * NUM_CLASSES)
        return "Neutral", False, True, uniform

    if face_valid and not audio_valid:
        p = face_res["probabilities"]
        return (
            face_res["emotion"],
            False,
            face_res["entropy"] > UNCERTAINTY_THRESHOLD,
            p,
        )

    if audio_valid and not face_valid:
        p = audio_res["probabilities"]
        return (
            audio_res["emotion"],
            False,
            audio_res["entropy"] > UNCERTAINTY_THRESHOLD,
            p,
        )

    # ── Both modalities valid: learned fusion ──────────────────────────────────
    ctx_prior = context.get_prior() if context is not None else np.ones(NUM_CLASSES) / NUM_CLASSES

    feat_vec = np.concatenate([
        face_res["probabilities"],
        audio_res["probabilities"],
        ctx_prior,
        [
            max(face_res["reliability"],  MIN_RELIABILITY),
            max(audio_res["reliability"], MIN_RELIABILITY),
        ],
    ]).astype(np.float32)

    tensor = torch.tensor(feat_vec).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits      = fusion_model(tensor)
        fused_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Blend context history prior
    fused_probs = (1 - CONTEXT_WEIGHT) * fused_probs + CONTEXT_WEIGHT * ctx_prior
    fused_probs = fused_probs / fused_probs.sum()

    final_idx     = int(np.argmax(fused_probs))
    final_emotion = COMMON_EMOTIONS[final_idx]
    entropy       = _prediction_entropy(fused_probs)
    uncertain     = entropy > UNCERTAINTY_THRESHOLD

    conflict = (
        face_res["emotion"] != audio_res["emotion"]
        and face_res["confidence"] > CONFLICT_THRESHOLD
        and audio_res["confidence"] > CONFLICT_THRESHOLD
    )

    return final_emotion, conflict, uncertain, fused_probs

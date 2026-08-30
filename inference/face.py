"""
inference/face.py — Facial emotion recognition inference.
Pure functions, no Streamlit dependency.
"""

from typing import Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import COMMON_EMOTIONS, FACE_TEMPERATURE, device


def compute_face_quality(gray_face: np.ndarray) -> float:
    """
    Estimates image quality based on sharpness (Laplacian variance)
    and brightness proximity to mid-grey.
    Returns a float in [0, 1].
    """
    blur       = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    brightness = np.mean(gray_face)
    blur_norm  = min(blur / 1000.0, 1.0)
    bright_norm = 1.0 - abs(brightness - 127) / 127.0
    return float(np.clip(0.7 * blur_norm + 0.3 * bright_norm, 0.0, 1.0))


def run_face_prediction(cv_image: np.ndarray, face_model, face_detector) -> Dict[str, Any]:
    """
    Detects the largest face in a BGR OpenCV image, runs EfficientNet-B2
    with 4-fold TTA, and returns a result dict.

    Args:
        cv_image:      BGR numpy array (from cv2 or converted from PIL/webcam).
        face_model:    Loaded EfficientNet-B2 (eval mode, on device).
        face_detector: OpenCV Haar cascade classifier.

    Returns:
        dict with keys: valid, emotion, probabilities, confidence, quality,
                        reliability, entropy
        If no face is detected, returns {"valid": False}.
    """
    faces = []
    if face_detector is not None and hasattr(face_detector, "detectMultiScale"):
        try:
            gray  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
        except Exception:
            faces = []

    if len(faces) > 0:
        # Pick the largest detected face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop   = cv_image[y : y + h, x : x + w]
    else:
        # Fall back to full image if no face bounding box found or detector unavailable
        face_crop   = cv_image

    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    quality   = compute_face_quality(gray_crop)

    pil_face  = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    face_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4-fold Test-Time Augmentation
    tta_transforms = [
        lambda img: face_transform(img),
        lambda img: face_transform(img.transpose(Image.FLIP_LEFT_RIGHT)),
        lambda img: face_transform(img.resize((232, 232)).crop((4, 4, 228, 228))),
        lambda img: face_transform(transforms.ColorJitter(brightness=0.1, contrast=0.1)(img)),
    ]

    tta_probs = []
    with torch.inference_mode():
        for tfm in tta_transforms:
            t    = tfm(pil_face).unsqueeze(0).to(device)
            out  = face_model(t)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            tta_probs.append(prob)

    face_probs  = _temperature_scale(np.mean(tta_probs, axis=0), FACE_TEMPERATURE)
    pred_idx    = int(np.argmax(face_probs))
    confidence  = float(np.max(face_probs))
    reliability = 0.7 * confidence + 0.3 * quality
    entropy     = _prediction_entropy(face_probs)

    return {
        "valid":         True,
        "emotion":       COMMON_EMOTIONS[pred_idx],
        "probabilities": face_probs,
        "confidence":    confidence,
        "quality":       quality,
        "reliability":   reliability,
        "entropy":       entropy,
    }


# ── Helpers (also used by audio.py and fusion.py via import) ──────────────────

def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    log_p   = np.log(np.clip(probs, 1e-9, 1.0))
    scaled  = log_p / temperature
    shifted = scaled - np.max(scaled)
    exp_p   = np.exp(shifted)
    return exp_p / exp_p.sum()


def _prediction_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log2(probs)))

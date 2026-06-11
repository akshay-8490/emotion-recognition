"""
models/loader.py — Loads all trained model weights from disk.
Uses @st.cache_resource so models are loaded once per Streamlit session.
"""

import os
from typing import Tuple

import cv2
import torch
import torchvision.models as tv_models
import torch.nn as nn
import streamlit as st

from config import (
    FACIAL_MODEL_PATH,
    AUDIO_MODEL_PATH,
    FUSION_MODEL_PATH,
    NUM_CLASSES,
    device,
)
from models.architectures import AudioCNN, FusionMLP


@st.cache_resource(show_spinner="Loading emotion recognition models...")
def load_all_models():
    """
    Loads face, audio (ensemble), and fusion models from disk.

    Returns:
        (face_model, audio_models_list, fusion_model, face_detector)
        OR raises FileNotFoundError with a descriptive message on failure.
    """

    # ── 1. Facial Model (EfficientNet-B2) ─────────────────────────────────────
    face_model_path = os.path.join(FACIAL_MODEL_PATH, "best_face_model_stage2_3class.pth")
    if not os.path.exists(face_model_path):
        raise FileNotFoundError(
            f"FER weights not found at: {face_model_path}\n"
            "Please run the training notebooks (Nb1 & Nb2) first, "
            "or set DREAMWEAVER_MODEL_ROOT to the correct folder."
        )

    face_model = tv_models.efficientnet_b2(weights=None)
    in_features = face_model.classifier[1].in_features
    face_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    face_model.load_state_dict(
        torch.load(face_model_path, map_location=device, weights_only=True)
    )
    face_model = face_model.to(device).eval()

    # ── 2. Audio Model (Ensemble or single seed) ───────────────────────────────
    ensemble_path = os.path.join(AUDIO_MODEL_PATH, "ensemble_audio_model.pth")
    single_path   = os.path.join(AUDIO_MODEL_PATH, "final_best_audio_model.pth")
    audio_models_list = []

    if os.path.exists(ensemble_path):
        state_dicts = torch.load(ensemble_path, map_location=device, weights_only=True)
        for sd in state_dicts:
            m = AudioCNN(num_classes=NUM_CLASSES)
            m.load_state_dict(sd)
            audio_models_list.append(m.to(device).eval())
    elif os.path.exists(single_path):
        m = AudioCNN(num_classes=NUM_CLASSES)
        m.load_state_dict(
            torch.load(single_path, map_location=device, weights_only=True)
        )
        audio_models_list.append(m.to(device).eval())
    else:
        raise FileNotFoundError(
            f"SER weights not found. Checked:\n  {ensemble_path}\n  {single_path}"
        )

    # ── 3. Fusion MLP ──────────────────────────────────────────────────────────
    fusion_bundle_path = os.path.join(FUSION_MODEL_PATH, "fusion_bundle.pt")
    if not os.path.exists(fusion_bundle_path):
        raise FileNotFoundError(f"Fusion bundle not found at: {fusion_bundle_path}")

    bundle = torch.load(fusion_bundle_path, map_location=device, weights_only=True)
    fusion_model = FusionMLP(
        input_dim=bundle["fusion_input_dim"],
        hidden=bundle["fusion_hidden"],
        num_classes=bundle["num_classes"],
    )
    fusion_model.load_state_dict(bundle["fusion_mlp_state"])
    fusion_model = fusion_model.to(device).eval()

    # ── 4. Haar Cascade (OpenCV built-in, no weights file needed) ─────────────
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return face_model, audio_models_list, fusion_model, face_detector

"""
ui/input_panel.py — Left-column input widgets for face and audio capture.
Returns (cv_image, audio_bytes) ready for inference.

Enhancement: includes "Record Audio" option using streamlit-audiorec for live mic input.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def render_input_panel() -> Tuple[Optional[np.ndarray], Optional[bytes]]:
    """
    Renders face capture and audio upload widgets.

    Returns:
        cv_image:    BGR numpy array if face input was provided, else None.
        audio_bytes: Raw WAV bytes if audio was uploaded/recorded, else None.
    """
    cv_image    = _render_face_section()
    audio_bytes = _render_audio_section()
    return cv_image, audio_bytes


def _render_face_section() -> Optional[np.ndarray]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📸 Face Capture")

    face_mode = st.radio(
        "Face input method",
        ["Skip Face", "Use Webcam", "Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    cv_image = None

    if face_mode == "Use Webcam":
        webcam_file = st.camera_input("Smile for the camera! 😊")
        if webcam_file:
            pil_img  = Image.open(webcam_file)
            cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.success("✅ Webcam photo captured!")

    elif face_mode == "Upload Image":
        upload_img = st.file_uploader(
            "Choose a photo…", type=["jpg", "jpeg", "png"], key="face_upload"
        )
        if upload_img:
            pil_img  = Image.open(upload_img)
            cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.image(pil_img, caption="Uploaded image", width=260)

    st.markdown("</div>", unsafe_allow_html=True)
    return cv_image


def _render_audio_section() -> Optional[bytes]:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎤 Voice Recording")

    audio_mode = st.radio(
        "Audio input method",
        ["Skip Audio", "Record Audio", "Upload WAV File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    audio_bytes = None

    if audio_mode == "Record Audio":
        try:
            from st_audiorec import st_audiorec
            recorded = st_audiorec()
            if recorded and len(recorded) > 0:
                audio_bytes = recorded
                st.audio(audio_bytes, format="audio/wav")
                st.success("✅ Audio recorded!")
        except ImportError:
            st.warning(
                "⚠️ `streamlit-audiorec` is not installed. "
                "Run `pip install streamlit-audiorec` to enable live recording. "
                "Using file upload instead."
            )
            upload_aud = st.file_uploader(
                "Upload a WAV recording…", type=["wav"], key="audio_upload_fallback"
            )
            if upload_aud:
                audio_bytes = upload_aud.read()
                st.audio(audio_bytes, format="audio/wav")
                st.success("✅ Audio loaded!")

    elif audio_mode == "Upload WAV File":
        upload_aud = st.file_uploader(
            "Upload a WAV recording…", type=["wav"], key="audio_upload"
        )
        if upload_aud:
            audio_bytes = upload_aud.read()
            st.audio(audio_bytes, format="audio/wav")
            st.success("✅ Audio loaded!")

    st.markdown(
        "<small style='color:#777;'>💡 Tip: record a few seconds of natural speech "
        "for best emotion detection results.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return audio_bytes

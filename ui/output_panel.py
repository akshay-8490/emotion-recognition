"""
ui/output_panel.py — Right-column output display.
Shows detected emotion, probability chart, story text, and video player.
"""

import os

import numpy as np
import streamlit as st

from config import COMMON_EMOTIONS, TMP_DIR


def render_output_panel() -> None:
    """Renders the full right-column output panel from st.session_state."""

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📖 Personalized Bedtime Story")

    if not st.session_state.get("final_emotion"):
        st.info(
            "Provide face and/or audio input on the left, "
            "then click **🚀 Generate Story** to begin."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    emotion = st.session_state.final_emotion

    # ── Emotion result ─────────────────────────────────────────────────────────
    badge_class = f"badge-{emotion.lower()}"
    st.markdown(
        f"**Detected Emotion:** "
        f"<span class='emotion-badge {badge_class}'>{emotion}</span>",
        unsafe_allow_html=True,
    )

    # ── Probability bar chart ─────────────────────────────────────────────────
    probs = st.session_state.get("fused_probs")
    if probs is not None:
        _render_probability_chart(probs)

    if st.session_state.get("tts_engine"):
        st.caption(f"🔊 Voice: {st.session_state.tts_engine}")

    # ── Story text ─────────────────────────────────────────────────────────────
    story = st.session_state.get("story_text")
    if story:
        st.markdown("---")
        st.markdown(story)

    # ── Video player + download ────────────────────────────────────────────────
    video_path = os.path.join(TMP_DIR, "dw_story_video.mp4")
    if st.session_state.get("video_ready") and os.path.exists(video_path):
        st.markdown("---")
        st.subheader("🎥 Story Video")
        st.video(video_path)

        with open(video_path, "rb") as f:
            st.download_button(
                label="💾 Download Story Video",
                data=f,
                file_name=f"dreamweaver_{emotion.lower()}_story.mp4",
                mime="video/mp4",
            )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_probability_chart(probs: np.ndarray) -> None:
    """Renders a horizontal bar chart of emotion probabilities using native Streamlit."""
    with st.expander("📊 Emotion Confidence Breakdown", expanded=False):
        for label, prob in zip(COMMON_EMOTIONS, probs):
            col1, col2 = st.columns([2, 8])
            with col1:
                st.markdown(f"**{label}**")
            with col2:
                st.progress(float(prob), text=f"{prob*100:.1f}%")

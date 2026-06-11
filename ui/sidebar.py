"""
ui/sidebar.py — Sidebar configuration panel.
Returns API keys and settings as a simple dict so app.py stays clean.

Enhancement: includes Gemini model selection dropdown.
"""

import streamlit as st

from config import GEMINI_MODEL_OPTIONS


def render_sidebar() -> dict:
    """
    Renders the sidebar configuration panel.

    Returns:
        dict with keys: gemini_key, gemini_model, elevenlabs_key, pollinations_key
    """
    st.sidebar.title("🛠️ Configuration")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**🤖 AI Story Generation**")
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Required for dynamic story generation. Leave blank to use a built-in story.",
    )

    gemini_model = st.sidebar.selectbox(
        "Gemini Model",
        options=GEMINI_MODEL_OPTIONS,
        index=0,
        help="Choose which Gemini model to use for story generation.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎙️ Voice Narration**")
    elevenlabs_key = st.sidebar.text_input(
        "ElevenLabs API Key",
        type="password",
        help="Optional. Leave blank to use free Microsoft Edge neural voices.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎨 Scene Illustrations**")
    pollinations_key = st.sidebar.text_input(
        "Pollinations API Key",
        type="password",
        help="Optional. Pro key for higher-quality image generation.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#777;'>Keys are not stored — they are used only for this session.</small>",
        unsafe_allow_html=True,
    )

    return {
        "gemini_key":       gemini_key,
        "gemini_model":     gemini_model,
        "elevenlabs_key":   elevenlabs_key,
        "pollinations_key": pollinations_key,
    }

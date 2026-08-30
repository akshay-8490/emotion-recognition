"""
ui/sidebar.py — Sidebar configuration panel.
Returns API keys and settings as a simple dict so app.py stays clean.

Enhancement: includes Gemini model selection dropdown.
"""

import streamlit as st
import google.generativeai as genai

from config import GEMINI_MODEL_OPTIONS


@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_gemini_models(api_key: str) -> list:
    """
    Attempts to fetch available text generation models for the provided Gemini API key.
    Filters models that support content generation and formats names cleanly.
    """
    if not api_key or not api_key.strip():
        return []
    try:
        genai.configure(api_key=api_key.strip())
        models = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", [])
            if "generateContent" in methods:
                name = m.name.replace("models/", "")
                models.append(name)
        return models
    except Exception:
        return []


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
        help="Required for dynamic story generation. Get a free tier key from Google AI Studio.",
    )

    available_models = []
    if gemini_key.strip():
        available_models = fetch_available_gemini_models(gemini_key.strip())

    options = list(GEMINI_MODEL_OPTIONS)
    if available_models:
        # Prepend dynamically fetched models while preserving uniqueness
        for m in reversed(available_models):
            if m in options:
                options.remove(m)
            options.insert(0, m)
        st.sidebar.caption(f"✨ Auto-detected {len(available_models)} models for your API key")

    selected_option = st.sidebar.selectbox(
        "Gemini Model",
        options=options,
        index=0,
        help="Choose a Gemini model (e.g. gemini-3.5-flash or gemini-2.5-flash for free tier).",
    )

    if selected_option == "Custom / Enter manually...":
        gemini_model = st.sidebar.text_input(
            "Custom Model Identifier",
            value="gemini-3.5-flash",
            help="Enter any valid model name from Google AI Studio (e.g., gemini-3.5-flash, gemini-1.5-flash).",
        ).strip()
    else:
        gemini_model = selected_option

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

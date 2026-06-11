"""
app.py — Dreamweaver entry point.
Run with: streamlit run app.py

This file is intentionally thin — it only orchestrates.
All logic lives in the imported modules.
"""

import os
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Dreamweaver — Personalized Bedtime Stories",
    page_icon="🌙",
    layout="wide",
)

# ── UI modules ────────────────────────────────────────────────────────────────
from ui.styles       import apply_theme
from ui.sidebar      import render_sidebar
from ui.input_panel  import render_input_panel
from ui.output_panel import render_output_panel

# ── ML / backend modules ──────────────────────────────────────────────────────
from models.loader       import load_all_models
from inference.face      import run_face_prediction
from inference.audio     import run_audio_prediction
from inference.fusion    import fuse_emotions, ContextAwareness, TemporalStabilizer
from generation.story    import generate_story
from generation.tts      import generate_narration
from generation.images   import fetch_scene_images
from video.assembler     import assemble_video
from config              import TMP_DIR

# ── Apply theme ───────────────────────────────────────────────────────────────
apply_theme()

st.markdown("<h1>🌙 Dreamweaver</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px; color:#A0A0B0;'>"
    "A magical bedtime story generator that senses your child's emotions "
    "and guides them into sweet dreams.</p>",
    unsafe_allow_html=True,
)

# ── Load models once (cached across reruns) ───────────────────────────────────
try:
    face_model, audio_models, fusion_model, face_detector = load_all_models()
except FileNotFoundError as e:
    st.error(f"❌ Model weights not found.\n\n{e}")
    st.stop()

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "final_emotion": None,
    "story_text":    None,
    "scene_prompts": None,
    "fused_probs":   None,
    "video_ready":   False,
    "tts_engine":    None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "session_context" not in st.session_state:
    st.session_state.session_context = ContextAwareness()
if "stabilizer" not in st.session_state:
    st.session_state.stabilizer = TemporalStabilizer()

# ── Layout ────────────────────────────────────────────────────────────────────
keys = render_sidebar()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    cv_image, audio_bytes = render_input_panel()

    if st.button("🚀 Generate Story", use_container_width=True):
        progress = st.progress(0, text="🔍 Analysing emotion…")

        # Step 1 — Unimodal inference
        face_res  = run_face_prediction(cv_image, face_model, face_detector) \
                    if cv_image is not None else {"valid": False}
        audio_res = run_audio_prediction(audio_bytes, audio_models) \
                    if audio_bytes is not None else {"valid": False}
        progress.progress(16, text="🔗 Fusing modalities…")

        # Step 2 — Fusion
        raw_emotion, conflict, uncertain, raw_fused_probs = fuse_emotions(
            face_res, audio_res, fusion_model, st.session_state.session_context
        )
        
        # Apply temporal stabilizer
        stable_res = st.session_state.stabilizer.update({
            "emotion": raw_emotion,
            "probabilities": raw_fused_probs
        })
        st.session_state.session_context.update(stable_res["emotion"])
        
        emotion = stable_res["emotion"]
        fused_probs = stable_res["probabilities"]

        st.session_state.final_emotion = emotion
        st.session_state.fused_probs   = fused_probs

        st.info(f"✨ Detected Emotion: **{emotion}**")
        if conflict:
            st.warning("⚠️ Face and voice signals conflict — using fusion result.")
        if uncertain:
            st.warning("⚠️ Low-confidence prediction — input may be ambiguous.")

        # Step 3 — Story generation
        progress.progress(33, text="📝 Writing your story…")
        story_text, scene_prompts = generate_story(
            emotion, keys["gemini_key"], keys["gemini_model"]
        )
        st.session_state.story_text    = story_text
        st.session_state.scene_prompts = scene_prompts

        # Step 4 — TTS narration
        progress.progress(50, text="🎙️ Recording narration…")
        narration_path, tts_label = generate_narration(
            story_text, emotion, keys["elevenlabs_key"]
        )
        st.session_state.tts_engine = tts_label

        # Step 5 — Scene illustrations
        progress.progress(66, text="🎨 Painting scenes…")
        image_paths = fetch_scene_images(scene_prompts, keys["pollinations_key"])

        # Step 6 — Video assembly
        progress.progress(83, text="🎬 Assembling video…")
        assemble_video(image_paths, narration_path, story_text)

        progress.progress(100, text="✅ Done!")
        st.session_state.video_ready = True
        st.balloons()

    # ── Reset button ──────────────────────────────────────────────────────────
    if st.session_state.final_emotion:
        if st.button("🔄 Reset", use_container_width=True):
            # Clean up temp video file
            video_path = os.path.join(TMP_DIR, "dw_story_video.mp4")
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

            # Reset stateful session context & stabilizer
            if "session_context" in st.session_state:
                st.session_state.session_context.reset()
            if "stabilizer" in st.session_state:
                st.session_state.stabilizer.reset()

            for key in ["final_emotion", "story_text", "scene_prompts",
                        "fused_probs", "video_ready", "tts_engine"]:
                st.session_state[key] = None
            st.session_state.video_ready = False
            st.rerun()

with col_right:
    render_output_panel()

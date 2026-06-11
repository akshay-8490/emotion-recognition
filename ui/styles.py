"""
ui/styles.py — Dreamweaver CSS theme injection.
Call apply_theme() once at the top of app.py.
"""

import streamlit as st


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
.main {
    background: #11111E;
    color: #E2E2E2;
}
h1 {
    color: #FFB703;
    font-weight: 700;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
}
h2, h3 {
    color: #2196F3;
}
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #3F3D56 100%);
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(108,99,255,0.4);
    transition: transform 0.2s;
}
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #8B85FF 0%, #4E4C6D 100%);
}
.glass-card {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.emotion-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
    margin-top: 8px;
}
.badge-happy   { background: rgba(255,183,3,0.2);  color: #FFB703; border: 1px solid #FFB703; }
.badge-sad     { background: rgba(33,150,243,0.2); color: #2196F3; border: 1px solid #2196F3; }
.badge-neutral { background: rgba(76,175,80,0.2);  color: #4CAF50; border: 1px solid #4CAF50; }
</style>
"""


def apply_theme() -> None:
    """Injects the Dreamweaver CSS theme into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)

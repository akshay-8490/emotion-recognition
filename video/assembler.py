"""
video/assembler.py — Assembles the final story video from images + audio.
Handles subtitle overlay directly on frames (no TextClip — avoids ImageMagick dep).

Critical fixes from monolithic app.py:
  - All temp files go to TMP_DIR (/tmp), not the CWD
  - Cross-platform font fallback with dynamic Google Fonts download
  - Proper cleanup of MoviePy clips to prevent resource leaks
"""

import os
import shutil
import tempfile
from typing import List

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

from config import (
    VIDEO_FPS,
    VIDEO_HEIGHT,
    SUBTITLE_CHARS_PER_LINE,
    SUBTITLE_FONT_SIZE,
    SUBTITLE_LINE_HEIGHT,
    SUBTITLE_PADDING,
    SUBTITLE_BG_ALPHA,
    TMP_DIR,
)


def assemble_video(
    image_paths: List[str],
    narration_path: str,
    story_text: str,
) -> str:
    """
    Builds an MP4 slideshow from scene images + narration audio with subtitle overlays.

    Args:
        image_paths:    List of PNG file paths (one per scene).
        narration_path: Path to the MP3 narration file.
        story_text:     Full story string (paragraphs separated by double newlines).

    Returns:
        Absolute path to the assembled MP4 file in TMP_DIR.
    """
    # ── Match paragraphs to image slots ───────────────────────────────────────
    raw_paragraphs = [p.strip() for p in story_text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [story_text]

    paragraphs = [
        raw_paragraphs[i] if i < len(raw_paragraphs) else ""
        for i in range(len(image_paths))
    ]

    # ── Burn subtitles onto image copies ──────────────────────────────────────
    subtitled_paths = []
    for img_path, text in zip(image_paths, paragraphs):
        out_path = img_path.replace(".png", "_sub.png")
        _burn_subtitle(img_path, text, out_path)
        subtitled_paths.append(out_path)

    # ── MoviePy assembly ───────────────────────────────────────────────────────
    audio_clip     = AudioFileClip(narration_path)
    slide_duration = audio_clip.duration / len(subtitled_paths)

    clips = [
        ImageClip(p).set_duration(slide_duration).resize(height=VIDEO_HEIGHT)
        for p in subtitled_paths
    ]

    base_video  = concatenate_videoclips(clips, method="compose")
    final_video = base_video.set_audio(audio_clip)

    output_path = os.path.join(TMP_DIR, "dw_story_video.mp4")
    final_video.write_videofile(
        output_path,
        fps=VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    audio_clip.close()
    final_video.close()
    for c in clips:
        try:
            c.close()
        except Exception:
            pass

    for p in subtitled_paths:
        if os.path.exists(p):
            os.remove(p)

    return output_path


def _burn_subtitle(img_path: str, text: str, output_path: str) -> None:
    """
    Draws a semi-transparent subtitle bar at the bottom of the image.
    Uses cross-platform font fallback — no arial.ttf dependency.
    """
    if not text:
        # No subtitle — just copy the file
        shutil.copy(img_path, output_path)
        return

    pil_img        = Image.open(img_path).convert("RGB")
    width, height  = pil_img.size
    lines          = _wrap_text(text, SUBTITLE_CHARS_PER_LINE)
    font           = _get_font(SUBTITLE_FONT_SIZE)
    box_height     = len(lines) * SUBTITLE_LINE_HEIGHT + SUBTITLE_PADDING

    # Semi-transparent dark bar
    overlay      = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle(
        [(0, height - box_height), (width, height)],
        fill=(0, 0, 0, SUBTITLE_BG_ALPHA),
    )
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")

    draw   = ImageDraw.Draw(pil_img)
    text_y = height - box_height + 15
    for line in lines:
        try:
            w      = draw.textlength(line, font=font)
            text_x = (width - w) / 2
        except Exception:
            text_x = 40
        draw.text((text_x, text_y), line, font=font, fill=(255, 255, 255))
        text_y += SUBTITLE_LINE_HEIGHT

    pil_img.save(output_path)


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Word-wraps text into lines of at most max_chars characters."""
    words        = text.split()
    lines        = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(" ".join(current_line)) > max_chars:
            current_line.pop()
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Attempts to load a sensible font, falling back gracefully.
    Works on Windows, macOS, and Linux (including Streamlit Cloud / Ubuntu).
    As a last resort, downloads Roboto from Google Fonts into TMP_DIR.
    """
    candidates = [
        # Linux (Ubuntu / Streamlit Cloud)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # ── Last resort: download Roboto from Google Fonts ─────────────────────────
    roboto_path = os.path.join(TMP_DIR, "Roboto-Regular.ttf")
    if not os.path.exists(roboto_path):
        try:
            import requests
            url = (
                "https://github.com/google/fonts/raw/main/ofl/roboto/"
                "Roboto%5Bwdth%2Cwght%5D.ttf"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(roboto_path, "wb") as f:
                    f.write(resp.content)
        except Exception:
            pass

    if os.path.exists(roboto_path):
        try:
            return ImageFont.truetype(roboto_path, size)
        except Exception:
            pass

    return ImageFont.load_default()

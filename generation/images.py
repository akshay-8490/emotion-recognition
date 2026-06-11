"""
generation/images.py — Scene illustration fetching via Pollinations AI.
Falls back to a styled placeholder image if the request fails.
"""

import io
import os
import time
import urllib.parse
import tempfile
from typing import List, Optional

import requests
from PIL import Image, ImageDraw

from config import (
    POLLINATIONS_BASE_URL,
    POLLINATIONS_PRO_URL,
    IMAGE_STYLE_SUFFIX,
    IMAGE_FETCH_RETRIES,
    IMAGE_RETRY_DELAY,
    TMP_DIR,
)

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}


def fetch_scene_images(
    scene_prompts: List[str],
    pollinations_api_key: str = "",
) -> List[str]:
    """
    Downloads one illustration per scene prompt, saves to /tmp, returns file paths.

    Args:
        scene_prompts:        List of scene description strings.
        pollinations_api_key: Optional. Uses pro endpoint if provided.

    Returns:
        List of absolute file paths to saved PNG images (same length as scene_prompts).
    """
    image_paths = []
    for i, scene in enumerate(scene_prompts):
        path = _fetch_single_image(scene, i + 1, pollinations_api_key)
        image_paths.append(path)
    return image_paths


def _fetch_single_image(scene: str, index: int, api_key: str) -> str:
    """Fetches one image, retries up to IMAGE_FETCH_RETRIES times, then falls back."""
    full_prompt    = f"{IMAGE_STYLE_SUFFIX}, {scene}"
    encoded_prompt = urllib.parse.quote(full_prompt)

    if api_key.strip():
        url = f"{POLLINATIONS_PRO_URL}/{encoded_prompt}?key={api_key.strip()}"
    else:
        url = f"{POLLINATIONS_BASE_URL}/{encoded_prompt}"

    img_data: Optional[bytes] = None
    for attempt in range(IMAGE_FETCH_RETRIES):
        try:
            res = requests.get(url, headers=_REQUEST_HEADERS, timeout=30)
            if res.status_code == 200 and len(res.content) > 1000:
                img_data = res.content
                break
        except Exception:
            pass
        if attempt < IMAGE_FETCH_RETRIES - 1:
            time.sleep(IMAGE_RETRY_DELAY)

    output_path = os.path.join(TMP_DIR, f"dw_scene_{index}.png")

    if img_data:
        with open(output_path, "wb") as f:
            f.write(img_data)
    else:
        # Generate a styled placeholder so the pipeline never breaks
        _create_placeholder_image(output_path, index, scene)

    return output_path


def _create_placeholder_image(path: str, index: int, scene_text: str) -> None:
    """Creates a dark-themed fallback image with the scene description as text."""
    img  = Image.new("RGB", (1024, 768), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)

    # Simple centered text — no font file dependency
    draw.text((40, 320), f"Scene {index}", fill=(180, 180, 255))
    draw.text((40, 360), scene_text[:80], fill=(200, 200, 200))

    img.save(path, format="PNG")

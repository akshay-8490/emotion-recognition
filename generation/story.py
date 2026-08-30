"""
generation/story.py — AI story and scene-prompt generation via Gemini.
Falls back to a pre-written story when no API key is provided.

Enhancement: accepts gemini_model_name to allow model selection from sidebar.
"""

import json
from typing import Tuple, List

import google.generativeai as genai

from config import GEMINI_MODEL, SCENE_COUNT


# ── Fallback stories per emotion (no API key needed) ──────────────────────────
_FALLBACK_STORIES = {
    "Happy": {
        "story": (
            "Once upon a time, a little firefly named Blink zipped through a golden meadow, "
            "painting the grass with tiny sparks of light.\n\n"
            "Blink's best friend, a fluffy cloud named Nimbus, floated overhead, "
            "dropping soft dandelion puffs that danced in the evening breeze.\n\n"
            "Together they chased the last rays of the setting sun, laughing and tumbling "
            "through fields of lavender until the stars came out to play.\n\n"
            "As the moon rose high, Blink settled on a warm leaf, Nimbus pillowed above, "
            "and both drifted into the sweetest, happiest dreams."
        ),
        "scenes": [
            "A glowing firefly flying over a golden meadow at sunset, cartoon illustration",
            "A fluffy white cloud dropping dandelion puffs, soft evening light",
            "A firefly and a cloud chasing the sunset through lavender fields",
            "A firefly resting on a leaf under a full moon, cozy and warm",
        ],
    },
    "Sad": {
        "story": (
            "Deep in a cozy forest, a little bear named Cosmo sat by a warm crackling fire, "
            "wrapped snugly in his favourite honey-coloured blanket.\n\n"
            "His friend Maple the owl landed gently beside him, and without a word, "
            "tucked a wing around Cosmo's shoulder — because sometimes, no words are needed.\n\n"
            "The fire crackled softly, the stars appeared one by one like tiny night-lights, "
            "and Cosmo felt the heaviness in his heart slowly melt into warmth.\n\n"
            "By the time the last ember glowed, Cosmo was fast asleep, safe and loved, "
            "dreaming of sunny mornings and the smell of fresh honey."
        ),
        "scenes": [
            "A small bear sitting by a warm campfire wrapped in a blanket, forest night",
            "A wise owl gently placing a wing around a bear, soft warm lighting",
            "Stars appearing one by one over a quiet forest, magical night sky",
            "A bear sleeping peacefully by dying embers, cozy safe atmosphere",
        ],
    },
    "Neutral": {
        "story": (
            "On a quiet hillside, a young rabbit named Pip listened to the evening sounds — "
            "crickets humming, leaves rustling, a faraway stream whispering its gentle song.\n\n"
            "Pip closed his eyes and followed the stream's sound in his mind, past smooth pebbles "
            "and mossy banks, through a tunnel of willow trees bowing in the breeze.\n\n"
            "The wind grew softer, the crickets slower, and Pip's breathing deepened "
            "with each imagined ripple of the water.\n\n"
            "When he opened his eyes, the first star of the night was twinkling just for him — "
            "a quiet, perfect end to a quiet, perfect day."
        ),
        "scenes": [
            "A small rabbit sitting on a hillside at dusk, listening peacefully",
            "A winding stream through mossy banks under willow trees, twilight",
            "A gentle breeze moving through tall grass, soft golden light",
            "A rabbit looking up at the first star of the night, calm and serene",
        ],
    },
}


def generate_story(
    emotion: str,
    gemini_api_key: str,
    gemini_model_name: str = "",
) -> Tuple[str, List[str]]:
    """
    Generates a bedtime story and 4 scene illustration prompts.

    Args:
        emotion:            One of "Happy", "Neutral", "Sad".
        gemini_api_key:     Gemini API key string (empty string → use fallback).
        gemini_model_name:  Optional Gemini model override (from sidebar dropdown).

    Returns:
        (story_text, scene_prompts_list)
    """
    if not gemini_api_key.strip():
        fallback = _FALLBACK_STORIES.get(emotion, _FALLBACK_STORIES["Neutral"])
        return fallback["story"], fallback["scenes"]

    try:
        genai.configure(api_key=gemini_api_key.strip())
        raw_model_name = gemini_model_name.strip() if gemini_model_name.strip() else GEMINI_MODEL
        # Clean model name (GenerativeModel works with or without models/ prefix)
        model_name = raw_model_name.replace("models/", "")
        model = genai.GenerativeModel(model_name)

        prompt = f"""Write a soothing bedtime story for a child whose detected emotion is {emotion}.

Guidelines:
- Happy  → fun, gentle, playful fantasy adventure leading to peaceful sleep
- Sad    → deeply comforting story about friendship, warmth, and feeling safe
- Neutral → relaxing, rhythmic tale focusing on cozy nature sounds and calm

Requirements:
- Exactly {SCENE_COUNT} paragraphs (~{200 // SCENE_COUNT} words each)
- Age-appropriate language (5–10 years)
- Ends with the child/character falling asleep peacefully
- Each paragraph maps to one illustration scene

Return ONLY valid JSON in this exact structure (no markdown fences):
{{
  "story": "paragraph1\\n\\nparagraph2\\n\\nparagraph3\\n\\nparagraph4",
  "scenes": ["scene 1 description", "scene 2 description", "scene 3 description", "scene 4 description"]
}}"""

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )

        # Strip any accidental markdown fences before parsing
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        content = json.loads(raw)

        story  = content["story"]
        scenes = content["scenes"][:SCENE_COUNT]   # Guard against over-generation
        return story, scenes

    except Exception as e:
        print(f"⚠️ Gemini story generation failed with model '{gemini_model_name}': {e}. Falling back to pre-written story.")
        # Fallback gracefully — don't crash the whole pipeline
        fallback = _FALLBACK_STORIES.get(emotion, _FALLBACK_STORIES["Neutral"])
        return (
            fallback["story"],
            fallback["scenes"],
        )

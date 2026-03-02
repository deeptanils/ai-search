"""Unified GPT-4o vision extraction for all pipeline dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.models import ImageExtraction

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert cinematic image analyst — think like a movie "
    "director analyzing a movie still. Specialize in mythology, "
    "religion, folklore, cultural art, and visual storytelling. Given "
    "an image and its generation prompt, extract comprehensive "
    "structured information.\n\n"
    "CRITICAL — Character & Scene Identification:\n"
    "Identify every mythological, religious, or cultural character by "
    "PROPER NAME (Hanuman, Rama, Sita, Ravana, Krishna, Arjuna, "
    "Lakshman, Sugriva, Vibhishana). Name the SPECIFIC STORY EPISODE "
    "(e.g. 'Lanka Dahan', 'Sanjeevani quest', 'Setu Bandhan', "
    "'Sita Haran', 'Swayamvar', 'Agni Pariksha', 'Bali Vadh'). "
    "Do NOT use generic terms like 'humanoid figure' or 'monkey-like "
    "creature' when a named character is identifiable.\n\n"
    "CRITICAL — Distinguish similar characters in different situations. "
    "Two images of Hanuman must produce VERY DIFFERENT descriptions "
    "if one shows him flying with a mountain and the other shows him "
    "fighting in Lanka. Focus on the dimensions below.\n\n"
    "For semantic_description (250 words, MANDATORY dimensions):\n"
    "- CHARACTERS: Name every character visible by proper name\n"
    "- ACTION: Specific verb — flying, fighting, kneeling, blessing, "
    "lifting, burning, talking, meditating, mourning, charging\n"
    "- WEAPONS & PROPS: Gada/mace, bow (Kodanda, Sharanga, Pinaka), "
    "arrows, mountain (Dronagiri/Sanjeevani), torch, chariot, trident, "
    "chakra (Sudarshana), lotus, Pushpaka Vimana, bridge/setu, "
    "Shiva's bow, golden deer\n"
    "- COSTUME & JEWELRY: Armor, crown, royal robes, forest attire, "
    "dhoti, war paint, ornamental jewelry, sacred thread (yajnopavita), "
    "anklets, armlets, mukut/crown, earrings, garlands\n"
    "- CHARACTER FORM/AVATAR: Giant Hanuman, multi-headed Ravana (10 "
    "heads), blue-skinned Rama/Krishna/Vishnu, four-armed Vishnu, "
    "golden-furred vanara, demon/rakshasa form\n"
    "- LOCATION: Battlefield (Lanka), palace (Ayodhya, Lanka throne "
    "room), forest (Dandaka, Panchavati), ocean (Ram Setu crossing), "
    "sky/clouds, Ashoka Vatika garden, Kishkindha caves, mountain "
    "peak, riverside, bridge construction site\n"
    "- EPISODE: Name the specific mythological episode from Ramayana, "
    "Mahabharata, or other source\n"
    "- EMOTIONAL TONE: Fury, devotion, sorrow, triumph, serenity, "
    "determination, anguish, love, heroism, sacrifice\n"
    "- SCALE & RELATIONSHIP: Relative sizes of characters, divine vs "
    "mortal presence, army vs individual\n\n"
    "For structural_description (150 words):\n"
    "Spatial composition, character count and positioning, scale "
    "relationships (giant vs normal, divine vs mortal), camera angle "
    "(low-angle heroic, aerial, close-up, wide establishing shot), "
    "foreground/midground/background layering, action direction and "
    "dynamic lines of movement, focal point placement, symmetry vs "
    "asymmetry, depth staging.\n\n"
    "For style_description (150 words):\n"
    "Artistic style, color palette (golden, fiery reds, cool blues, "
    "earthy greens, jewel tones), lighting type (divine glow, dramatic "
    "backlight, firelight, moonlight, sunset, war-scene fire), "
    "texture (detailed ornamental, painterly, photorealistic), "
    "atmosphere (misty, smoky, radiant, stormy, celestial), rendering "
    "approach and cultural art conventions (Indian miniature, digital "
    "painting, cinematic VFX).\n\n"
    "For each character detected: use character_id as the character's "
    "proper name in lowercase (e.g. 'hanuman', 'lord_rama', 'sita'). "
    "Describe their specific costume, weapons held, body form/avatar, "
    "emotional state, and exact pose/action in 2-3 sentences each for "
    "semantic, emotion, and pose fields.\n\n"
    "For metadata:\n"
    "- scene_type: specific genre (e.g. 'Mythological battle', "
    "'Divine coronation', 'Forest exile', 'Aerial flight', "
    "'Bridge construction', 'Palace confrontation')\n"
    "- tags: 10-15 lowercase keywords including character names, "
    "episode name, weapons/props, location, action, and cultural "
    "tradition (e.g. ['hanuman', 'sanjeevani', 'dronagiri', 'mountain', "
    "'flight', 'sky', 'giant_form', 'gada', 'determination', "
    "'ramayana', 'indian_mythology'])\n"
    "- primary_subject: the main named character or element\n"
    "- narrative_theme: the mythological or thematic significance\n\n"
    "For narrative: identify story_summary (reference the specific "
    "mythological episode), narrative_type "
    "(cinematic/documentary/surreal/fantasy/mythological/devotional), "
    "and tone.\n\n"
    "For emotion: identify starting_emotion, mid_emotion, end_emotion, "
    "and emotional_polarity (-1.0 to 1.0).\n\n"
    "For objects: identify key_objects (weapons, artifacts, divine items "
    "by name with specifics — 'golden gada/mace', 'Kodanda bow with "
    "arrow drawn', 'burning tail torch', 'Dronagiri mountain'), "
    "contextual_objects, and symbolic_elements.\n\n"
    "For low_light: score brightness, contrast, noise_estimate, "
    "shadow_dominance, and visibility_confidence (all 0.0 to 1.0)."
)


def extract_image(image_input: ImageInput) -> ImageExtraction:
    """Run unified GPT-4o vision extraction on an image."""
    config = load_config()
    client = get_openai_client()

    response = client.beta.chat.completions.parse(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {  # type: ignore[misc, list-item]
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Generation prompt: {image_input.generation_prompt}"
                            "\n\nAnalyze this image comprehensively."
                        ),
                    },
                    image_input.to_openai_image_content(),
                ],
            },
        ],
        response_format=ImageExtraction,
        temperature=config.extraction.temperature,
        max_tokens=config.extraction.max_tokens,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        msg = "GPT-4o returned no parsed extraction"
        raise ValueError(msg)
    return parsed

"""Synthetic metadata generation via GPT-4o."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.models import ImageMetadata

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput

METADATA_SYSTEM_PROMPT = (
    "You are an expert cinematic metadata analyst specializing in "
    "mythology, religion, folklore, cultural art, and visual "
    "storytelling. Think like a movie director cataloguing production "
    "stills. Given an image and its generation prompt, extract "
    "structured metadata.\n\n"
    "CRITICAL: Focus on details that DISTINGUISH this scene from "
    "other scenes with the same character. Two images of Hanuman "
    "must produce very different metadata if one shows him flying "
    "with a mountain and the other shows him fighting in Lanka.\n\n"
    "Field-level guidance:\n"
    "- scene_type: Specific cinematic genre (e.g. 'Mythological battle', "
    "'Divine coronation', 'Forest exile', 'Aerial flight', "
    "'Bridge construction', 'Palace confrontation', 'Abduction scene', "
    "'Devotional meditation'). Avoid vague labels like 'Fantasy'.\n"
    "- time_of_day: dawn/morning/midday/afternoon/dusk/night/twilight/"
    "timeless.\n"
    "- lighting_condition: Cinematic lighting (e.g. 'divine golden "
    "backlighting', 'war-fire glow', 'moonlit forest', 'sunset over "
    "ocean', 'torch-lit palace interior').\n"
    "- primary_subject: The main named character (use PROPER NAMES: "
    "'Hanuman', 'Lord Rama', 'Sita', 'Ravana', not generic terms).\n"
    "- secondary_subjects: Other named characters or notable elements.\n"
    "- character_action: The PRIMARY ACTION being performed — use a "
    "specific verb (flying, fighting, kneeling, blessing, lifting, "
    "burning, talking, meditating, mourning, charging, building, "
    "carrying, shooting, running, sitting). This is critical for "
    "distinguishing different scenes of the same character.\n"
    "- weapons_props: List ALL weapons and props visible — be specific: "
    "'golden gada/mace', 'Kodanda bow', 'flaming arrows', 'Dronagiri "
    "mountain', 'burning tail-torch', 'Pushpaka Vimana chariot', "
    "'Sudarshana Chakra', 'trident/trishul', 'lotus', 'crown/mukut', "
    "'Shiva\'s bow (Pinaka)'. Include armor, shields, chariots.\n"
    "- location_name: Named location from the story — 'Lanka city', "
    "'Ayodhya palace', 'Kishkindha caves', 'Ashoka Vatika garden', "
    "'Dandaka forest', 'Panchavati hermitage', 'Ram Setu bridge', "
    "'ocean shore', 'sky/clouds', 'throne room', 'battlefield'. "
    "Be as specific as possible.\n"
    "- episode_name: Named story episode — 'Lanka Dahan', 'Sanjeevani "
    "quest', 'Setu Bandhan', 'Sita Haran', 'Swayamvar', 'Agni "
    "Pariksha', 'Bali Vadh', 'Ravana Vadh', 'Sundara Kanda', "
    "'Ashoka Vatika visit'. Use the canonical episode name.\n"
    "- artistic_style: Rendering approach (e.g. 'cinematic VFX', "
    "'digital painting', 'photorealistic mythological art', "
    "'traditional Indian miniature', 'concept art').\n"
    "- color_palette: 4-6 dominant colors as lowercase names.\n"
    "- tags: 10-15 lowercase keywords for discoverability. MUST "
    "include: character names, episode name, action verb, weapons, "
    "location, costume details, emotional state, and cultural "
    "tradition. Example: ['hanuman', 'sanjeevani', 'dronagiri', "
    "'mountain', 'flying', 'giant_form', 'determination', "
    "'sky', 'night', 'gada', 'dhoti', 'ramayana', "
    "'indian_mythology', 'devotion'].\n"
    "- narrative_theme: The mythological or thematic significance "
    "(e.g. 'devotion and selfless service', 'triumph of dharma "
    "over adharma', 'sacrifice for duty')."
)


def generate_metadata(image_input: ImageInput) -> ImageMetadata:
    """Generate synthetic metadata for an image using GPT-4o."""
    config = load_config()
    client = get_openai_client()

    response = client.beta.chat.completions.parse(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {  # type: ignore[misc, list-item]
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generation prompt: {image_input.generation_prompt}\n\nExtract metadata.",
                    },
                    image_input.to_openai_image_content(),
                ],
            },
        ],
        response_format=ImageMetadata,
        temperature=config.extraction.temperature,
        max_tokens=1500,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        msg = "GPT-4o returned no parsed metadata"
        raise ValueError(msg)
    return parsed

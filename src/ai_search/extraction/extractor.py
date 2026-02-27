"""Unified GPT-4o vision extraction for all pipeline dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.models import ImageExtraction

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput

EXTRACTION_SYSTEM_PROMPT = (
    "You are an image analysis system. Given an image and the prompt "
    "that generated it, extract comprehensive structured information.\n\n"
    "For semantic_description: Write a rich 200-word description covering "
    "scene content, subjects, actions, environment, mood, and thematic "
    "elements.\n\n"
    "For structural_description: Write a 150-word analysis focusing "
    "exclusively on spatial composition, layout, object positioning, "
    "foreground/midground/background, lines of composition, and "
    "geometric structure.\n\n"
    "For style_description: Write a 150-word analysis focusing "
    "exclusively on artistic style, color palette, lighting, texture, "
    "rendering technique, and visual treatment.\n\n"
    "For each character detected: provide semantic (identity/appearance), "
    "emotion (expression/body language), and pose (position/orientation) "
    "descriptions of 2-3 sentences each.\n\n"
    "For metadata: extract scene_type, time_of_day, lighting_condition, "
    "primary_subject, secondary_subjects, artistic_style, color_palette, "
    "tags, and narrative_theme.\n\n"
    "For narrative: identify story_summary, narrative_type "
    "(cinematic/documentary/surreal/fantasy/etc.), and tone.\n\n"
    "For emotion: identify starting_emotion, mid_emotion, end_emotion, "
    "and emotional_polarity (-1.0 to 1.0).\n\n"
    "For objects: identify key_objects, contextual_objects, and "
    "symbolic_elements.\n\n"
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

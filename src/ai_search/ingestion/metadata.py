"""Synthetic metadata generation via GPT-4o."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.models import ImageMetadata

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput

METADATA_SYSTEM_PROMPT = (
    "You are a metadata extraction assistant. Given an image and its "
    "generation prompt, extract structured metadata. Be specific and accurate."
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
        max_tokens=1000,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        msg = "GPT-4o returned no parsed metadata"
        raise ValueError(msg)
    return parsed

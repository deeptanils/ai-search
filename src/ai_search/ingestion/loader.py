"""Image loading utilities for URL and binary inputs."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ImageInput(BaseModel):
    """Validated image input."""

    image_id: str
    generation_prompt: str
    image_url: str | None = None
    image_base64: str | None = None

    @classmethod
    def from_url(cls, image_id: str, prompt: str, url: str) -> ImageInput:
        """Create an ImageInput from a URL."""
        return cls(image_id=image_id, generation_prompt=prompt, image_url=url)

    @classmethod
    def from_file(cls, image_id: str, prompt: str, path: str | Path) -> ImageInput:
        """Create an ImageInput from a local file path."""
        data = Path(path).read_bytes()
        b64 = base64.standard_b64encode(data).decode("utf-8")
        return cls(image_id=image_id, generation_prompt=prompt, image_base64=b64)

    def to_openai_image_content(self) -> dict[str, Any]:
        """Create OpenAI image_url content part."""
        if self.image_url:
            return {"type": "image_url", "image_url": {"url": self.image_url, "detail": "high"}}
        if self.image_base64:
            url = f"data:image/jpeg;base64,{self.image_base64}"
            return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
        msg = "Either image_url or image_base64 must be set"
        raise ValueError(msg)

"""Gradio UI for AI Search — text-to-image and image-to-image search.

Launch with::

    python -m ai_search.ui.app
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from typing import Any

import gradio as gr
import httpx
import structlog
from PIL import Image

from ai_search.models import SearchMode, SearchResult
from ai_search.retrieval.pipeline import search

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RESULTS = 10


async def _download_image(url: str) -> Image.Image | None:
    """Download an image from a URL and return as a PIL Image.

    Uses authenticated Azure Blob Storage client for blob URLs,
    falls back to unauthenticated HTTP for other URLs.
    """
    if not url:
        return None
    try:
        if ".blob.core.windows.net/" in url:
            return await _download_blob_image(url)
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
    except Exception:
        logger.warning("Failed to download image", url=url)
        return None


async def _download_blob_image(url: str) -> Image.Image | None:
    """Download an image from Azure Blob Storage using Entra ID auth."""
    import asyncio

    from azure.storage.blob import BlobClient

    from ai_search.clients import _get_credential

    def _sync_download() -> bytes:
        blob_client = BlobClient.from_blob_url(url, credential=_get_credential())
        return blob_client.download_blob().readall()

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _sync_download)
    return Image.open(io.BytesIO(data))


def _pil_to_tempfile(img: Image.Image | None) -> str | None:
    """Write a PIL image to a temp file and return the path."""
    if img is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(tmp, format="JPEG", quality=85)
    tmp.close()
    return tmp.name


def _build_gallery_items(
    results: list[SearchResult],
    images: list[Image.Image | None],
) -> list[tuple[str, str]]:
    """Build Gradio gallery items: list of (filepath, caption)."""
    items: list[tuple[str, str]] = []
    for result, img in zip(results, images):
        caption = (
            f"Score: {result.search_score:.4f} | "
            f"{result.image_id} | "
            f"{result.scene_type or '\u2014'}"
        )
        path = _pil_to_tempfile(img)
        if path:
            items.append((path, caption))
    return items


def _build_details_html(results: list[SearchResult]) -> str:
    """Build an HTML table with detailed result information."""
    rows = []
    for i, result in enumerate(results, 1):
        bar_width = int(result.search_score * 100)
        tags = ", ".join(result.tags[:5])
        prompt = (result.generation_prompt or "")[:120]
        rows.append(
            f"<tr>"
            f"<td style='text-align:center;font-weight:bold'>{i}</td>"
            f"<td>{result.image_id}</td>"
            f"<td>"
            f"  <div style='display:flex;align-items:center;gap:8px'>"
            f"    <div style='background:linear-gradient(90deg,#4f46e5,#818cf8);height:14px;"
            f"width:{bar_width}%;border-radius:4px;min-width:2px'></div>"
            f"    <span style='white-space:nowrap'>{result.search_score:.4f}</span>"
            f"  </div>"
            f"</td>"
            f"<td>{result.scene_type or '\u2014'}</td>"
            f"<td style='font-size:0.85em'>{tags}</td>"
            f"<td style='font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis'>{prompt}</td>"
            f"</tr>"
        )
    table = (
        "<table style='width:100%;border-collapse:collapse;margin-top:8px'>"
        "<thead><tr style='border-bottom:2px solid #555'>"
        "<th>#</th><th>Image ID</th><th>Score (0-1)</th>"
        "<th>Scene</th><th>Tags</th><th>Prompt</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return table


async def _render_results(
    results: list[SearchResult],
) -> tuple[list[tuple[str, str]], str]:
    """Download images and build gallery + details HTML from search results."""
    if not results:
        return [], "<p>No results found.</p>"

    urls = [result.image_url or "" for result in results]
    images = list(await asyncio.gather(*[_download_image(u) for u in urls]))

    gallery = _build_gallery_items(results, images)
    details = _build_details_html(results)
    return gallery, details


# ---------------------------------------------------------------------------
# Search handlers — thin wrappers around the unified pipeline
# ---------------------------------------------------------------------------


async def _text_search(query: str, top_k: int) -> tuple[list[tuple[str, str]], str]:
    """Run text-to-image hybrid search and return gallery items + details."""
    if not query.strip():
        return [], "<p>Please enter a search query.</p>"

    top_k = min(max(int(top_k), 1), _MAX_RESULTS)

    results = await search(
        mode=SearchMode.TEXT,
        query_text=query,
        top=top_k,
    )
    return await _render_results(results)


async def _image_search(
    image: Image.Image | None,
    top_k: int,
) -> tuple[list[tuple[str, str]], str]:
    """Run image-to-image search and return gallery items + details."""
    if image is None:
        return [], "<p>Please upload an image.</p>"

    top_k = min(max(int(top_k), 1), _MAX_RESULTS)

    # Convert PIL image to JPEG bytes
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=85)

    results = await search(
        mode=SearchMode.IMAGE,
        query_image_bytes=buf.getvalue(),
        top=top_k,
    )
    return await _render_results(results)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks application."""
    with gr.Blocks(
        title="AI Image Search",
    ) as app:
        gr.Markdown(
            "# 🔍 AI Image Search\n"
            "Search your indexed images using **text-to-image** or **image-to-image** hybrid search.\n"
            "Scores are normalized 0-1 (1 = most similar)."
        )

        with gr.Tabs():
            # ---- Tab 1: Text-to-Image ----
            with gr.Tab("📝 Text → Image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        txt_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., ocean waves crashing on rocks",
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        txt_top_k = gr.Slider(
                            minimum=1,
                            maximum=_MAX_RESULTS,
                            value=5,
                            step=1,
                            label="Results",
                        )
                txt_btn = gr.Button("🔎 Search", variant="primary")
                txt_gallery = gr.Gallery(
                    label="Results",
                    columns=3,
                    height="auto",
                    object_fit="cover",
                    elem_classes=["result-gallery"],
                )
                txt_details = gr.HTML(label="Details")

                txt_btn.click(
                    fn=_text_search,
                    inputs=[txt_query, txt_top_k],
                    outputs=[txt_gallery, txt_details],
                )
                txt_query.submit(
                    fn=_text_search,
                    inputs=[txt_query, txt_top_k],
                    outputs=[txt_gallery, txt_details],
                )

            # ---- Tab 2: Image-to-Image ----
            with gr.Tab("🖼️ Image → Image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_input = gr.Image(
                            label="Upload Query Image",
                            type="pil",
                            height=300,
                        )
                    with gr.Column(scale=1):
                        img_top_k = gr.Slider(
                            minimum=1,
                            maximum=_MAX_RESULTS,
                            value=5,
                            step=1,
                            label="Results",
                        )
                img_btn = gr.Button("🔎 Find Similar", variant="primary")
                img_gallery = gr.Gallery(
                    label="Similar Images",
                    columns=3,
                    height="auto",
                    object_fit="cover",
                    elem_classes=["result-gallery"],
                )
                img_details = gr.HTML(label="Details")

                img_btn.click(
                    fn=_image_search,
                    inputs=[img_input, img_top_k],
                    outputs=[img_gallery, img_details],
                )

    return app


def main() -> None:
    """Launch the Gradio UI."""
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
            .result-gallery { min-height: 300px; }
            table td, table th { padding: 6px 10px; border-bottom: 1px solid #ddd; }
        """,
    )


if __name__ == "__main__":
    main()

"""Batch ingestion pipeline — load, extract, embed, upload to blob, and index.

This module contains the reusable core of the ingestion workflow.  The
``scripts/ingest_samples.py`` CLI is a thin wrapper around the functions
exposed here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

from ai_search.embeddings.pipeline import generate_all_vectors
from ai_search.extraction.extractor import extract_image
from ai_search.indexing.indexer import build_search_document, upload_documents
from ai_search.ingestion.loader import ImageInput
from ai_search.models import SearchDocument
from ai_search.storage.blob import ensure_container_exists, upload_images_batch

logger = structlog.get_logger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────
INTER_IMAGE_DELAY_S = 1
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30
MAX_CONCURRENT = 5
UPLOAD_BATCH_SIZE = 10


# ── Result dataclass ─────────────────────────────────────────────────
@dataclass
class IngestionResult:
    """Summary returned after a batch ingestion run."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failed_ids: list[str] = field(default_factory=list)
    blob_urls: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ── Image loading ────────────────────────────────────────────────────
def load_image_inputs(
    data: list[dict],
    project_root: Path,
) -> tuple[list[ImageInput], dict[str, str]]:
    """Parse sample-image JSON entries into ``ImageInput`` objects.

    Returns:
        A tuple of (inputs, local_paths) where *local_paths* maps
        image IDs to their on-disk relative paths.
    """
    local_paths: dict[str, str] = {}
    inputs: list[ImageInput] = []

    for item in data:
        iid = item["image_id"]
        if item.get("image_path"):
            local_paths[iid] = item["image_path"]
            resolved = project_root / item["image_path"]
            if resolved.exists():
                b64 = base64.standard_b64encode(resolved.read_bytes()).decode("utf-8")
                inputs.append(
                    ImageInput(
                        image_id=iid,
                        generation_prompt=item["generation_prompt"],
                        image_base64=b64,
                    )
                )
                continue
        inputs.append(
            ImageInput(
                image_id=iid,
                generation_prompt=item["generation_prompt"],
                image_url=item.get("image_url"),
            )
        )

    return inputs, local_paths


def load_image_bytes(
    inputs: list[ImageInput],
    local_paths: dict[str, str],
    project_root: Path,
) -> dict[str, bytes]:
    """Load raw image bytes from local files or by downloading URLs.

    Local file paths take priority over URLs.  Paths are resolved
    relative to *project_root*.
    """
    loaded: dict[str, bytes] = {}
    for inp in inputs:
        local_path = local_paths.get(inp.image_id)
        if local_path:
            resolved = project_root / local_path
            if resolved.exists():
                loaded[inp.image_id] = resolved.read_bytes()
                logger.info(
                    "Loaded from file",
                    image_id=inp.image_id,
                    path=str(resolved),
                    size_bytes=len(loaded[inp.image_id]),
                )
                continue
            logger.warning(
                "Local file not found, falling back to URL",
                image_id=inp.image_id,
                path=str(resolved),
            )
        if inp.image_url:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                logger.info("Downloading image", image_id=inp.image_id)
                resp = client.get(inp.image_url)
                resp.raise_for_status()
                loaded[inp.image_id] = resp.content
                logger.info(
                    "Downloaded",
                    image_id=inp.image_id,
                    size_bytes=len(resp.content),
                )
    return loaded


# ── Single-image pipeline ────────────────────────────────────────────
async def process_image(
    image_input: ImageInput,
    image_bytes: bytes | None = None,
    *,
    max_retries: int = MAX_RETRIES,
    retry_backoff: float = RETRY_BACKOFF_S,
) -> SearchDocument:
    """Run extraction → embedding → document build for one image.

    Extraction is offloaded to a thread executor because the underlying
    GPT-4o call is synchronous.  Embedding retries on HTTP 429.
    """
    iid = image_input.image_id
    t0 = time.perf_counter()
    print(f"  [{iid}] Starting...")

    # Step 1: Extraction (sync → thread)
    loop = asyncio.get_running_loop()
    extraction = await loop.run_in_executor(None, extract_image, image_input)
    t1 = time.perf_counter()
    print(f"  [{iid}] Extraction done ({t1 - t0:.1f}s)")

    # Step 2: Embeddings (with retry)
    for attempt in range(1, max_retries + 1):
        try:
            vectors = await generate_all_vectors(extraction, image_bytes=image_bytes)
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries:
                wait = retry_backoff * attempt
                logger.warning(
                    "Rate limited, retrying",
                    image_id=iid,
                    attempt=attempt,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
            else:
                raise
    t2 = time.perf_counter()
    print(f"  [{iid}] Embeddings done ({t2 - t1:.1f}s)")

    # Validate embedding dimensions
    expected = {"semantic": 3072, "structural": 1024, "style": 512, "image": 1024}
    actual = {
        "semantic": len(vectors.semantic_vector),
        "structural": len(vectors.structural_vector),
        "style": len(vectors.style_vector),
        "image": len(vectors.image_vector),
    }
    for name, exp_dim in expected.items():
        if actual[name] != exp_dim:
            raise ValueError(
                f"Dimension mismatch for {iid}: {name}={actual[name]} expected {exp_dim}"
            )

    # Step 3: Build document
    doc = build_search_document(image_input, extraction, vectors)
    elapsed = time.perf_counter() - t0
    print(f"  [{iid}] Complete ({elapsed:.1f}s total)")
    return doc


# ── Blob upload helper ───────────────────────────────────────────────
def upload_to_blob(image_data: dict[str, bytes]) -> dict[str, str]:
    """Upload images to Azure Blob Storage if configured.

    Returns:
        Mapping of image_id → blob URL for uploaded images.  Empty
        dict when blob storage is not configured.
    """
    if not ensure_container_exists():
        logger.info("Blob storage not configured, skipping cloud upload")
        return {}

    print("Uploading images to Azure Blob Storage...")
    blob_urls = upload_images_batch(image_data)
    if blob_urls:
        print(f"  {len(blob_urls)} images uploaded to blob storage")
    return blob_urls


# ── Skip-detection ───────────────────────────────────────────────────
def get_already_indexed(inputs: list[ImageInput]) -> set[str]:
    """Return image IDs that already exist in the search index."""
    from ai_search.clients import get_search_client

    indexed: set[str] = set()
    client = get_search_client()
    for inp in inputs:
        try:
            client.get_document(key=inp.image_id)
            indexed.add(inp.image_id)
        except Exception:  # noqa: BLE001
            pass
    return indexed


# ── Batch index upload ───────────────────────────────────────────────
def index_documents(
    docs: list[SearchDocument],
    batch_size: int = UPLOAD_BATCH_SIZE,
) -> tuple[int, list[str]]:
    """Upload a list of search documents to the index in batches.

    Returns:
        (upload_success_count, list_of_failed_image_ids)
    """
    failed: list[str] = []
    success = 0
    print(f"\nUploading {len(docs)} documents in batches of {batch_size}...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        try:
            uploaded = upload_documents(batch)
            if uploaded:
                success += len(batch)
                print(f"  Batch {i // batch_size + 1}: {len(batch)} docs uploaded")
            else:
                logger.error("Batch upload returned False", batch_start=i)
                failed.extend(d.image_id for d in batch)
        except Exception as exc:
            logger.error("Batch upload failed", batch_start=i, error=str(exc))
            failed.extend(d.image_id for d in batch)
    return success, failed


# ── Full orchestrator ────────────────────────────────────────────────
async def ingest(
    data: list[dict],
    project_root: Path,
    *,
    force: bool = False,
    concurrency: int = MAX_CONCURRENT,
    batch_size: int = UPLOAD_BATCH_SIZE,
) -> IngestionResult:
    """Run the full ingest-and-index pipeline for a list of image entries.

    Steps:
        1. Parse inputs and load image bytes from disk / URL.
        2. Upload images to Azure Blob Storage (when configured).
        3. Skip already-indexed images (unless *force* is True).
        4. Extract metadata + generate embeddings concurrently.
        5. Attach blob URLs to the completed documents.
        6. Batch-upload documents to the search index.

    Args:
        data: Raw entries from ``sample_images.json``.
        project_root: Workspace root used to resolve relative image paths.
        force: Re-index every image even if already present.
        concurrency: Max images processed in parallel.
        batch_size: Documents per index-upload batch.

    Returns:
        An ``IngestionResult`` summarising the run.
    """
    t_start = time.perf_counter()

    # 1. Parse inputs
    inputs, local_paths = load_image_inputs(data, project_root)
    logger.info("Loaded sample images", count=len(inputs))

    # 2. Load raw bytes
    image_data = load_image_bytes(inputs, local_paths, project_root)

    # 3. Blob upload
    blob_urls = upload_to_blob(image_data)

    # 4. Skip check
    already_indexed: set[str] = set()
    if not force:
        already_indexed = get_already_indexed(inputs)
        if already_indexed:
            logger.info("Skipping already indexed", ids=sorted(already_indexed))

    to_process = [inp for inp in inputs if inp.image_id not in already_indexed]
    if not to_process:
        print("All images already indexed. Use --force to re-index.")
        return IngestionResult(
            skipped=len(already_indexed),
            blob_urls=blob_urls,
            elapsed_seconds=time.perf_counter() - t_start,
        )

    # 5. Process concurrently
    print(f"\nProcessing {len(to_process)} images with concurrency={concurrency}...")
    sem = asyncio.Semaphore(concurrency)
    completed_docs: list[SearchDocument] = []
    failed_ids: list[str] = []
    lock = asyncio.Lock()

    async def _ingest_one(inp: ImageInput) -> None:
        async with sem:
            try:
                doc = await process_image(inp, image_bytes=image_data.get(inp.image_id))
                # Attach blob storage URL to the document metadata
                if inp.image_id in blob_urls:
                    doc.image_url = blob_urls[inp.image_id]
                async with lock:
                    completed_docs.append(doc)
            except Exception as exc:
                logger.error("Error processing", image_id=inp.image_id, error=str(exc))
                async with lock:
                    failed_ids.append(inp.image_id)
            finally:
                await asyncio.sleep(INTER_IMAGE_DELAY_S)

    await asyncio.gather(*[_ingest_one(inp) for inp in to_process])

    # 6. Index upload
    if completed_docs:
        _, upload_failures = index_documents(completed_docs, batch_size=batch_size)
        failed_ids.extend(upload_failures)

    elapsed = time.perf_counter() - t_start
    result = IngestionResult(
        processed=len(completed_docs),
        skipped=len(already_indexed),
        failed=len(failed_ids),
        failed_ids=failed_ids,
        blob_urls=blob_urls,
        elapsed_seconds=elapsed,
    )
    print(
        f"\nDone — {result.processed}/{len(to_process)} processed, "
        f"{result.skipped} skipped, {result.failed} failed "
        f"({result.elapsed_seconds:.0f}s total)"
    )
    if result.failed_ids:
        print(f"  Failed: {result.failed_ids}")
    return result

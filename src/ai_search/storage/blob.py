"""Upload images to Azure Blob Storage and return public URLs."""

from __future__ import annotations

import structlog
from azure.storage.blob import ContentSettings

from ai_search.clients import get_blob_container_client

logger = structlog.get_logger(__name__)


def upload_image(
    image_id: str,
    image_bytes: bytes,
    content_type: str = "image/png",
) -> str | None:
    """Upload a single image to Azure Blob Storage.

    Args:
        image_id: Unique identifier used as the blob name.
        image_bytes: Raw image data.
        content_type: MIME type for the blob (default: image/png).

    Returns:
        The public URL of the uploaded blob, or None if storage
        is not configured.
    """
    container = get_blob_container_client()
    if container is None:
        return None

    blob_name = f"{image_id}.png"
    blob_client = container.get_blob_client(blob_name)

    blob_client.upload_blob(
        image_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )

    url = blob_client.url
    logger.info("Uploaded to blob", image_id=image_id, url=url)
    return url


def upload_images_batch(
    images: dict[str, bytes],
    content_type: str = "image/png",
) -> dict[str, str]:
    """Upload multiple images to Azure Blob Storage.

    Args:
        images: Mapping of image_id to raw bytes.
        content_type: MIME type for the blobs.

    Returns:
        Mapping of image_id to blob URL for successfully uploaded images.
    """
    container = get_blob_container_client()
    if container is None:
        logger.warning("Blob storage not configured, skipping batch upload")
        return {}

    urls: dict[str, str] = {}
    for image_id, data in images.items():
        try:
            blob_name = f"{image_id}.png"
            blob_client = container.get_blob_client(blob_name)
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            urls[image_id] = blob_client.url
        except Exception as exc:
            logger.error("Blob upload failed", image_id=image_id, error=str(exc))

    logger.info("Batch blob upload complete", total=len(images), uploaded=len(urls))
    return urls


def ensure_container_exists() -> bool:
    """Create the blob container if it does not exist.

    Returns:
        True if the container exists or was created, False if
        storage is not configured.
    """
    container = get_blob_container_client()
    if container is None:
        return False

    try:
        container.create_container()
        logger.info("Container created", container=container.container_name)
    except Exception:
        # Container already exists — ignore the conflict error.
        pass
    return True

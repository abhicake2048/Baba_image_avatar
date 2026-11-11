from __future__ import annotations

import base64
import hashlib
import io
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .models import PreparedReference


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 50_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_image(path: Path, max_edge: int) -> bytes:
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"{path.name} exceeds the 20 MB source-image limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if getattr(source, "is_animated", False):
                    raise ValueError(f"{path.name} is animated; use a still image")
                if source.width * source.height > MAX_SOURCE_PIXELS:
                    raise ValueError(f"{path.name} exceeds the 50 megapixel limit")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                # Re-encoding removes EXIF and other source metadata.
                image.save(output, format="JPEG", quality=92, optimize=True)
                return output.getvalue()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{path.name} is not a valid supported image: {exc}") from exc


def load_reference_library(
    directory: Path,
    *,
    min_count: int = 4,
    max_count: int = 10,
    max_edge: int = 1536,
) -> list[PreparedReference]:
    if not directory.is_dir():
        raise ValueError(f"Reference directory not found: {directory}")
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not min_count <= len(files) <= max_count:
        raise ValueError(
            f"Reference directory must contain {min_count}-{max_count} supported images; found {len(files)}"
        )

    references: list[PreparedReference] = []
    for path in files:
        jpeg_bytes = _prepare_image(path, max_edge)
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        references.append(
            PreparedReference(
                name=path.name,
                sha256=_sha256_file(path),
                data_url=f"data:image/jpeg;base64,{encoded}",
            )
        )
    return references


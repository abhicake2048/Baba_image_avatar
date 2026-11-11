from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .image_service import ImagePayload
from .models import GeneratedImage


@dataclass(frozen=True)
class HistoryItem:
    request_dir: Path
    thumbnail: Path
    caption: str


class StorageManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_request_dir(self) -> tuple[str, Path]:
        now = datetime.now(timezone.utc)
        request_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:10]}"
        request_dir = self.output_dir / request_id
        request_dir.mkdir(mode=0o700)
        return request_id, request_dir

    def save_image(
        self,
        request_dir: Path,
        index: int,
        payload: ImagePayload,
        *,
        view_key: str | None = None,
        view_label: str | None = None,
        generation_prompt: str | None = None,
    ) -> GeneratedImage:
        safe_view = re.sub(r"[^a-z0-9-]+", "-", (view_key or "").lower()).strip("-")
        suffix = f"-{safe_view}" if safe_view else ""
        target = self._checked_child(request_dir) / f"image-{index:02d}{suffix}{payload.extension}"
        with target.open("wb") as handle:
            handle.write(payload.content)
        return GeneratedImage(
            path=target,
            media_type=payload.media_type,
            sha256=payload.sha256,
            usage=payload.usage,
            view_key=view_key,
            view_label=view_label,
            generation_prompt=generation_prompt,
        )

    def write_metadata(self, request_dir: Path, metadata: dict[str, Any]) -> Path:
        directory = self._checked_child(request_dir)
        target = directory / "metadata.json"
        content = json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix="metadata-", suffix=".tmp", delete=False
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(target)
        return target

    def create_zip(self, request_dir: Path, images: list[GeneratedImage]) -> Path | None:
        if not images:
            return None
        directory = self._checked_child(request_dir)
        target = directory / "download.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for image in images:
                archive.write(image.path, arcname=image.path.name)
            metadata = directory / "metadata.json"
            if metadata.is_file():
                archive.write(metadata, arcname=metadata.name)
        return target

    def list_history(self, limit: int = 50) -> list[HistoryItem]:
        items: list[HistoryItem] = []
        directories = sorted(
            (path for path in self.output_dir.iterdir() if path.is_dir()),
            key=lambda value: value.name,
            reverse=True,
        )
        for directory in directories:
            metadata_path = directory / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            images = sorted(directory.glob("image-*.*"))
            if not images:
                continue
            model = str(metadata.get("model", "unknown model")).split("/")[-1]
            prompt = str(metadata.get("original_prompt", "")).strip()
            caption = f"{directory.name} · {model} · {prompt[:80]}"
            items.append(HistoryItem(directory, images[0], caption))
            if len(items) >= limit:
                break
        return items

    def delete_request(self, request_dir: Path | str) -> None:
        target = Path(request_dir).resolve()
        self._checked_child(target)
        if not target.is_dir():
            raise ValueError("The selected history item no longer exists")
        shutil.rmtree(target)

    def delete_all(self) -> int:
        count = 0
        for child in list(self.output_dir.iterdir()):
            if child.is_dir():
                self.delete_request(child)
                count += 1
        return count

    def _checked_child(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved.parent != self.output_dir:
            raise ValueError("Refusing to access a path outside the configured output directory")
        return resolved


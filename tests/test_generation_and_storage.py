from __future__ import annotations

from pathlib import Path

import pytest

from identity_generator.generation import GenerationService
from identity_generator.image_service import ImagePayload
from identity_generator.logging_config import configure_logging
from identity_generator.models import EnhancedPrompt, GenerationRequest, PosePromptSet, ViewPrompt
from identity_generator.storage import StorageManager


class FakePromptService:
    async def enhance(self, prompt, profile):
        return EnhancedPrompt(enhanced_prompt=f"Detailed photorealistic request: {prompt}")

    async def build_pose_prompts(self, prompt, profile):
        return PosePromptSet(
            base_prompt=prompt,
            front_headshot="Detailed front-facing waist-up portrait of the authorized subject.",
            three_quarter_left="Detailed waist-up left three-quarter portrait of the authorized subject.",
            three_quarter_right="Detailed waist-up right three-quarter portrait of the authorized subject.",
            full_body_front="Detailed front-facing full-body standing portrait of the authorized subject.",
            full_body_seated="Detailed front-facing full-body seated portrait of the authorized subject.",
        )


class PartiallyFailingImageService:
    def __init__(self, png_bytes: bytes):
        self.png_bytes = png_bytes
        self.calls = 0

    async def generate_one(self, request, enhanced, references):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated provider failure")
        return ImagePayload(
            content=self.png_bytes,
            extension=".png",
            media_type="image/png",
            sha256=f"image-{self.calls}",
            usage={"cost": 0.1},
        )


@pytest.mark.asyncio
async def test_four_image_batch_retains_partial_success(
    output_dir: Path, profile, prepared_references, png_bytes: bytes
) -> None:
    log_path = configure_logging(output_dir.parent / "logs")
    storage = StorageManager(output_dir)
    image_service = PartiallyFailingImageService(png_bytes)
    service = GenerationService(
        prompt_service=FakePromptService(),
        image_service=image_service,
        storage=storage,
        profile=profile,
        references=prepared_references,
    )
    result = await service.generate(
        GenerationRequest(
            prompt="seated in a library wearing a red shirt",
            model="google/gemini-3-pro-image",
            count=4,
        )
    )
    assert len(result.images) == 3
    assert len(result.failures) == 1
    assert result.zip_path and result.zip_path.is_file()
    metadata = (result.request_dir / "metadata.json").read_text(encoding="utf-8")
    assert "simulated provider failure" in metadata
    assert "data:image" not in metadata
    log_text = log_path.read_text(encoding="utf-8")
    assert "generation.request" in log_text
    assert '"user_prompt": "seated in a library wearing a red shirt"' in log_text
    assert '"model": "google/gemini-3-pro-image"' in log_text
    assert '"image_count": 4' in log_text
    assert "generation.enhanced_prompt" in log_text
    assert "generation.complete" in log_text
    assert "outputs_shown" in log_text
    assert "data:image" not in log_text


def test_history_and_guarded_deletion(output_dir: Path, png_bytes: bytes) -> None:
    storage = StorageManager(output_dir)
    _, request_dir = storage.create_request_dir()
    payload = ImagePayload(png_bytes, ".png", "image/png", "hash", None)
    image = storage.save_image(request_dir, 1, payload)
    storage.write_metadata(
        request_dir,
        {"model": "test/model", "original_prompt": "portrait", "images": [image.path.name]},
    )
    assert len(storage.list_history()) == 1
    with pytest.raises(ValueError, match="outside"):
        storage.delete_request(output_dir.parent)
    storage.delete_request(request_dir)
    assert not request_dir.exists()


@pytest.mark.asyncio
async def test_fixed_pose_generation_saves_one_labeled_image_per_view(
    output_dir: Path, profile, prepared_references, png_bytes: bytes
) -> None:
    storage = StorageManager(output_dir)
    image_service = PartiallyFailingImageService(png_bytes)
    service = GenerationService(
        prompt_service=FakePromptService(),
        image_service=image_service,
        storage=storage,
        profile=profile,
        references=prepared_references,
    )
    prompt_set = await service.prepare_pose_prompts("wearing red robes in a garden")
    selected = [prompt_set.views()[0], prompt_set.views()[2], prompt_set.views()[4]]
    result = await service.generate_pose_views(
        base_prompt=prompt_set.base_prompt,
        model="google/gemini-3-pro-image",
        aspect_ratio="3:4",
        quality_preset="Standard",
        views=selected,
    )

    assert [image.view_key for image in result.images] == ["front-headshot", "full-body-seated"]
    assert result.failures[0].startswith("Right three-quarter view:")
    assert result.images[0].path.name.endswith("front-headshot.png")
    assert result.images[1].path.name.endswith("full-body-seated.png")
    metadata = (result.request_dir / "metadata.json").read_text(encoding="utf-8")
    assert '"key": "three-quarter-right"' in metadata
    assert '"generation_prompt"' in metadata
    assert '"count_requested": 3' in metadata


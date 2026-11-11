from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from .image_service import ImageService
from .logging_config import get_logger
from .models import (
    EnhancedPrompt,
    GeneratedImage,
    GenerationRequest,
    GenerationResult,
    PosePromptSet,
    PreparedReference,
    SubjectProfile,
    ViewPrompt,
)
from .prompt_service import PromptService
from .safety import validate_user_prompt
from .storage import StorageManager


class GenerationService:
    def __init__(
        self,
        *,
        prompt_service: PromptService,
        image_service: ImageService,
        storage: StorageManager,
        profile: SubjectProfile,
        references: list[PreparedReference],
        batch_concurrency: int = 2,
    ) -> None:
        self.prompt_service = prompt_service
        self.image_service = image_service
        self.storage = storage
        self.profile = profile
        self.references = references
        self.batch_concurrency = batch_concurrency

    async def prepare_pose_prompts(self, prompt: str) -> PosePromptSet:
        cleaned = validate_user_prompt(prompt)
        logger = get_logger()
        logger.info("prompt_set.request %s", json.dumps({"user_prompt": cleaned}, ensure_ascii=False))
        try:
            prompt_set = await self.prompt_service.build_pose_prompts(cleaned, self.profile)
        except Exception as exc:
            logger.exception("prompt_set.failed error=%s", exc)
            raise
        logger.info(
            "prompt_set.ready %s",
            json.dumps(
                {
                    "user_prompt": cleaned,
                    "fallback_used": prompt_set.used_fallback,
                    "view_prompts": [view.model_dump() for view in prompt_set.views()],
                },
                ensure_ascii=False,
            ),
        )
        return prompt_set

    async def generate_pose_views(
        self,
        *,
        base_prompt: str,
        model: str,
        aspect_ratio: str,
        quality_preset: str,
        views: list[ViewPrompt],
    ) -> GenerationResult:
        cleaned_base = validate_user_prompt(base_prompt)
        if not views:
            raise ValueError("Select at least one view to generate.")
        cleaned_views = [
            view.model_copy(update={"prompt": validate_user_prompt(view.prompt)}) for view in views
        ]
        request = GenerationRequest(
            prompt=cleaned_base,
            model=model,
            count=len(cleaned_views),
            aspect_ratio=aspect_ratio,
            quality_preset=quality_preset,
        )
        request_id, request_dir = self.storage.create_request_dir()
        logger = get_logger()
        logger.info(
            "pose_generation.request %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "user_prompt": cleaned_base,
                    "model": model,
                    "image_count": len(cleaned_views),
                    "aspect_ratio": aspect_ratio,
                    "quality_preset": quality_preset,
                    "view_prompts": [view.model_dump() for view in cleaned_views],
                },
                ensure_ascii=False,
            ),
        )
        semaphore = asyncio.Semaphore(self.batch_concurrency)

        async def one(index: int, view: ViewPrompt) -> tuple[int, GeneratedImage | None, str | None]:
            async with semaphore:
                try:
                    enhanced = EnhancedPrompt(enhanced_prompt=view.prompt)
                    payload = await self.image_service.generate_one(request, enhanced, self.references)
                    image = self.storage.save_image(
                        request_dir,
                        index,
                        payload,
                        view_key=view.key,
                        view_label=view.label,
                        generation_prompt=view.prompt,
                    )
                    logger.info(
                        "pose_generation.image_ready %s",
                        json.dumps(
                            {
                                "request_id": request_id,
                                "view_key": view.key,
                                "view_label": view.label,
                                "generation_prompt": view.prompt,
                                "output_path": str(image.path),
                                "usage": image.usage,
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    )
                    return index, image, None
                except Exception as exc:
                    logger.exception(
                        "pose_generation.image_failed request_id=%s view_key=%s error=%s",
                        request_id,
                        view.key,
                        exc,
                    )
                    return index, None, f"{view.label}: {exc}"

        outcomes = await asyncio.gather(
            *(one(index, view) for index, view in enumerate(cleaned_views, start=1))
        )
        outcomes.sort(key=lambda item: item[0])
        images = [image for _, image, _ in outcomes if image is not None]
        failures = [failure for _, _, failure in outcomes if failure is not None]
        combined_prompt = "\n\n".join(f"{view.label}: {view.prompt}" for view in cleaned_views)
        metadata = self._metadata(
            request_id,
            request,
            EnhancedPrompt(enhanced_prompt=combined_prompt),
            images,
            failures,
            stage="completed",
            view_prompts=cleaned_views,
        )
        self.storage.write_metadata(request_dir, metadata)
        zip_path = self.storage.create_zip(request_dir, images)
        logger.info(
            "pose_generation.complete %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "model": model,
                    "requested_views": [view.key for view in cleaned_views],
                    "outputs_shown": [str(image.path) for image in images],
                    "failures": failures,
                },
                ensure_ascii=False,
            ),
        )
        return GenerationResult(
            request_id=request_id,
            request_dir=request_dir,
            images=images,
            enhanced_prompt=combined_prompt,
            failures=failures,
            zip_path=zip_path,
            view_prompts=cleaned_views,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = validate_user_prompt(request.prompt)
        request = request.model_copy(update={"prompt": prompt})
        request_id, request_dir = self.storage.create_request_dir()
        logger = get_logger()
        logger.info(
            "generation.request %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "user_prompt": request.prompt,
                    "model": request.model,
                    "image_count": request.count,
                    "aspect_ratio": request.aspect_ratio,
                    "quality_preset": request.quality_preset,
                },
                ensure_ascii=False,
            ),
        )

        try:
            enhanced = await self.prompt_service.enhance(prompt, self.profile)
        except Exception as exc:
            logger.exception(
                "generation.prompt_enhancement_failed request_id=%s error=%s", request_id, exc
            )
            self.storage.write_metadata(
                request_dir,
                self._metadata(request_id, request, None, [], [str(exc)], stage="prompt_enhancement"),
            )
            raise

        logger.info(
            "generation.enhanced_prompt %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "enhanced_prompt": enhanced.enhanced_prompt,
                    "fallback_used": enhanced.used_fallback,
                },
                ensure_ascii=False,
            ),
        )

        semaphore = asyncio.Semaphore(self.batch_concurrency)

        async def one(index: int) -> tuple[int, GeneratedImage | None, str | None]:
            async with semaphore:
                try:
                    payload = await self.image_service.generate_one(request, enhanced, self.references)
                    image = self.storage.save_image(request_dir, index, payload)
                    logger.info(
                        "generation.image_ready %s",
                        json.dumps(
                            {
                                "request_id": request_id,
                                "image_index": index,
                                "output_path": str(image.path),
                                "media_type": image.media_type,
                                "usage": image.usage,
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    )
                    return index, image, None
                except Exception as exc:
                    logger.exception(
                        "generation.image_failed request_id=%s image_index=%s error=%s",
                        request_id,
                        index,
                        exc,
                    )
                    return index, None, f"Image {index}: {exc}"

        outcomes = await asyncio.gather(*(one(index) for index in range(1, request.count + 1)))
        outcomes.sort(key=lambda item: item[0])
        images = [image for _, image, _ in outcomes if image is not None]
        failures = [failure for _, _, failure in outcomes if failure is not None]
        metadata = self._metadata(request_id, request, enhanced, images, failures, stage="completed")
        self.storage.write_metadata(request_dir, metadata)
        zip_path = self.storage.create_zip(request_dir, images)
        logger.info(
            "generation.complete %s",
            json.dumps(
                {
                    "request_id": request_id,
                    "model": request.model,
                    "requested": request.count,
                    "generated": len(images),
                    "outputs_shown": [str(image.path) for image in images],
                    "failures": failures,
                },
                ensure_ascii=False,
            ),
        )
        return GenerationResult(
            request_id=request_id,
            request_dir=request_dir,
            images=images,
            enhanced_prompt=enhanced.enhanced_prompt,
            failures=failures,
            zip_path=zip_path,
        )

    def _metadata(
        self,
        request_id: str,
        request: GenerationRequest,
        enhanced: EnhancedPrompt | None,
        images: list[GeneratedImage],
        failures: list[str],
        *,
        stage: str,
        view_prompts: list[ViewPrompt] | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "original_prompt": request.prompt,
            "enhanced_prompt": enhanced.enhanced_prompt if enhanced else None,
            "prompt_fallback_used": enhanced.used_fallback if enhanced else None,
            "preservation_constraints": enhanced.preservation_constraints if enhanced else [],
            "view_prompts": [view.model_dump() for view in (view_prompts or [])],
            "model": request.model,
            "count_requested": request.count,
            "aspect_ratio": request.aspect_ratio,
            "quality_preset": request.quality_preset,
            "reference_files": [
                {"name": reference.name, "sha256": reference.sha256} for reference in self.references
            ],
            "images": [
                {
                    "filename": image.path.name,
                    "media_type": image.media_type,
                    "sha256": image.sha256,
                    "usage": image.usage,
                    "view_key": image.view_key,
                    "view_label": image.view_label,
                    "generation_prompt": image.generation_prompt,
                }
                for image in images
            ],
            "failures": failures,
        }


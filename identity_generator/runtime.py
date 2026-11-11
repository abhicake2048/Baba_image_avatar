from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, load_profile
from .generation import GenerationService
from .image_service import ImageService
from .openrouter import OpenRouterClient, catalog_supports_references
from .prompt_service import PromptService
from .references import load_reference_library
from .storage import StorageManager


@dataclass
class AppRuntime:
    settings: Settings
    generation: GenerationService
    storage: StorageManager
    available_models: dict[str, str]
    prompt_client: OpenRouterClient
    image_client: OpenRouterClient

    async def close(self) -> None:
        await self.prompt_client.close()
        await self.image_client.close()


async def build_runtime(settings: Settings) -> AppRuntime:
    profile = load_profile(settings.profile_path)
    references = load_reference_library(
        settings.reference_dir,
        min_count=settings.min_references,
        max_count=settings.max_references,
        max_edge=settings.max_reference_edge,
    )
    api_key = settings.openrouter_api_key.get_secret_value()
    prompt_client = OpenRouterClient(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        timeout=settings.prompt_timeout_seconds,
    )
    image_client = OpenRouterClient(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        timeout=settings.image_timeout_seconds,
    )

    try:
        catalog = await image_client.get("images/models")
        available_models = {
            label: model_id
            for label, model_id in settings.image_models.items()
            if catalog_supports_references(catalog, model_id)
        }
        if not available_models:
            raise ValueError(
                "Neither configured image model currently advertises input_references in the OpenRouter catalog."
            )

        storage = StorageManager(settings.output_dir)
        prompt_service = PromptService(prompt_client, settings.prompt_model)
        image_service = ImageService(
            image_client,
            nano_banana_model=settings.nano_banana_model,
            openai_image_model=settings.openai_image_model,
        )
        generation = GenerationService(
            prompt_service=prompt_service,
            image_service=image_service,
            storage=storage,
            profile=profile,
            references=references,
        )
        return AppRuntime(
            settings=settings,
            generation=generation,
            storage=storage,
            available_models=available_models,
            prompt_client=prompt_client,
            image_client=image_client,
        )
    except Exception:
        await prompt_client.close()
        await image_client.close()
        raise


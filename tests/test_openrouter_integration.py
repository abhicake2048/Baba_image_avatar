from __future__ import annotations

import json

import httpx
import pytest

from identity_generator.image_service import ImageService
from identity_generator.models import EnhancedPrompt, GenerationRequest
from identity_generator.openrouter import OpenRouterClient, OpenRouterError, catalog_supports_references


NANO = "google/gemini-3-pro-image"
OPENAI = "openai/gpt-image-2.6-sunburst"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [NANO, OPENAI])
async def test_both_models_use_unified_image_request(
    model: str, image_b64: str, prepared_references
) -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"data": [{"b64_json": image_b64}], "usage": {"cost": 0.05}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenRouterClient(
            api_key="test-key", base_url="https://openrouter.test/api/v1", timeout=10, client=http_client
        )
        service = ImageService(client, nano_banana_model=NANO, openai_image_model=OPENAI)
        request = GenerationRequest(prompt="portrait", model=model, count=1)
        await service.generate_one(
            request,
            EnhancedPrompt(enhanced_prompt="A detailed photorealistic portrait of the authorized subject."),
            prepared_references,
        )

    body = captured[0]
    assert body["model"] == model
    assert body["n"] == 1
    assert len(body["input_references"]) == 4
    assert all(item["type"] == "image_url" for item in body["input_references"])
    assert "data:image" in body["input_references"][0]["image_url"]["url"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "API key"), (402, "balance"), (429, "rate-limited"), (500, "temporarily unavailable")],
)
async def test_actionable_http_errors(status: int, message: str) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"error": {"message": "provider"}}))
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenRouterClient(
            api_key="secret", base_url="https://openrouter.test/api/v1", timeout=10, client=http_client
        )
        with pytest.raises(OpenRouterError, match=message):
            await client.post("images", {"prompt": "safe"})


def test_catalog_requires_reference_parameter() -> None:
    catalog = {
        "data": [
            {"id": NANO, "supported_parameters": {"input_references": {"max": 14}}},
            {"id": "text-only", "supported_parameters": {}},
        ]
    }
    assert catalog_supports_references(catalog, NANO)
    assert not catalog_supports_references(catalog, "text-only")


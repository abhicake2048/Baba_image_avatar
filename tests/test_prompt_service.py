from __future__ import annotations

import json

import httpx
import pytest

from identity_generator.openrouter import OpenRouterClient
from identity_generator.prompt_service import PromptService


@pytest.mark.asyncio
async def test_prompt_service_uses_structured_output_and_no_images(profile) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        content = json.dumps(
            {
                "enhanced_prompt": "A hyper-realistic seated portrait with natural window light and a red shirt.",
                "preservation_constraints": ["Preserve identity."],
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenRouterClient(
            api_key="secret", base_url="https://openrouter.test/api/v1", timeout=10, client=http_client
        )
        service = PromptService(client, "openai/gpt-4.1-mini")
        result = await service.enhance("seated in a library wearing a red shirt", profile)

    body = requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert "data:image" not in json.dumps(body)
    assert "identity anchors" in result.enhanced_prompt
    assert not result.used_fallback


@pytest.mark.asyncio
async def test_prompt_service_falls_back_after_malformed_json(profile) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenRouterClient(
            api_key="secret", base_url="https://openrouter.test/api/v1", timeout=10, client=http_client
        )
        service = PromptService(client, "openai/gpt-4.1-mini", max_attempts=1)
        result = await service.enhance("portrait near a window", profile)
    assert result.used_fallback
    assert "portrait near a window" in result.enhanced_prompt


@pytest.mark.asyncio
async def test_prompt_service_builds_five_fixed_editable_views(profile) -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        content = json.dumps(
            {
                "front_headshot": "A front-facing waist-up portrait in a red robe with natural light.",
                "three_quarter_left": "A left three-quarter waist-up portrait in a red robe with natural light.",
                "three_quarter_right": "A right three-quarter waist-up portrait in a red robe with natural light.",
                "full_body_front": "A front-facing full-body standing portrait in a red robe with natural light.",
                "full_body_seated": "A front-facing full-body seated portrait in a red robe with natural light.",
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenRouterClient(
            api_key="secret", base_url="https://openrouter.test/api/v1", timeout=10, client=http_client
        )
        result = await PromptService(client, "openai/gpt-4.1-mini").build_pose_prompts(
            "wearing a red robe in natural window light", profile
        )

    assert [view.key for view in result.views()] == [
        "front-headshot",
        "three-quarter-left",
        "three-quarter-right",
        "full-body-front",
        "full-body-seated",
    ]
    assert all("identity anchors" in view.prompt for view in result.views())
    assert "Required framing: straight-on front-facing waist-up" in result.front_headshot
    assert "subject's right" in result.three_quarter_right
    assert requests[0]["response_format"]["json_schema"]["name"] == "fixed_pose_prompt_set"
    assert "data:image" not in json.dumps(requests[0])


@pytest.mark.asyncio
async def test_fixed_view_prompt_service_has_deterministic_fallback(profile) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
    )
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenRouterClient(
            api_key="secret", base_url="https://openrouter.test/api/v1", timeout=10, client=http_client
        )
        result = await PromptService(
            client, "openai/gpt-4.1-mini", max_attempts=1
        ).build_pose_prompts("standing near a window", profile)

    assert result.used_fallback
    assert len(result.views()) == 5
    assert "left three-quarter" in result.three_quarter_left


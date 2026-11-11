from __future__ import annotations

import base64

import pytest

from identity_generator.image_service import decode_image_item, generation_parameters
from identity_generator.models import GenerationRequest
from identity_generator.openrouter import OpenRouterError


NANO = "google/gemini-3-pro-image"
OPENAI = "openai/gpt-image-2.6-sunburst"


@pytest.mark.parametrize(
    ("model", "preset", "expected"),
    [
        (NANO, "Draft", {"resolution": "1K"}),
        (NANO, "Standard", {"resolution": "2K"}),
        (NANO, "High", {"resolution": "4K"}),
        (OPENAI, "Draft", {"quality": "low"}),
        (OPENAI, "Standard", {"quality": "high"}),
        (OPENAI, "High", {"quality": "max"}),
    ],
)
def test_provider_specific_presets(model: str, preset: str, expected: dict[str, str]) -> None:
    request = GenerationRequest(
        prompt="portrait in a studio",
        model=model,
        count=1,
        aspect_ratio="3:2",
        quality_preset=preset,
    )
    params = generation_parameters(request, nano_banana_model=NANO, openai_image_model=OPENAI)
    assert params["n"] == 1
    assert params["aspect_ratio"] == "3:2"
    assert expected.items() <= params.items()


def test_auto_aspect_ratio_is_omitted() -> None:
    request = GenerationRequest(prompt="studio portrait", model=NANO, count=1)
    params = generation_parameters(request, nano_banana_model=NANO, openai_image_model=OPENAI)
    assert "aspect_ratio" not in params


def test_decode_valid_image(image_b64: str) -> None:
    payload = decode_image_item({"b64_json": image_b64, "media_type": "image/png"}, {"cost": 0.1})
    assert payload.extension == ".png"
    assert payload.media_type == "image/png"
    assert payload.usage == {"cost": 0.1}


@pytest.mark.parametrize("value", ["not-base64", base64.b64encode(b"not an image").decode("ascii")])
def test_decode_rejects_malformed_images(value: str) -> None:
    with pytest.raises(OpenRouterError):
        decode_image_item({"b64_json": value})


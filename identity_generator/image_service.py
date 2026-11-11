from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

from .models import EnhancedPrompt, GenerationRequest, PreparedReference
from .openrouter import OpenRouterClient, OpenRouterError


@dataclass(frozen=True)
class ImagePayload:
    content: bytes
    extension: str
    media_type: str
    sha256: str
    usage: dict[str, Any] | None


def generation_parameters(
    request: GenerationRequest, *, nano_banana_model: str, openai_image_model: str
) -> dict[str, Any]:
    params: dict[str, Any] = {"n": 1}
    if request.aspect_ratio != "Auto":
        params["aspect_ratio"] = request.aspect_ratio
    if request.model == nano_banana_model:
        params["resolution"] = {"Draft": "1K", "Standard": "2K", "High": "4K"}[
            request.quality_preset
        ]
    elif request.model == openai_image_model:
        params["quality"] = {"Draft": "low", "Standard": "high", "High": "max"}[
            request.quality_preset
        ]
    else:
        raise ValueError("The selected model is not one of the configured image models")
    return params


def decode_image_item(item: dict[str, Any], usage: dict[str, Any] | None = None) -> ImagePayload:
    encoded = item.get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise OpenRouterError("The provider response did not contain base64 image data.")
    if encoded.startswith("data:"):
        try:
            encoded = encoded.split(",", 1)[1]
        except IndexError as exc:
            raise OpenRouterError("The provider returned a malformed image data URL.") from exc
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OpenRouterError("The provider returned malformed base64 image data.") from exc
    if not content:
        raise OpenRouterError("The provider returned an empty image.")

    try:
        with Image.open(io.BytesIO(content)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            if width < 64 or height < 64:
                raise OpenRouterError("The provider returned an image that is unexpectedly small.")
            if width * height > 50_000_000:
                raise OpenRouterError("The provider returned an image larger than the safety limit.")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise OpenRouterError("The provider returned bytes that are not a valid image.") from exc

    formats = {
        "PNG": (".png", "image/png"),
        "JPEG": (".jpg", "image/jpeg"),
        "WEBP": (".webp", "image/webp"),
        "GIF": (".gif", "image/gif"),
    }
    if image_format not in formats:
        raise OpenRouterError(f"The provider returned unsupported image format {image_format or 'unknown'}.")
    extension, detected_type = formats[image_format]
    declared_type = item.get("media_type")
    media_type = declared_type if isinstance(declared_type, str) and declared_type.startswith("image/") else detected_type
    return ImagePayload(
        content=content,
        extension=extension,
        media_type=media_type,
        sha256=hashlib.sha256(content).hexdigest(),
        usage=usage,
    )


class ImageService:
    def __init__(
        self,
        client: OpenRouterClient,
        *,
        nano_banana_model: str,
        openai_image_model: str,
    ) -> None:
        self.client = client
        self.nano_banana_model = nano_banana_model
        self.openai_image_model = openai_image_model

    async def generate_one(
        self,
        request: GenerationRequest,
        enhanced: EnhancedPrompt,
        references: list[PreparedReference],
    ) -> ImagePayload:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": enhanced.enhanced_prompt,
            "input_references": [
                {"type": "image_url", "image_url": {"url": reference.data_url}}
                for reference in references
            ],
            **generation_parameters(
                request,
                nano_banana_model=self.nano_banana_model,
                openai_image_model=self.openai_image_model,
            ),
        }
        response = await self.client.post("images", payload)
        data = response.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise OpenRouterError("The provider returned no generated images.")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        return decode_image_item(data[0], usage)


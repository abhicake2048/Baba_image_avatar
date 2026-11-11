from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


AspectRatio = Literal["Auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9"]
QualityPreset = Literal["Draft", "Standard", "High"]


class SubjectProfile(BaseModel):
    subject_label: str = Field(min_length=1, max_length=100)
    consent_confirmed: bool
    adult_confirmed: bool
    allowed_context_notes: list[str] = Field(default_factory=list)
    identity_instructions: list[str] = Field(default_factory=list)

    @field_validator("consent_confirmed")
    @classmethod
    def require_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError("consent_confirmed must be true")
        return value

    @field_validator("adult_confirmed")
    @classmethod
    def require_adult(cls, value: bool) -> bool:
        if not value:
            raise ValueError("adult_confirmed must be true")
        return value


class PreparedReference(BaseModel):
    name: str
    sha256: str
    data_url: str = Field(repr=False)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    model: str = Field(min_length=1)
    count: int = Field(ge=1, le=5)
    aspect_ratio: AspectRatio = "Auto"
    quality_preset: QualityPreset = "Standard"

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        return " ".join(value.strip().split())


class EnhancedPrompt(BaseModel):
    enhanced_prompt: str = Field(min_length=20, max_length=8000)
    preservation_constraints: list[str] = Field(default_factory=list)
    used_fallback: bool = False


class ViewPrompt(BaseModel):
    key: str = Field(pattern=r"^[a-z0-9-]+$")
    label: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=20, max_length=4000)


class PosePromptSet(BaseModel):
    base_prompt: str = Field(min_length=3, max_length=4000)
    front_headshot: str = Field(min_length=20, max_length=4000)
    three_quarter_left: str = Field(min_length=20, max_length=4000)
    three_quarter_right: str = Field(min_length=20, max_length=4000)
    full_body_front: str = Field(min_length=20, max_length=4000)
    full_body_seated: str = Field(min_length=20, max_length=4000)
    used_fallback: bool = False

    def views(self) -> list[ViewPrompt]:
        return [
            ViewPrompt(key="front-headshot", label="Front headshot / waist-up", prompt=self.front_headshot),
            ViewPrompt(
                key="three-quarter-left",
                label="Left three-quarter view",
                prompt=self.three_quarter_left,
            ),
            ViewPrompt(
                key="three-quarter-right",
                label="Right three-quarter view",
                prompt=self.three_quarter_right,
            ),
            ViewPrompt(key="full-body-front", label="Full-body front standing", prompt=self.full_body_front),
            ViewPrompt(key="full-body-seated", label="Full-body front seated", prompt=self.full_body_seated),
        ]


class GeneratedImage(BaseModel):
    path: Path
    media_type: str
    sha256: str
    usage: dict[str, Any] | None = None
    view_key: str | None = None
    view_label: str | None = None
    generation_prompt: str | None = None


class GenerationResult(BaseModel):
    request_id: str
    request_dir: Path
    images: list[GeneratedImage]
    enhanced_prompt: str
    failures: list[str] = Field(default_factory=list)
    zip_path: Path | None = None
    view_prompts: list[ViewPrompt] = Field(default_factory=list)


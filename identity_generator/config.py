from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import SubjectProfile


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    prompt_model: str = "openai/gpt-4.1-mini"
    nano_banana_model: str = "google/gemini-3-pro-image"
    openai_image_model: str = "openai/gpt-image-2.6-sunburst"
    profile_path: Path = Path("profile.yaml")
    reference_dir: Path = Path("references")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    image_timeout_seconds: float = Field(default=240, ge=30, le=900)
    prompt_timeout_seconds: float = Field(default=60, ge=10, le=180)
    max_reference_edge: int = Field(default=1536, ge=512, le=4096)
    min_references: int = Field(default=4, ge=1, le=10)
    max_references: int = Field(default=10, ge=4, le=14)
    app_title: str = "Private Portrait Studio"

    @field_validator("openrouter_base_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("openrouter_api_key")
    @classmethod
    def reject_placeholder_key(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if not raw or "replace-me" in raw.lower():
            raise ValueError("Set OPENROUTER_API_KEY to a real OpenRouter API key")
        return SecretStr(raw)

    def resolved(self, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        data = self.model_dump()
        for name in ("profile_path", "reference_dir", "output_dir", "log_dir"):
            value = Path(data[name])
            data[name] = value if value.is_absolute() else root / value
        return Settings.model_validate(data)

    @property
    def image_models(self) -> dict[str, str]:
        return {
            "Nano Banana Pro": self.nano_banana_model,
            "OpenAI GPT Image": self.openai_image_model,
        }


def load_settings(project_root: Path | None = None) -> Settings:
    return Settings().resolved(project_root)


def load_profile(path: Path) -> SubjectProfile:
    if not path.is_file():
        raise ValueError(
            f"Profile not found at {path}. Copy profile.example.yaml to profile.yaml and review it."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read profile {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile.yaml must contain a YAML object")
    try:
        return SubjectProfile.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid profile.yaml: {exc}") from exc


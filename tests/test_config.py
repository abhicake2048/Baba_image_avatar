from __future__ import annotations

from pathlib import Path

import pytest

from identity_generator.config import Settings, load_profile


def test_settings_resolve_relative_paths(tmp_path: Path) -> None:
    settings = Settings(openrouter_api_key="secret").resolved(tmp_path)
    assert settings.reference_dir == tmp_path / "references"
    assert settings.output_dir == tmp_path / "outputs"
    assert settings.log_dir == tmp_path / "logs"
    assert settings.openrouter_api_key.get_secret_value() == "secret"
    assert "secret" not in repr(settings.openrouter_api_key)


def test_profile_requires_consent_and_adult(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "subject_label: test\nconsent_confirmed: false\nadult_confirmed: true\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="consent_confirmed"):
        load_profile(profile_path)


def test_profile_loads_valid_yaml(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "subject_label: test\nconsent_confirmed: true\nadult_confirmed: true\n", encoding="utf-8"
    )
    profile = load_profile(profile_path)
    assert profile.subject_label == "test"


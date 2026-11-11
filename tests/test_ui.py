from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from identity_generator.config import Settings
from identity_generator.storage import StorageManager
from identity_generator.ui import build_app, carousel_view


def test_gradio_app_constructs_with_current_components(tmp_path) -> None:
    settings = Settings(
        openrouter_api_key="test-key",
        output_dir=tmp_path / "outputs",
        reference_dir=tmp_path / "references",
        profile_path=tmp_path / "profile.yaml",
    ).resolved(tmp_path)
    runtime = SimpleNamespace(
        settings=settings,
        storage=StorageManager(settings.output_dir),
        available_models=settings.image_models,
        generation=None,
    )
    app = build_app(runtime)
    assert isinstance(app, gr.Blocks)
    assert len(app.blocks) >= 30


def test_carousel_wraps_and_returns_current_download() -> None:
    paths = ["one.png", "two.png", "three.png"]
    assert carousel_view(paths, 0) == ("one.png", 0, "Image 1 of 3", "one.png")
    assert carousel_view(paths, 3) == ("one.png", 0, "Image 1 of 3", "one.png")
    assert carousel_view(paths, -1) == ("three.png", 2, "Image 3 of 3", "three.png")
    assert carousel_view([], 0) == (None, 0, "No images yet", None)


def test_carousel_discloses_fixed_view_label() -> None:
    path = "outputs/request/image-01-three-quarter-left.png"
    assert carousel_view([path], 0) == (
        path,
        0,
        "**Left three-quarter view** · Image 1 of 1",
        path,
    )


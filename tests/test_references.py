from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from identity_generator.references import load_reference_library


def _make_references(directory: Path, count: int) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for index in range(count):
        path = directory / f"person-{index}.png"
        Image.new("RGB", (2000, 1200), (index * 20, 80, 120)).save(path, format="PNG")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_reference_preprocessing_is_in_memory_and_strips_to_jpeg(tmp_path: Path) -> None:
    original_hashes = _make_references(tmp_path, 4)
    references = load_reference_library(tmp_path, max_edge=512)
    assert len(references) == 4
    assert all(reference.data_url.startswith("data:image/jpeg;base64,") for reference in references)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()} == original_hashes


def test_reference_count_is_validated(tmp_path: Path) -> None:
    _make_references(tmp_path, 3)
    with pytest.raises(ValueError, match="4-10"):
        load_reference_library(tmp_path)


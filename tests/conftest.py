from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from identity_generator.models import PreparedReference, SubjectProfile


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (128, 128), color: str = "#4b63b8") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=fmt)
    return output.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return image_bytes()


@pytest.fixture
def image_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


@pytest.fixture
def profile() -> SubjectProfile:
    return SubjectProfile(
        subject_label="the authorized subject",
        consent_confirmed=True,
        adult_confirmed=True,
        allowed_context_notes=["ordinary portrait scenes"],
        identity_instructions=[
            "Preserve identity from the references.",
            "Keep the subject as the only recognizable person.",
        ],
    )


@pytest.fixture
def prepared_references() -> list[PreparedReference]:
    return [
        PreparedReference(
            name=f"reference-{index}.jpg",
            sha256=f"hash-{index}",
            data_url=f"data:image/jpeg;base64,{base64.b64encode(b'reference').decode('ascii')}",
        )
        for index in range(4)
    ]


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "outputs"
    path.mkdir()
    return path


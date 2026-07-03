"""Benchmark model presets and AIVMX download helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

AIVISHUB_API_BASE_URL = "https://api.aivis-project.com/v1"
AIVMX_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ModelPreset:
    """A reproducible AivisHub AIVMX benchmark model."""

    key: str
    name: str
    aivishub_url: str
    model_uuid: str
    version: str
    style_id: int
    sha256: str


MODEL_PRESETS = {
    "mao": ModelPreset(
        key="mao",
        name="まお",
        aivishub_url=(
            "https://hub.aivis-project.com/aivm-models/"
            "a59cb814-0083-4369-8542-f51a29e72af7"
        ),
        model_uuid="a59cb814-0083-4369-8542-f51a29e72af7",
        version="1.2.0",
        style_id=888753760,
        sha256="f87ccea2e8e2de0e0bfe52e803945af903b4086bf25621a015111628f00e4119",
    ),
    "kohaku": ModelPreset(
        key="kohaku",
        name="コハク",
        aivishub_url=(
            "https://hub.aivis-project.com/aivm-models/"
            "22e8ed77-94fe-4ef2-871f-a86f94e9a579"
        ),
        model_uuid="22e8ed77-94fe-4ef2-871f-a86f94e9a579",
        version="1.1.0",
        style_id=1878365376,
        sha256="3f5c08b52bb8a64efd361268580c81510f96c927cd6905aa7dbae6851333270a",
    ),
}


def aivmx_download_url(model: ModelPreset) -> str:
    """Return the AivisHub API download URL for an AIVMX preset."""

    return (
        f"{AIVISHUB_API_BASE_URL}/aivm-models/"
        f"{model.model_uuid}/download?model_type=AIVMX"
    )


def default_aivmx_download_path(model: ModelPreset, *, root_dir: str | Path) -> Path:
    """Return the stable local artifact path for a downloaded preset."""

    return (
        Path(root_dir)
        / "benchmark-artifacts"
        / "models"
        / f"{model.key}-{model.version}.aivmx"
    )


def download_aivmx(model: ModelPreset, destination: str | Path) -> Path:
    """Download and SHA-256 verify an AIVMX preset."""

    destination_path = Path(destination).expanduser().resolve()
    if destination_path.is_file() and file_sha256(destination_path) == model.sha256:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f"{destination_path.name}.download")
    request = Request(
        aivmx_download_url(model),
        headers={
            "User-Agent": "onnxruntime-ep-style-bert-vits2-ggml benchmark reproduction"
        },
    )
    with urlopen(request, timeout=60) as response, temporary_path.open("wb") as file:
        while True:
            chunk = response.read(AIVMX_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            file.write(chunk)

    actual_sha256 = file_sha256(temporary_path)
    if actual_sha256 != model.sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded AIVMX SHA-256 does not match the preset.\n"
            f"expected: {model.sha256}\n"
            f"actual:   {actual_sha256}\n"
            f"AivisHub: {model.aivishub_url}"
        )

    temporary_path.replace(destination_path)
    return destination_path


def file_sha256(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

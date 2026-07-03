"""Tests for benchmark model preset helpers."""

from pathlib import Path

from onnxruntime_ep_style_bert_vits2_ggml.models import (
    MODEL_PRESETS,
    aivmx_download_url,
    default_aivmx_download_path,
)


def test_aivmx_download_url_uses_aivishub_api_download_endpoint() -> None:
    model = MODEL_PRESETS["mao"]

    assert aivmx_download_url(model) == (
        "https://api.aivis-project.com/v1/aivm-models/"
        "a59cb814-0083-4369-8542-f51a29e72af7/download?model_type=AIVMX"
    )


def test_default_aivmx_download_path_is_stable_artifact_path() -> None:
    model = MODEL_PRESETS["mao"]

    assert default_aivmx_download_path(model, root_dir=Path("repo")) == (
        Path("repo") / "benchmark-artifacts" / "models" / "mao-1.2.0.aivmx"
    )

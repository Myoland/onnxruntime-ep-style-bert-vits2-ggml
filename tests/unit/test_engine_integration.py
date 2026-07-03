"""Tests for downstream Engine integration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from onnxruntime_ep_style_bert_vits2_ggml.engine_integration import (
    StyleBertVits2GgmlRuntime,
    build_engine_execution_provider_config,
    resolve_aivmx_onnx_source_path,
    validate_session_provider,
)
from onnxruntime_ep_style_bert_vits2_ggml.runtime import PluginExecutionProviderConfig


def test_build_engine_execution_provider_config_sets_runtime_defaults(
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "libstyle_bert_vits2_ggml_onnx_ep.so"
    tts_cpp_library_path = Path("lib/libtts.so")

    config = build_engine_execution_provider_config(
        base_options={},
        backend="vulkan",
        precision="accurate",
        tts_cpp_library_path=tts_cpp_library_path,
        vulkan_device="0",
        path_base_dir=tmp_path,
        library_path=library_path,
        provider_name="StyleBertVits2GgmlExecutionProvider",
        cpu_count=8,
    )

    assert config.provider_name == "StyleBertVits2GgmlExecutionProvider"
    assert config.library_path == library_path
    assert config.strict is True
    assert config.provider_options == {
        "backend": "vulkan",
        "claim_jp_bert_graph": "1",
        "claim_synthesis_graph": "1",
        "device": "0",
        "eager_load_model": "1",
        "n_threads": "0",
        "precision": "accurate",
        "tts_cpp_library_path": str((tmp_path / tts_cpp_library_path).resolve()),
    }


def test_prepare_jp_bert_providers_fills_cache_path(tmp_path: Path) -> None:
    jp_bert_onnx_path = tmp_path / "model_fp16.onnx"
    jp_bert_gguf_path = tmp_path / "jp-bert.gguf"

    class FakeJpBertGgufCache:
        def ensure(self, *, onnx_path: Path) -> Any:
            assert onnx_path == jp_bert_onnx_path
            return SimpleNamespace(gguf_path=jp_bert_gguf_path)

    config = PluginExecutionProviderConfig(
        provider_name="StyleBertVits2GgmlExecutionProvider",
        provider_options={
            "backend": "vulkan",
            "claim_jp_bert_graph": "1",
            "claim_synthesis_graph": "1",
        },
        strict=True,
    )
    runtime = StyleBertVits2GgmlRuntime(
        config=config,
        jp_bert_cache=cast(Any, FakeJpBertGgufCache()),
    )

    providers = runtime.prepare_jp_bert_providers(
        providers=[
            (
                "StyleBertVits2GgmlExecutionProvider",
                dict(config.provider_options),
            ),
            "CPUExecutionProvider",
        ],
        resolve_jp_bert_onnx_path=lambda: jp_bert_onnx_path,
    )

    assert config.provider_options["jp_bert_gguf_path"] == str(jp_bert_gguf_path)
    assert providers[0] == (
        "StyleBertVits2GgmlExecutionProvider",
        {
            "backend": "vulkan",
            "claim_jp_bert_graph": "1",
            "claim_synthesis_graph": "0",
            "jp_bert_gguf_path": str(jp_bert_gguf_path),
        },
    )


def test_prepare_synthesis_providers_fills_model_gguf_path(tmp_path: Path) -> None:
    gguf_path = tmp_path / "model.gguf"

    class FakeAivmGgufCache:
        def ensure(self, *, aivm_file_path: Path, aivm_metadata: Any) -> Any:
            assert aivm_file_path == tmp_path / "model.aivmx"
            assert aivm_metadata is not None
            return SimpleNamespace(gguf_path=gguf_path)

    config = PluginExecutionProviderConfig(
        provider_name="StyleBertVits2GgmlExecutionProvider",
        provider_options={
            "backend": "vulkan",
            "claim_jp_bert_graph": "0",
            "claim_synthesis_graph": "1",
        },
        strict=True,
    )
    runtime = StyleBertVits2GgmlRuntime(
        config=config,
        synthesis_cache=cast(Any, FakeAivmGgufCache()),
    )

    providers = runtime.prepare_synthesis_providers(
        providers=[
            (
                "StyleBertVits2GgmlExecutionProvider",
                dict(config.provider_options),
            ),
            "CPUExecutionProvider",
        ],
        onnx_source_path=tmp_path / "model.aivmx",
        aivm_metadata=object(),
        resolve_jp_bert_onnx_path=lambda: tmp_path / "model_fp16.onnx",
    )

    assert providers[0] == (
        "StyleBertVits2GgmlExecutionProvider",
        {
            "backend": "vulkan",
            "claim_jp_bert_graph": "0",
            "claim_synthesis_graph": "1",
            "gguf_path": str(gguf_path),
        },
    )


def test_validate_session_provider_rejects_silent_cpu_fallback() -> None:
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="StyleBertVits2GgmlExecutionProvider"):
        validate_session_provider(
            session=session,
            required_provider_name="StyleBertVits2GgmlExecutionProvider",
            context="test session",
        )


def test_resolve_aivmx_onnx_source_path_prefers_same_uuid_aivmx(
    tmp_path: Path,
) -> None:
    installed_file_path = tmp_path / "model.aivm"
    aivmx_source_path = tmp_path / "model-uuid.aivmx"
    installed_file_path.write_bytes(b"aivm")
    aivmx_source_path.write_bytes(b"aivmx")

    assert (
        resolve_aivmx_onnx_source_path(
            installed_file_path=installed_file_path,
            aivm_model_uuid="model-uuid",
        )
        == aivmx_source_path
    )

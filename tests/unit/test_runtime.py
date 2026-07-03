"""Tests for downstream runtime integration helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from onnxruntime_ep_style_bert_vits2_ggml.runtime import (
    PluginExecutionProviderConfig,
    build_provider_options,
    configure_execution_provider,
    default_backend_for_platform,
    inference_session_scope,
)


def test_default_backend_for_platform_uses_packaged_accelerator() -> None:
    assert default_backend_for_platform("darwin") == "metal"
    assert default_backend_for_platform("linux") == "vulkan"
    assert default_backend_for_platform("win32") == "vulkan"


def test_build_provider_options_defaults_to_claiming_supported_graphs(
    tmp_path: Path,
) -> None:
    library_path = Path("lib/libtts.so")

    provider_options = build_provider_options(
        base_options={},
        backend="vulkan",
        precision="accurate",
        tts_cpp_library_path=library_path,
        vulkan_device="0",
        path_base_dir=tmp_path,
    )

    assert provider_options == {
        "backend": "vulkan",
        "claim_jp_bert_graph": "1",
        "claim_synthesis_graph": "1",
        "device": "0",
        "eager_load_model": "1",
        "n_threads": "0",
        "precision": "accurate",
        "tts_cpp_library_path": str((tmp_path / library_path).resolve()),
    }


def test_build_provider_options_sets_positive_threads_for_cpu_backend() -> None:
    provider_options = build_provider_options(
        base_options={},
        backend="cpu",
        precision="accurate",
        tts_cpp_library_path="/opt/tts.cpp/libtts.so",
        vulkan_device="0",
        cpu_count=8,
    )

    assert provider_options["backend"] == "cpu"
    assert provider_options["n_threads"] == "8"
    assert "device" not in provider_options


def test_build_provider_options_requires_tts_cpp_library() -> None:
    with pytest.raises(RuntimeError, match="tts_cpp_library_path"):
        build_provider_options(
            base_options={},
            backend="vulkan",
            precision="accurate",
            tts_cpp_library_path=None,
        )


def test_configure_execution_provider_registers_and_prepends_plugin(
    tmp_path: Path,
) -> None:
    register_calls: list[tuple[str, str]] = []
    ort = SimpleNamespace(
        register_execution_provider_library=lambda registration_name, library_path: (
            register_calls.append((registration_name, library_path))
        ),
        get_available_providers=lambda: ["CPUExecutionProvider"],
        get_ep_devices=lambda: [
            SimpleNamespace(ep_name="StyleBertVits2GgmlExecutionProvider")
        ],
    )

    providers = configure_execution_provider(
        base_providers=[
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
        config=PluginExecutionProviderConfig(
            provider_name="StyleBertVits2GgmlExecutionProvider",
            provider_options={"backend": "vulkan", "device": "0"},
            library_path=tmp_path / "libstyle_bert_vits2_ggml_ep.so",
            registration_name="style-bert-vits2-ggml",
            strict=True,
        ),
        ort_module=ort,
    )

    assert register_calls == [
        (
            "style-bert-vits2-ggml",
            str(tmp_path / "libstyle_bert_vits2_ggml_ep.so"),
        )
    ]
    assert providers[0] == (
        "StyleBertVits2GgmlExecutionProvider",
        {"backend": "vulkan", "device": "0"},
    )
    assert providers[1:] == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]


def test_configure_execution_provider_raises_when_strict(tmp_path: Path) -> None:
    ort = SimpleNamespace(
        register_execution_provider_library=lambda _registration_name, _library_path: (
            None
        ),
        get_available_providers=lambda: ["CPUExecutionProvider"],
        get_ep_devices=lambda: [],
    )

    with pytest.raises(RuntimeError) as exc_info:
        configure_execution_provider(
            base_providers=["CPUExecutionProvider"],
            config=PluginExecutionProviderConfig(
                provider_name="StyleBertVits2GgmlExecutionProvider",
                provider_options={},
                library_path=tmp_path / "libstyle_bert_vits2_ggml_ep.so",
                strict=True,
            ),
            ort_module=ort,
        )

    assert "StyleBertVits2GgmlExecutionProvider" in str(exc_info.value)
    assert "Available providers" in str(exc_info.value)


def test_inference_session_scope_uses_ep_devices() -> None:
    calls: list[tuple[str, Any]] = []

    class FakeSessionOptions:
        def add_provider_for_devices(
            self,
            ep_devices: list[Any],
            provider_options: dict[str, str],
        ) -> None:
            calls.append(("add_provider_for_devices", (ep_devices, provider_options)))

    def fake_inference_session(
        path_or_bytes: str,
        *,
        sess_options: Any | None = None,
        providers: list[str | tuple[str, dict[str, Any]]] | None = None,
        provider_options: list[dict[Any, Any]] | None = None,
        **_kwargs: Any,
    ) -> str:
        calls.append(
            (
                "InferenceSession",
                {
                    "path_or_bytes": path_or_bytes,
                    "sess_options": sess_options,
                    "providers": providers,
                    "provider_options": provider_options,
                },
            )
        )
        return "session"

    ep_device = SimpleNamespace(ep_name="StyleBertVits2GgmlExecutionProvider")
    ort = SimpleNamespace(
        get_ep_devices=lambda: [ep_device],
        SessionOptions=FakeSessionOptions,
        InferenceSession=fake_inference_session,
    )
    config = PluginExecutionProviderConfig(
        provider_name="StyleBertVits2GgmlExecutionProvider",
        provider_options={"backend": "vulkan"},
    )
    providers = [
        ("StyleBertVits2GgmlExecutionProvider", {"backend": "cpu", "n_threads": "4"}),
        "CPUExecutionProvider",
    ]

    with inference_session_scope(config, ort_module=ort):
        session = ort.InferenceSession("model.onnx", providers=providers)

    assert session == "session"
    assert calls[0] == (
        "add_provider_for_devices",
        ([ep_device], {"backend": "cpu", "n_threads": "4"}),
    )
    assert calls[1][0] == "InferenceSession"
    assert calls[1][1]["providers"] is None

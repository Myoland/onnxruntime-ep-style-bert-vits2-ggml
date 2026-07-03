"""Downstream Engine integration helpers for the Style-Bert-VITS2 GGML EP."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from onnxruntime_ep_style_bert_vits2_ggml import get_ep_name
from onnxruntime_ep_style_bert_vits2_ggml.engine_cache import (
    DEFAULT_GGUF_CONVERTER_VERSION,
    AivmGgufCache,
    JpBertGgufCache,
)
from onnxruntime_ep_style_bert_vits2_ggml.runtime import (
    DEFAULT_REGISTRATION_NAME,
    PluginExecutionProviderConfig,
    ProviderSetting,
    build_execution_provider_config,
    build_provider_options,
    configure_execution_provider,
    inference_session_scope,
    provider_names,
    remove_provider,
    replace_provider_options,
    resolve_library_path,
)

JpBertOnnxPathResolver = Callable[[], Path]
GraphKind = Literal["jp_bert", "synthesis"]


def resolve_aivmx_onnx_source_path(
    *,
    installed_file_path: Path,
    aivm_model_uuid: str,
) -> Path:
    """Prefer same-UUID AIVMX/ONNX files as ONNX Runtime sources."""

    if installed_file_path.suffix == ".aivmx":
        return installed_file_path
    aivmx_source_path = installed_file_path.with_name(f"{aivm_model_uuid}.aivmx")
    if aivmx_source_path.exists() and aivmx_source_path.is_file():
        return aivmx_source_path
    return installed_file_path


def build_engine_execution_provider_config(
    *,
    base_options: dict[str, str] | None = None,
    backend: str | None = None,
    precision: str = "accurate",
    tts_cpp_library_path: str | Path | None = None,
    vulkan_device: str | None = None,
    path_base_dir: str | Path | None = None,
    library_path: str | Path | None = None,
    registration_name: str = DEFAULT_REGISTRATION_NAME,
    provider_name: str | None = None,
    strict: bool = True,
    cpu_count: int | None = None,
) -> PluginExecutionProviderConfig:
    """Build the strict Plugin EP config expected by downstream Engine startup."""

    try:
        resolved_library_path = resolve_library_path(
            library_path,
            base_dir=path_base_dir,
        )
    except Exception as ex:
        raise RuntimeError(
            "--onnx_provider ggml requires --onnx_ep_library_path or a packaged "
            "onnxruntime_ep_style_bert_vits2_ggml library."
        ) from ex

    try:
        provider_options = build_provider_options(
            base_options=base_options,
            backend=backend,
            precision=precision,
            tts_cpp_library_path=tts_cpp_library_path,
            vulkan_device=vulkan_device,
            path_base_dir=path_base_dir,
            cpu_count=cpu_count if cpu_count is not None else os.cpu_count(),
        )
    except RuntimeError as ex:
        raise RuntimeError(
            "--onnx_provider ggml requires --ggml_native_library_path or "
            "--onnx_ep_option tts_cpp_library_path=<path-to-libtts>."
        ) from ex

    return build_execution_provider_config(
        provider_options=provider_options,
        library_path=resolved_library_path,
        provider_name=provider_name or get_ep_name(),
        registration_name=registration_name,
        strict=strict,
    )


@dataclass
class StyleBertVits2GgmlRuntime:
    """Runtime state needed to attach the GGML Plugin EP to one Engine process."""

    config: PluginExecutionProviderConfig | None
    synthesis_cache: AivmGgufCache | None = None
    jp_bert_cache: JpBertGgufCache | None = None

    @classmethod
    def create(
        cls,
        *,
        config: PluginExecutionProviderConfig | None,
        cache_dir: Path | None = None,
        default_cache_dir: Path | None = None,
        synthesis_converter_version: str | None = None,
    ) -> StyleBertVits2GgmlRuntime:
        """Create caches only when the Plugin EP is enabled."""

        if config is None:
            return cls(config=None)

        resolved_cache_dir = cache_dir if cache_dir is not None else default_cache_dir
        return cls(
            config=config,
            synthesis_cache=AivmGgufCache(
                cache_dir=resolved_cache_dir,
                converter_version=(
                    synthesis_converter_version or DEFAULT_GGUF_CONVERTER_VERSION
                ),
            ),
            jp_bert_cache=JpBertGgufCache(cache_dir=resolved_cache_dir),
        )

    def strict_provider_name(self, graph: GraphKind) -> str | None:
        """Return the provider that must own a graph in strict mode."""

        if self.config is None or not self.config.strict:
            return None
        option_name = (
            "claim_jp_bert_graph" if graph == "jp_bert" else "claim_synthesis_graph"
        )
        if self.config.provider_options.get(option_name) != "1":
            return None
        return self.config.provider_name

    def configure_providers(
        self,
        *,
        base_providers: Sequence[ProviderSetting],
        ort_module: Any | None = None,
        logger: Any | None = None,
    ) -> list[ProviderSetting]:
        """Register the Plugin EP and prepend it to the ordinary provider chain."""

        return configure_execution_provider(
            base_providers=base_providers,
            config=self.config,
            ort_module=ort_module,
            logger=logger,
        )

    def prepare_jp_bert_providers(
        self,
        *,
        providers: Sequence[ProviderSetting],
        resolve_jp_bert_onnx_path: JpBertOnnxPathResolver,
    ) -> list[ProviderSetting]:
        """Prepare BERT-only provider options before the global BERT session opens."""

        if self.config is None:
            return list(providers)
        if self.config.provider_name not in provider_names(providers):
            return list(providers)
        if self.config.provider_options.get("claim_jp_bert_graph") != "1":
            return remove_provider(
                providers=providers,
                provider_name=self.config.provider_name,
            )

        return replace_provider_options(
            providers=providers,
            provider_name=self.config.provider_name,
            provider_options=self._jp_bert_provider_options(
                resolve_jp_bert_onnx_path=resolve_jp_bert_onnx_path,
            ),
        )

    def prepare_synthesis_providers(
        self,
        *,
        providers: Sequence[ProviderSetting],
        onnx_source_path: Path,
        aivm_metadata: Any,
        resolve_jp_bert_onnx_path: JpBertOnnxPathResolver,
    ) -> list[ProviderSetting]:
        """Prepare model-specific provider options before synthesis session load."""

        if self.config is None:
            return list(providers)

        provider_options = dict(self.config.provider_options)
        if provider_options.get(
            "claim_synthesis_graph"
        ) == "1" and not provider_options.get("gguf_path"):
            if self.synthesis_cache is None:
                raise RuntimeError(
                    "ONNX Plugin EP requires a GGUF cache to prepare synthesis artifacts."
                )
            gguf_entry = self.synthesis_cache.ensure(
                aivm_file_path=onnx_source_path,
                aivm_metadata=aivm_metadata,
            )
            provider_options["gguf_path"] = str(gguf_entry.gguf_path)

        if provider_options.get(
            "claim_jp_bert_graph"
        ) == "1" and not provider_options.get("jp_bert_gguf_path"):
            provider_options.update(
                self._jp_bert_provider_options(
                    resolve_jp_bert_onnx_path=resolve_jp_bert_onnx_path,
                    claim_synthesis_graph=True,
                )
            )

        return replace_provider_options(
            providers=providers,
            provider_name=self.config.provider_name,
            provider_options=provider_options,
        )

    def inference_session_scope(self) -> Any:
        """Return a context manager that routes Plugin EP sessions through OrtEpDevice."""

        return inference_session_scope(self.config)

    def _jp_bert_provider_options(
        self,
        *,
        resolve_jp_bert_onnx_path: JpBertOnnxPathResolver,
        claim_synthesis_graph: bool = False,
    ) -> dict[str, str]:
        if self.config is None:
            return {}

        provider_options = dict(self.config.provider_options)
        provider_options["claim_synthesis_graph"] = (
            "1" if claim_synthesis_graph else "0"
        )
        provider_options.pop("gguf_path", None)
        if provider_options.get("jp_bert_gguf_path"):
            return provider_options
        if self.jp_bert_cache is None:
            raise RuntimeError(
                "ONNX Plugin EP requires a JP-BERT GGUF cache when "
                "claim_jp_bert_graph=1."
            )

        jp_bert_gguf_entry = self.jp_bert_cache.ensure(
            onnx_path=resolve_jp_bert_onnx_path()
        )
        provider_options["jp_bert_gguf_path"] = str(jp_bert_gguf_entry.gguf_path)
        self.config.provider_options["jp_bert_gguf_path"] = str(
            jp_bert_gguf_entry.gguf_path
        )
        return provider_options


def validate_session_provider(
    *,
    session: Any | None,
    required_provider_name: str | None,
    context: str,
) -> None:
    """Ensure an explicit provider selection did not silently fall back."""

    if required_provider_name is None:
        return
    if session is None:
        raise RuntimeError(
            f"Strict ONNX provider mode expected {context} to expose an ONNX "
            "Runtime session, but none was found."
        )
    get_providers = getattr(session, "get_providers", None)
    if not callable(get_providers):
        return
    actual_providers = list(get_providers())
    actual_provider = actual_providers[0] if actual_providers else None
    if actual_provider != required_provider_name:
        raise RuntimeError(
            f"Strict ONNX provider mode expected {context} to use "
            f"{required_provider_name!r}, but ONNX Runtime selected "
            f"{actual_provider!r}. Full provider list: {actual_providers}"
        )

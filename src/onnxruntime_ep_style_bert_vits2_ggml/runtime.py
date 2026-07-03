"""Runtime integration helpers for the Style-Bert-VITS2 GGML Plugin EP."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onnxruntime_ep_style_bert_vits2_ggml import (
    get_ep_name,
    get_library_path,
)

ProviderSetting = str | tuple[str, dict[str, Any]]
DEFAULT_REGISTRATION_NAME = "style_bert_vits2_onnx_plugin_ep"

_INFERENCE_SESSION_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True)
class PluginExecutionProviderConfig:
    """External ONNX Runtime Plugin EP registration and selection settings."""

    provider_name: str
    provider_options: dict[str, str]
    library_path: Path | None = None
    registration_name: str = DEFAULT_REGISTRATION_NAME
    strict: bool = False


def default_backend_for_platform(platform_name: str | None = None) -> str:
    """Return the default TTS.cpp backend for the current packaged platform."""

    if (platform_name or sys.platform) == "darwin":
        return "metal"
    return "vulkan"


def default_cpu_threads(cpu_count: int | None = None) -> int:
    """Return a positive thread count for the TTS.cpp CPU backend."""

    return max(cpu_count if cpu_count is not None else os.cpu_count() or 1, 1)


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a possibly relative runtime path against a downstream base dir."""

    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    if base_dir is None:
        return resolved_path.resolve()
    return (Path(base_dir) / resolved_path).resolve()


def resolve_library_path(
    explicit_path: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicit or packaged native Plugin EP library path."""

    if explicit_path is not None:
        return resolve_path(explicit_path, base_dir=base_dir)
    return Path(get_library_path())


def build_provider_options(
    *,
    base_options: dict[str, str] | None = None,
    backend: str | None = None,
    precision: str = "accurate",
    tts_cpp_library_path: str | Path | None = None,
    vulkan_device: str | None = None,
    path_base_dir: str | Path | None = None,
    claim_jp_bert_graph: bool = True,
    claim_synthesis_graph: bool = True,
    eager_load_model: bool = True,
    require_tts_cpp_library: bool = True,
    cpu_count: int | None = None,
) -> dict[str, str]:
    """Build provider options for production Engine integration."""

    provider_options = dict(base_options or {})
    provider_options.setdefault("backend", backend or default_backend_for_platform())
    provider_options.setdefault(
        "claim_jp_bert_graph", _bool_option(claim_jp_bert_graph)
    )
    provider_options.setdefault(
        "claim_synthesis_graph", _bool_option(claim_synthesis_graph)
    )
    provider_options.setdefault("eager_load_model", _bool_option(eager_load_model))
    provider_options.setdefault("n_threads", "0")
    provider_options.setdefault("precision", precision)

    if tts_cpp_library_path is not None:
        provider_options.setdefault(
            "tts_cpp_library_path",
            str(resolve_path(tts_cpp_library_path, base_dir=path_base_dir)),
        )
    resolved_tts_cpp_library_path = provider_options.get("tts_cpp_library_path")
    if (
        resolved_tts_cpp_library_path is not None
        and resolved_tts_cpp_library_path != ""
    ):
        provider_options["tts_cpp_library_path"] = str(
            resolve_path(resolved_tts_cpp_library_path, base_dir=path_base_dir)
        )

    if vulkan_device is not None and provider_options.get("backend") != "cpu":
        provider_options.setdefault("device", vulkan_device)
    if (
        provider_options.get("backend") == "cpu"
        and provider_options.get("n_threads") == "0"
    ):
        provider_options["n_threads"] = str(default_cpu_threads(cpu_count))

    if require_tts_cpp_library and provider_options.get("tts_cpp_library_path") in {
        None,
        "",
    }:
        raise RuntimeError(
            "Style-Bert-VITS2 GGML Plugin EP requires tts_cpp_library_path."
        )
    return provider_options


def build_execution_provider_config(
    *,
    provider_options: dict[str, str],
    library_path: str | Path | None = None,
    path_base_dir: str | Path | None = None,
    provider_name: str | None = None,
    registration_name: str = DEFAULT_REGISTRATION_NAME,
    strict: bool = False,
) -> PluginExecutionProviderConfig:
    """Build a Plugin EP config using the packaged library when needed."""

    return PluginExecutionProviderConfig(
        provider_name=provider_name or get_ep_name(),
        provider_options=dict(provider_options),
        library_path=resolve_library_path(library_path, base_dir=path_base_dir),
        registration_name=registration_name,
        strict=strict,
    )


def configure_execution_provider(
    *,
    base_providers: Sequence[ProviderSetting],
    config: PluginExecutionProviderConfig | None,
    ort_module: Any | None = None,
    logger: Any | None = None,
) -> list[ProviderSetting]:
    """Register and prepend an external ONNX Runtime Plugin EP when configured."""

    providers = list(base_providers)
    if config is None:
        return providers

    ort = _load_onnxruntime(ort_module)
    registration_error: Exception | None = None
    if config.library_path is not None:
        try:
            ort.register_execution_provider_library(
                config.registration_name,
                str(config.library_path),
            )
            _log_info(
                logger,
                "Registered ONNX Runtime Plugin EP library %s from %s.",
                config.registration_name,
                config.library_path,
            )
        except Exception as ex:
            registration_error = ex

    ep_devices = get_ep_devices(config.provider_name, ort_module=ort)
    available_providers = ort.get_available_providers()
    if config.provider_name not in available_providers and not ep_devices:
        detail = (
            f"ONNX Runtime Plugin EP '{config.provider_name}' is not available. "
            f"Available providers: {available_providers}. "
            f"Available EP devices: {available_ep_device_names(ort_module=ort)}"
        )
        if registration_error is not None:
            detail += f" Registration failed: {registration_error}"
        if config.strict:
            raise RuntimeError(detail) from registration_error
        _log_warning(logger, detail)
        return providers

    if registration_error is not None:
        _log_warning(
            logger,
            "ONNX Runtime Plugin EP library registration reported an error, "
            "but provider %s is already available; continuing.",
            config.provider_name,
            exc_info=registration_error,
        )

    providers = [
        provider
        for provider in providers
        if provider_name_from_setting(provider) != config.provider_name
    ]
    providers.insert(0, (config.provider_name, dict(config.provider_options)))
    _log_info(
        logger,
        "Using external ONNX Runtime Plugin EP %s before fallback providers %s.",
        config.provider_name,
        [provider_name_from_setting(provider) for provider in providers[1:]],
    )
    return providers


def provider_name_from_setting(provider: ProviderSetting) -> str:
    """Return the ONNX Runtime provider name from string or option tuple forms."""

    return provider if isinstance(provider, str) else provider[0]


def provider_options_from_setting(provider: ProviderSetting) -> dict[str, Any]:
    """Return ONNX provider options from string or tuple provider forms."""

    return {} if isinstance(provider, str) else dict(provider[1])


def provider_names(providers: Sequence[ProviderSetting]) -> list[str]:
    """Return only ONNX Runtime provider names."""

    return [provider_name_from_setting(provider) for provider in providers]


def get_ep_devices(
    provider_name: str,
    *,
    ort_module: Any | None = None,
) -> list[Any]:
    """Return OrtEpDevice objects for a registered Plugin EP."""

    ort = _load_onnxruntime(ort_module)
    get_ep_devices_fn = getattr(ort, "get_ep_devices", None)
    if not callable(get_ep_devices_fn):
        return []

    return [
        ep_device
        for ep_device in get_ep_devices_fn()
        if getattr(ep_device, "ep_name", None) == provider_name
    ]


def available_ep_device_names(*, ort_module: Any | None = None) -> list[str]:
    """Return the ONNX Runtime EP device provider names visible to Python."""

    ort = _load_onnxruntime(ort_module)
    get_ep_devices_fn = getattr(ort, "get_ep_devices", None)
    if not callable(get_ep_devices_fn):
        return []

    return [str(getattr(ep_device, "ep_name", "")) for ep_device in get_ep_devices_fn()]


def find_provider_options(
    *,
    providers: Sequence[ProviderSetting] | None,
    provider_options: Sequence[dict[Any, Any]] | None,
    provider_name: str,
) -> dict[str, Any] | None:
    """Find provider options for a provider in ONNX Runtime Python arguments."""

    if providers is None:
        return None

    for index, provider in enumerate(providers):
        if provider_name_from_setting(provider) != provider_name:
            continue
        if isinstance(provider, tuple):
            return dict(provider[1])
        if provider_options is not None and index < len(provider_options):
            return dict(provider_options[index])
        return {}

    return None


@contextmanager
def inference_session_scope(
    config: PluginExecutionProviderConfig | None,
    *,
    ort_module: Any | None = None,
) -> Iterator[None]:
    """Create Plugin EP sessions through the ORT 1.24 EpDevice API when needed."""

    if config is None:
        yield
        return

    ort = _load_onnxruntime(ort_module)
    with _INFERENCE_SESSION_PATCH_LOCK:
        original_inference_session = ort.InferenceSession

        def plugin_aware_inference_session(
            path_or_bytes: str | bytes | Path,
            sess_options: Any | None = None,
            providers: Sequence[ProviderSetting] | None = None,
            provider_options: Sequence[dict[Any, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            plugin_provider_options = find_provider_options(
                providers=providers,
                provider_options=provider_options,
                provider_name=config.provider_name,
            )
            if plugin_provider_options is None:
                return original_inference_session(
                    path_or_bytes,
                    sess_options=sess_options,
                    providers=providers,
                    provider_options=provider_options,
                    **kwargs,
                )

            ep_devices = get_ep_devices(config.provider_name, ort_module=ort)
            if not ep_devices:
                return original_inference_session(
                    path_or_bytes,
                    sess_options=sess_options,
                    providers=providers,
                    provider_options=provider_options,
                    **kwargs,
                )

            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.add_provider_for_devices(
                ep_devices,
                {key: str(value) for key, value in plugin_provider_options.items()},
            )
            return original_inference_session(
                path_or_bytes,
                sess_options=sess_options,
                **kwargs,
            )

        ort.InferenceSession = plugin_aware_inference_session
        try:
            yield
        finally:
            ort.InferenceSession = original_inference_session


def replace_provider_options(
    *,
    providers: Sequence[ProviderSetting],
    provider_name: str,
    provider_options: dict[str, str],
) -> list[ProviderSetting]:
    """Replace one configured provider tuple while preserving provider order."""

    replaced: list[ProviderSetting] = []
    found = False
    for provider in providers:
        if provider_name_from_setting(provider) == provider_name:
            replaced.append((provider_name, dict(provider_options)))
            found = True
        else:
            replaced.append(provider)
    if not found:
        replaced.insert(0, (provider_name, dict(provider_options)))
    return replaced


def remove_provider(
    *,
    providers: Sequence[ProviderSetting],
    provider_name: str,
) -> list[ProviderSetting]:
    """Remove one configured provider while preserving fallback provider order."""

    return [
        provider
        for provider in providers
        if provider_name_from_setting(provider) != provider_name
    ]


def _bool_option(value: bool) -> str:
    return "1" if value else "0"


def _load_onnxruntime(ort_module: Any | None = None) -> Any:
    if ort_module is not None:
        return ort_module
    import onnxruntime

    return onnxruntime


def _log_info(logger: Any | None, message: str, *args: Any, **kwargs: Any) -> None:
    if logger is None:
        return
    logger.info(message, *args, **kwargs)


def _log_warning(logger: Any | None, message: str, *args: Any, **kwargs: Any) -> None:
    if logger is None:
        return
    logger.warning(message, *args, **kwargs)

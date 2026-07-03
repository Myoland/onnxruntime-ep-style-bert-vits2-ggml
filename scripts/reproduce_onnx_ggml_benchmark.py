"""Run a reviewer-friendly ONNX GGML benchmark reproduction."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

_EP_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_SRC = _EP_REPO_ROOT / "src"
if str(_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SRC))

from onnxruntime_ep_style_bert_vits2_ggml.engine_cache import (
    F16_GGUF_CONVERTER_VERSION,
    F32_GGUF_CONVERTER_VERSION,
)
from onnxruntime_ep_style_bert_vits2_ggml.models import (
    MODEL_PRESETS,
    ModelPreset,
    default_aivmx_download_path,
    download_aivmx,
    file_sha256,
)

_BENCHMARK_RUNNER = _EP_REPO_ROOT / "scripts" / "benchmark_onnx_ggml_provider.py"

_DEFAULT_TEXTS = (
    "テストです。",
    "今日はいい天気ですね。",
    "これは少し長めの文章です。GPUバックエンドの推論速度と音声品質を確認しています。",
)

_DEFAULT_WARMUP_TEXTS = (
    "測定用ではない短い文です。",
    "ウォームアップのために別の文章を読み上げます。",
    "測定対象とは異なる長めのウォームアップ文章です。バックエンドの初回処理だけを先に済ませます。",
)

_TEXT_LABELS = ("short", "medium", "long")


def _default_backends() -> list[str]:
    system = platform.system().lower()
    if system == "darwin":
        return ["onnx-cpu", "onnx-ggml-metal"]
    if system == "windows":
        return ["onnx-cpu", "onnx-directml", "onnx-ggml-vulkan"]
    return ["onnx-cpu", "onnx-ggml-vulkan"]


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _EP_REPO_ROOT / "benchmark-artifacts" / f"onnx-ggml-{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the AivisHub model, run the ONNX GGML benchmark runner, "
            "and write raw JSON plus a Markdown summary for PR review."
        )
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PRESETS),
        default="mao",
        help="AivisHub model preset used for style id and SHA-256 verification.",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "AivisSpeech-Engine checkout used to run the benchmark. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--aivmx-path",
        type=Path,
        default=None,
        help=(
            "Downloaded AIVMX path. When omitted, the selected preset is downloaded "
            "from AivisHub and verified."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=(
            "onnx-cpu",
            "onnx-directml",
            "onnx-cuda",
            "onnx-ggml-metal",
            "onnx-ggml-cpu",
            "onnx-ggml-vulkan",
        ),
        action="append",
        default=None,
        help="Backend to benchmark. Repeat to override the platform default set.",
    )
    parser.add_argument(
        "--include-cuda",
        action="store_true",
        help="Add onnx-cuda to the default Linux backend set.",
    )
    parser.add_argument(
        "--ggml-native-library-path",
        type=Path,
        required=True,
        help="TTS.cpp shared library path, such as libtts.so, libtts.dylib, or tts.dll.",
    )
    parser.add_argument(
        "--onnx-ep-library-path",
        type=Path,
        default=None,
        help="Style-Bert-VITS2 GGML ONNX Runtime Plugin EP shared library path.",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional native library directory. Repeated values are prepended "
            "to PATH on Windows, DYLD_LIBRARY_PATH on macOS, or LD_LIBRARY_PATH elsewhere."
        ),
    )
    parser.add_argument(
        "--ggml-model-cache-dir",
        type=Path,
        default=None,
        help="GGUF cache directory. Defaults to <output-dir>/gguf-cache.",
    )
    parser.add_argument(
        "--ggml-jp-bert-gguf-path",
        type=Path,
        default=None,
        help=(
            "Optional prepared JP-BERT GGUF. When omitted, the engine cache uses "
            "the default JP-BERT F16 linear artifact."
        ),
    )
    parser.add_argument(
        "--synthesis-gguf",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Synthesis GGUF profile. The PR default is fp32; fp16 is explicit opt-in.",
    )
    parser.add_argument(
        "--ggml-vulkan-device",
        default=None,
        help="Provider option device id for Vulkan.",
    )
    parser.add_argument(
        "--ggml-vulkan-precision",
        choices=("accurate", "fast"),
        default="fast",
        help="Provider option precision for Vulkan.",
    )
    parser.add_argument(
        "--ggml-vulkan-math-mode",
        choices=("f32", "coopmat", "fp16", "fp16-coopmat"),
        default="coopmat",
        help="Provider option controlling ggml-vulkan F16 and cooperative matrix use.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured syntheses per backend/text.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup syntheses per backend/text.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use tempoDynamicsScale=0, noise=0, and noise_w=0 so the runner "
            "also emits PCM truth comparisons against ONNX CPU."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory. Defaults to benchmark-artifacts/onnx-ggml-<timestamp>.",
    )
    parser.add_argument(
        "--skip-sha256-check",
        action="store_true",
        help="Skip AIVMX SHA-256 verification. Use only for private local models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the benchmark command without executing it.",
    )
    return parser.parse_args()


def _validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} が見つかりません: {path}")


def _selected_backends(args: argparse.Namespace) -> list[str]:
    backends = list(args.backend or _default_backends())
    if args.include_cuda and "onnx-cuda" not in backends:
        backends.insert(1 if "onnx-cpu" in backends else 0, "onnx-cuda")
    return backends


def _synthesis_converter_version(profile: str) -> str:
    if profile == "fp16":
        return F16_GGUF_CONVERTER_VERSION
    return F32_GGUF_CONVERTER_VERSION


def _extend_command(command: list[str], option: str, values: Iterable[Any]) -> None:
    for value in values:
        command.extend([option, str(value)])


def _benchmark_command(
    *,
    args: argparse.Namespace,
    model: ModelPreset,
    backends: Sequence[str],
    output_dir: Path,
) -> list[str]:
    output_json = output_dir / "raw.json"
    audio_dir = output_dir / "audio"
    gguf_cache_dir = args.ggml_model_cache_dir or output_dir / "gguf-cache"

    command = [
        sys.executable,
        str(_BENCHMARK_RUNNER),
        "--aivmx_path",
        str(args.aivmx_path),
        "--style_id",
        str(model.style_id),
        "--warmup_runs",
        str(args.warmup_runs),
        "--runs",
        str(args.runs),
        "--tempo_dynamics_scale",
        "0.0" if args.deterministic else "1.0",
        "--ggml_native_library_path",
        str(args.ggml_native_library_path),
        "--ggml_model_cache_dir",
        str(gguf_cache_dir),
        "--ggml_synthesis_converter_version",
        _synthesis_converter_version(args.synthesis_gguf),
        "--ggml_vulkan_precision",
        args.ggml_vulkan_precision,
        "--ggml_vulkan_math_mode",
        args.ggml_vulkan_math_mode,
        "--output_json",
        str(output_json),
        "--audio_output_dir",
        str(audio_dir),
    ]
    if args.onnx_ep_library_path is not None:
        command.extend(["--onnx_ep_library_path", str(args.onnx_ep_library_path)])
    if args.ggml_jp_bert_gguf_path is not None:
        command.extend(["--ggml_jp_bert_gguf_path", str(args.ggml_jp_bert_gguf_path)])
    if args.ggml_vulkan_device is not None:
        command.extend(["--ggml_vulkan_device", args.ggml_vulkan_device])
    if args.deterministic:
        command.extend(["--noise_scale", "0.0", "--noise_scale_w", "0.0"])
    else:
        command.append("--skip_truth_comparison")

    _extend_command(command, "--backend", backends)
    _extend_command(command, "--text", _DEFAULT_TEXTS)
    _extend_command(command, "--warmup_text", _DEFAULT_WARMUP_TEXTS)
    return command


def _library_path_env_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "PATH"
    if system == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def _build_env(library_dirs: Sequence[Path]) -> dict[str, str]:
    env = os.environ.copy()
    if not library_dirs:
        return env
    key = _library_path_env_name()
    current = env.get(key)
    prefix = os.pathsep.join(str(path) for path in library_dirs)
    env[key] = prefix if not current else f"{prefix}{os.pathsep}{current}"
    return env


def _display_command(command: Sequence[str]) -> str:
    if platform.system().lower() == "windows":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _run_command(
    command: Sequence[str], *, env: dict[str, str], log_path: Path, cwd: Path
) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(
            f"benchmark runner failed with exit code {return_code}. See {log_path}"
        )


def _try_available_onnx_providers() -> list[str]:
    try:
        import onnxruntime
    except Exception as error:  # noqa: BLE001
        return [f"<onnxruntime import failed: {error}>"]
    return list(onnxruntime.get_available_providers())


def _ordered_text_labels(summary: Sequence[dict[str, Any]]) -> list[str]:
    labels = {str(item["text_label"]) for item in summary}
    ordered = [label for label in _TEXT_LABELS if label in labels]
    ordered.extend(sorted(labels.difference(ordered)))
    return ordered


def _ordered_backends(summary: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "onnx-cpu",
        "onnx-cuda",
        "onnx-directml",
        "onnx-ggml-vulkan",
        "onnx-ggml-metal",
        "onnx-ggml-cpu",
    ]
    backends = {str(item["backend"]) for item in summary}
    ordered = [backend for backend in preferred if backend in backends]
    ordered.extend(sorted(backends.difference(ordered)))
    return ordered


def _format_float(value: Any) -> str:
    return f"{float(value):.3f}"


def _summary_table(payload: dict[str, Any]) -> str:
    summary = list(payload.get("summary", []))
    if not summary:
        return "結果がありません。\n"
    by_key = {(str(item["backend"]), str(item["text_label"])): item for item in summary}
    labels = _ordered_text_labels(summary)
    backends = _ordered_backends(summary)
    lines = [
        "| テキスト | " + " | ".join(backends) + " |",
        "| --- | " + " | ".join("---:" for _ in backends) + " |",
    ]
    for label in labels:
        cells = []
        for backend in backends:
            item = by_key.get((backend, label))
            cells.append("-" if item is None else _format_float(item["rtf_mean"]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    overall_cells = []
    for backend in backends:
        values = [
            float(item["rtf_mean"])
            for item in summary
            if str(item["backend"]) == backend
        ]
        overall_cells.append(
            "-" if not values else _format_float(sum(values) / len(values))
        )
    lines.append("| 平均 | " + " | ".join(overall_cells) + " |")
    return "\n".join(lines) + "\n"


def _truth_table(payload: dict[str, Any]) -> str:
    comparisons = list(payload.get("truth_comparison", []))
    if not comparisons:
        return "決定論的な PCM 比較は実行していません。\n"
    lines = [
        "| backend | text | sample delta | RMSE | max abs diff | correlation |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        correlation = item.get("correlation")
        lines.append(
            "| "
            f"{item['backend']} | {item['text_label']} | {item['sample_delta']} | "
            f"{float(item['rmse']):.6f} | {float(item['max_abs_diff']):.6f} | "
            f"{'-' if correlation is None else f'{float(correlation):.6f}'} |"
        )
    return "\n".join(lines) + "\n"


def _write_summary(
    *,
    payload: dict[str, Any],
    output_dir: Path,
    command: Sequence[str],
    env_key: str,
    model: ModelPreset,
    args: argparse.Namespace,
) -> None:
    summary_path = output_dir / "summary.md"
    profile = payload.get("profile", {})
    provider_evidence = payload.get("provider_evidence", {})
    reproduction = payload.get("reproduction", {})
    lines = [
        "# ONNX GGML ベンチマーク再現結果",
        "",
        "## モデル",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| モデル | {model.name} |",
        f"| AivisHub | {model.aivishub_url} |",
        f"| バージョン | `{model.version}` |",
        f"| スタイル ID | `{model.style_id}` |",
        f"| SHA-256 | `{model.sha256}` |",
        "",
        "## 実行プロファイル",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| synthesis GGUF | `{args.synthesis_gguf.upper()}` |",
        "| JP-BERT GGUF | `F16 linear`（既定キャッシュ、または指定されたローカルファイル） |",
        f"| warmup / runs | `{profile.get('warmup_runs')}` / `{profile.get('runs')}` |",
        f"| tempoDynamicsScale | `{profile.get('tempo_dynamics_scale')}` |",
        f"| noise / noise_w | `{profile.get('noise_scale')}` / `{profile.get('noise_scale_w')}` |",
        f"| Vulkan precision | `{profile.get('ggml_vulkan_precision')}` |",
        f"| Vulkan math mode | `{profile.get('ggml_vulkan_math_mode')}` |",
        "",
        "## 環境",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| OS | `{reproduction.get('platform')}` |",
        f"| Python | `{reproduction.get('python')}` |",
        f"| ONNX Runtime providers | `{', '.join(reproduction.get('onnxruntime_providers', []))}` |",
        f"| native library env | `{env_key}` |",
        "",
        "## RTF",
        "",
        "RTF は `elapsed_seconds / output_duration_seconds` です。小さいほど高速です。",
        "",
        _summary_table(payload).rstrip(),
        "",
        "## PCM 比較",
        "",
        _truth_table(payload).rstrip(),
        "",
        "## Provider 証跡",
        "",
        "```json",
        json.dumps(provider_evidence, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 生成物",
        "",
        f"- Raw JSON: `{(output_dir / 'raw.json').name}`",
        f"- 実行ログ: `{(output_dir / 'benchmark.log').name}`",
        "- 代表音声 WAV: `audio/`",
        "- GGUF キャッシュ: `gguf-cache/`",
        "",
        "## 実行コマンド",
        "",
        "```bash",
        _display_command(command),
        "```",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _augment_payload(
    *,
    raw_json_path: Path,
    model: ModelPreset,
    command: Sequence[str],
    args: argparse.Namespace,
    env_key: str,
) -> dict[str, Any]:
    payload = cast(
        dict[str, Any], json.loads(raw_json_path.read_text(encoding="utf-8"))
    )
    payload.setdefault("profile", {})
    payload["profile"]["model_source"] = {
        "repository": "AivisHub",
        "url": model.aivishub_url,
        "model_name": model.name,
        "model_uuid": model.model_uuid,
        "model_version": model.version,
        "sha256": model.sha256,
    }
    payload["reproduction"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "onnxruntime_providers": _try_available_onnx_providers(),
        "command": list(command),
        "display_command": _display_command(command),
        "engine_dir": str(args.engine_dir),
        "native_library_env": env_key,
        "library_dirs": [str(path) for path in args.library_dir],
    }
    raw_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    """Run benchmark reproduction and write reviewer artifacts."""

    args = _parse_args()
    model = MODEL_PRESETS[args.model]
    args.engine_dir = args.engine_dir.expanduser().resolve()
    if not (args.engine_dir / "run.py").is_file() or not (
        args.engine_dir / "voicevox_engine"
    ).is_dir():
        raise SystemExit(f"AivisSpeech-Engine checkout was not found: {args.engine_dir}")
    if args.aivmx_path is None:
        default_aivmx_path = default_aivmx_download_path(
            model, root_dir=_EP_REPO_ROOT
        ).resolve()
        args.aivmx_path = (
            default_aivmx_path
            if args.dry_run
            else download_aivmx(model, default_aivmx_path)
        )
    else:
        args.aivmx_path = args.aivmx_path.expanduser().resolve()
    args.ggml_native_library_path = args.ggml_native_library_path.expanduser().resolve()
    if args.onnx_ep_library_path is not None:
        args.onnx_ep_library_path = args.onnx_ep_library_path.expanduser().resolve()
    if args.ggml_jp_bert_gguf_path is not None:
        args.ggml_jp_bert_gguf_path = args.ggml_jp_bert_gguf_path.expanduser().resolve()
    args.library_dir = [path.expanduser().resolve() for path in args.library_dir]

    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    backends = _selected_backends(args)

    if not args.dry_run:
        _validate_file(args.aivmx_path, "AIVMX")
        _validate_file(args.ggml_native_library_path, "TTS.cpp shared library")
        if args.onnx_ep_library_path is not None:
            _validate_file(args.onnx_ep_library_path, "ONNX GGML Plugin EP library")
        if args.ggml_jp_bert_gguf_path is not None:
            _validate_file(args.ggml_jp_bert_gguf_path, "JP-BERT GGUF")

    if not args.dry_run and not args.skip_sha256_check:
        actual_sha256 = file_sha256(args.aivmx_path)
        if actual_sha256 != model.sha256:
            raise SystemExit(
                "AIVMX SHA-256 が一致しません。\n"
                f"expected: {model.sha256}\n"
                f"actual:   {actual_sha256}\n"
                f"AivisHub: {model.aivishub_url}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    env_key = _library_path_env_name()
    env = _build_env(args.library_dir)
    env["AIVIS_ENGINE_DIR"] = str(args.engine_dir)
    command = _benchmark_command(
        args=args,
        model=model,
        backends=backends,
        output_dir=output_dir,
    )

    command_text = _display_command(command)
    print(command_text)
    if args.dry_run:
        return

    log_path = output_dir / "benchmark.log"
    _run_command(command, env=env, log_path=log_path, cwd=args.engine_dir)

    raw_json_path = output_dir / "raw.json"
    payload = _augment_payload(
        raw_json_path=raw_json_path,
        model=model,
        command=command,
        args=args,
        env_key=env_key,
    )
    _write_summary(
        payload=payload,
        output_dir=output_dir,
        command=command,
        env_key=env_key,
        model=model,
        args=args,
    )
    print(f"summary: {output_dir / 'summary.md'}")
    print(f"raw json: {raw_json_path}")


if __name__ == "__main__":
    main()

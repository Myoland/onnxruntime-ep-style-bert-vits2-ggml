"""PyInstaller helpers for downstream Engine runtime bundling."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns, which

PACKAGE_NAME = "onnxruntime_ep_style_bert_vits2_ggml"


@dataclass(frozen=True)
class RuntimeBundlePaths:
    """Resolved native sidecars from a runtime bundle."""

    tts_library_path: Path
    onnx_ep_library_path: Path
    library_dirs: tuple[Path, ...]


def env_paths(name: str) -> list[Path]:
    """Return path list from an os.pathsep-separated environment variable."""

    value = os.environ.get(name, "")
    return [Path(item) for item in value.split(os.pathsep) if item]


def resolve_runtime_bundle(bundle_dir: str | Path) -> RuntimeBundlePaths:
    """Resolve documented sidecar locations from a runtime bundle directory."""

    resolved_bundle_dir = Path(bundle_dir).expanduser().resolve()
    if not resolved_bundle_dir.is_dir():
        raise FileNotFoundError(
            f"ONNX GGML runtime bundle directory was not found: {resolved_bundle_dir}"
        )

    tts_library_path = _resolve_existing_bundle_file(
        bundle_dir=resolved_bundle_dir,
        candidates=_bundle_tts_library_candidates(),
        label="TTS.cpp shared library",
    )
    onnx_ep_library_path = _resolve_existing_bundle_file(
        bundle_dir=resolved_bundle_dir,
        candidates=_bundle_onnx_ep_library_candidates(),
        label="ONNX GGML Plugin EP library",
    )
    library_dirs = tuple(
        path.resolve()
        for path in (resolved_bundle_dir / "lib", onnx_ep_library_path.parent)
        if path.is_dir()
    )
    return RuntimeBundlePaths(
        tts_library_path=tts_library_path,
        onnx_ep_library_path=onnx_ep_library_path,
        library_dirs=library_dirs,
    )


def copy_runtime_bundle_from_env(target_dir: str | Path) -> None:
    """Copy ONNX GGML runtime files using downstream Engine build env vars."""

    copy_runtime_bundle(
        target_dir=target_dir,
        required=os.environ.get("STYLE_BERT_VITS2_GGML_REQUIRED") == "1",
        bundle_dirs=env_paths("STYLE_BERT_VITS2_GGML_BUNDLE_DIR"),
        ep_library_paths=env_paths("STYLE_BERT_VITS2_GGML_EP_LIBRARY_PATH"),
        tts_library_paths=env_paths("STYLE_BERT_VITS2_TTS_CPP_LIBRARY_PATH"),
        tts_library_dirs=env_paths("STYLE_BERT_VITS2_TTS_CPP_LIBRARY_DIRS"),
    )


def copy_runtime_bundle(
    *,
    target_dir: str | Path,
    required: bool = False,
    bundle_dirs: Sequence[str | Path] = (),
    ep_library_paths: Sequence[str | Path] = (),
    tts_library_paths: Sequence[str | Path] = (),
    tts_library_dirs: Sequence[str | Path] = (),
) -> None:
    """Copy packaged Plugin EP, TTS.cpp sidecar, and ggml dependencies."""

    target_path = Path(target_dir)
    resolved_bundle_dirs = [Path(path).expanduser().resolve() for path in bundle_dirs]
    resolved_ep_library_paths = [
        Path(path).expanduser().resolve() for path in ep_library_paths
    ]
    resolved_tts_library_paths = [
        Path(path).expanduser().resolve() for path in tts_library_paths
    ]
    resolved_tts_library_dirs = [
        Path(path).expanduser().resolve() for path in tts_library_dirs
    ]

    package_src = _package_dir(PACKAGE_NAME)
    package_dest = target_path / PACKAGE_NAME
    if package_src is not None and package_src.exists():
        copytree(
            package_src,
            package_dest,
            dirs_exist_ok=True,
            ignore=ignore_patterns("__pycache__", "*.pyc"),
        )

    ep_candidates = list(resolved_ep_library_paths)
    ep_candidates += [
        bundle_dir / PACKAGE_NAME / "lib" / name
        for bundle_dir in resolved_bundle_dirs
        for name in _ep_library_names()
    ]
    ep_candidates += [package_dest / "lib" / name for name in _ep_library_names()]
    if package_src is not None:
        ep_candidates += [package_src / "lib" / name for name in _ep_library_names()]
    copied_ep = _copy_existing_files(
        ep_candidates,
        target_path / PACKAGE_NAME / "lib",
    )

    resolved_tts_library_dirs += [
        bundle_dir / "lib" for bundle_dir in resolved_bundle_dirs
    ]
    copied_tts = _copy_existing_files(resolved_tts_library_paths, target_path / "lib")
    copied_tts += _copy_matching_files(
        resolved_tts_library_dirs,
        _tts_library_patterns(),
        target_path / "lib",
    )
    copied_deps = _copy_matching_files(
        resolved_tts_library_dirs,
        _dependency_library_patterns(),
        target_path / "lib",
    )
    _patch_linux_rpath([*copied_tts, *copied_deps])
    _patch_macos_rpath([*copied_ep, *copied_tts, *copied_deps])

    if required:
        if not copied_ep:
            raise RuntimeError(
                "STYLE_BERT_VITS2_GGML_REQUIRED=1 but native Plugin EP was not packaged."
            )
        if not copied_tts:
            raise RuntimeError(
                "STYLE_BERT_VITS2_GGML_REQUIRED=1 but TTS.cpp runtime was not packaged."
            )

    print(f"Packaged ONNX GGML EP sidecars: {[str(path) for path in copied_ep]}")
    print(
        f"Packaged TTS.cpp sidecars: {[str(path) for path in [*copied_tts, *copied_deps]]}"
    )


def _platform_key() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _bundle_tts_library_candidates() -> tuple[Path, ...]:
    platform_key = _platform_key()
    if platform_key == "windows":
        return (Path("lib") / "tts.dll", Path("lib") / "libtts.dll")
    if platform_key == "darwin":
        return (Path("lib") / "libtts.dylib",)
    return (Path("lib") / "libtts.so",)


def _bundle_onnx_ep_library_candidates() -> tuple[Path, ...]:
    ep_lib_dir = Path(PACKAGE_NAME) / "lib"
    platform_key = _platform_key()
    if platform_key == "windows":
        return (ep_lib_dir / "style_bert_vits2_ggml_onnx_ep.dll",)
    if platform_key == "darwin":
        return (ep_lib_dir / "libstyle_bert_vits2_ggml_onnx_ep.dylib",)
    return (ep_lib_dir / "libstyle_bert_vits2_ggml_onnx_ep.so",)


def _resolve_existing_bundle_file(
    *,
    bundle_dir: Path,
    candidates: Sequence[Path],
    label: str,
) -> Path:
    for relative_path in candidates:
        candidate = bundle_dir / relative_path
        if candidate.is_file():
            return candidate.resolve()

    expected = "\n".join(f"  - {bundle_dir / candidate}" for candidate in candidates)
    raise FileNotFoundError(f"{label} was not found in runtime bundle:\n{expected}")


def _ep_library_names() -> tuple[str, ...]:
    return (
        "libstyle_bert_vits2_ggml_onnx_ep.so",
        "libstyle_bert_vits2_ggml_onnx_ep.dylib",
        "style_bert_vits2_ggml_onnx_ep.dll",
    )


def _tts_library_patterns() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("tts.dll", "libtts.dll")
    if sys.platform == "darwin":
        return ("libtts.dylib",)
    return ("libtts.so", "libtts.so.*")


def _dependency_library_patterns() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("ggml*.dll", "libggml*.dll")
    if sys.platform == "darwin":
        return ("libggml*.dylib",)
    return ("libggml*.so", "libggml*.so.*")


def _package_dir(package_name: str) -> Path | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return None
    locations = list(spec.submodule_search_locations)
    if not locations:
        return None
    return Path(locations[0]).resolve()


def _copy_existing_files(candidates: Iterable[Path], dest_dir: Path) -> list[Path]:
    copied: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen_basenames: set[str] = set()
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.name in seen_basenames:
            continue
        seen_basenames.add(candidate.name)
        dest = dest_dir / candidate.name
        if candidate.resolve() == dest.resolve():
            copied.append(dest)
            continue
        copy2(candidate, dest)
        copied.append(dest)
    return copied


def _copy_matching_files(
    search_dirs: Iterable[Path],
    patterns: Sequence[str],
    dest_dir: Path,
) -> list[Path]:
    candidates: list[Path] = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(search_dir.rglob(pattern)))
    return _copy_existing_files(candidates, dest_dir)


def _patch_linux_rpath(libraries: Sequence[Path]) -> None:
    if not sys.platform.startswith("linux"):
        return
    patchelf = which("patchelf")
    if patchelf is None:
        print(
            "WARNING: patchelf is not available; ONNX GGML sidecars keep their original rpath."
        )
        return
    for library in libraries:
        if ".so" not in library.name:
            continue
        subprocess.run(
            [patchelf, "--set-rpath", "$ORIGIN", str(library)],
            check=True,
        )


def _macos_rpaths(library: Path) -> list[str]:
    result = subprocess.run(
        ["otool", "-l", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    rpaths: list[str] = []
    is_rpath_command = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped == "cmd LC_RPATH":
            is_rpath_command = True
            continue
        if is_rpath_command and stripped.startswith("path "):
            rpaths.append(stripped.removeprefix("path ").split(" (offset", 1)[0])
            is_rpath_command = False
    return rpaths


def _patch_macos_rpath(libraries: Sequence[Path]) -> None:
    if sys.platform != "darwin":
        return
    install_name_tool = which("install_name_tool")
    if install_name_tool is None:
        print(
            "WARNING: install_name_tool is not available; ONNX GGML sidecars keep their original rpath."
        )
        return

    for library in libraries:
        if library.suffix != ".dylib":
            continue
        rpaths = _macos_rpaths(library)
        if "@loader_path" not in rpaths:
            subprocess.run(
                [install_name_tool, "-add_rpath", "@loader_path", str(library)],
                check=True,
            )
        for rpath in rpaths:
            if rpath == "@loader_path":
                continue
            subprocess.run(
                [install_name_tool, "-delete_rpath", rpath, str(library)],
                check=True,
            )

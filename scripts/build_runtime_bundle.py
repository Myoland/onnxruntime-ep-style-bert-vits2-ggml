"""Build a local Aivis GGML ONNX Runtime Plugin EP bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TTS_CPP_REPO = "https://github.com/Myoland/TTS.cpp.git"
DEFAULT_TTS_CPP_REF = "94792ed2599656618c1d5eb3934754c391eb2a54"
DEFAULT_GGML_REPO = "https://github.com/Myoland/ggml.git"


@dataclass(frozen=True)
class PlatformConfig:
    ort_version: str
    ort_archive: str
    tts_library_names: tuple[str, ...]
    ggml_patterns: tuple[str, ...]
    tts_cmake_options: tuple[str, ...]


def _platform_config() -> PlatformConfig:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine in {"x86_64", "amd64"}:
        return PlatformConfig(
            ort_version="1.26.0",
            ort_archive="onnxruntime-linux-x64-{version}.tgz",
            tts_library_names=("libtts.so",),
            ggml_patterns=("libggml*.so", "libggml*.so.*"),
            tts_cmake_options=("-DGGML_VULKAN=ON",),
        )
    if system == "Darwin" and machine == "x86_64":
        return PlatformConfig(
            ort_version="1.23.2",
            ort_archive="onnxruntime-osx-x86_64-{version}.tgz",
            tts_library_names=("libtts.dylib",),
            ggml_patterns=("libggml*.dylib",),
            tts_cmake_options=("-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"),
        )
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return PlatformConfig(
            ort_version="1.27.0",
            ort_archive="onnxruntime-osx-arm64-{version}.tgz",
            tts_library_names=("libtts.dylib",),
            ggml_patterns=("libggml*.dylib",),
            tts_cmake_options=("-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"),
        )
    if system == "Windows" and machine in {"amd64", "x86_64"}:
        return PlatformConfig(
            ort_version="1.24.4",
            ort_archive="onnxruntime-win-x64-{version}.zip",
            tts_library_names=("tts.dll", "libtts.dll"),
            ggml_patterns=("ggml*.dll", "libggml*.dll"),
            tts_cmake_options=("-DGGML_VULKAN=ON",),
        )
    raise SystemExit(f"Unsupported platform for local bundle build: {system}:{machine}")


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _download(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    with urllib.request.urlopen(url) as response:
        output_path.write_bytes(response.read())


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".extracted"
    if marker.exists():
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir)
        children = [path for path in output_dir.iterdir() if path.is_dir()]
        if len(children) == 1 and children[0].name.startswith("onnxruntime-"):
            for child in children[0].iterdir():
                shutil.move(str(child), output_dir / child.name)
            children[0].rmdir()
        shutil.rmtree(output_dir / "lib", ignore_errors=True)
    else:
        with tarfile.open(archive_path) as archive:
            top_level = archive.getmembers()[0].name.split("/", 1)[0]
            archive.extractall(output_dir)
        extracted_root = output_dir / top_level
        for child in extracted_root.iterdir():
            if child.name == "lib":
                continue
            shutil.move(str(child), output_dir / child.name)
        shutil.rmtree(extracted_root, ignore_errors=True)
    marker.write_text("ok\n", encoding="utf-8")


def _find_ort_include_dir(ort_dir: Path) -> Path:
    matches = sorted(ort_dir.rglob("onnxruntime_cxx_api.h"))
    if not matches:
        raise SystemExit(f"onnxruntime_cxx_api.h was not found under {ort_dir}")
    return matches[0].parent


def _prepare_ort_headers(
    *, build_dir: Path, download_dir: Path, config: PlatformConfig, ort_version: str
) -> Path:
    archive_name = config.ort_archive.format(version=ort_version)
    archive_path = download_dir / archive_name
    url = (
        "https://github.com/microsoft/onnxruntime/releases/download/"
        f"v{ort_version}/{archive_name}"
    )
    ort_dir = build_dir / f"onnxruntime-{ort_version}"
    _download(url, archive_path)
    _extract_archive(archive_path, ort_dir)
    return _find_ort_include_dir(ort_dir)


def _prepare_tts_cpp_source(
    *,
    source_dir: Path,
    repo: str,
    ref: str,
    ggml_repo: str,
) -> None:
    if not source_dir.exists():
        _run(["git", "clone", repo, str(source_dir)])
    _run(["git", "-C", str(source_dir), "fetch", "--tags", "origin"])
    _run(["git", "-C", str(source_dir), "checkout", ref])
    _run(["git", "-C", str(source_dir), "submodule", "set-url", "ggml", ggml_repo])
    _run(["git", "-C", str(source_dir), "submodule", "update", "--init", "--recursive"])


def _build_tts_cpp(
    *, source_dir: Path, build_dir: Path, config: PlatformConfig
) -> None:
    configure_command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DTTS_BUILD_EXAMPLES=OFF",
        "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
        "-DCMAKE_BUILD_RPATH=$ORIGIN",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        *config.tts_cmake_options,
    ]
    _run(configure_command)
    _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "tts",
            "--parallel",
        ]
    )


def _build_plugin_ep(*, ort_include_dir: Path, native_build_dir: Path) -> None:
    _run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT / "native"),
            "-B",
            str(native_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DORT_INCLUDE_DIR={ort_include_dir}",
        ]
    )
    _run(
        ["cmake", "--build", str(native_build_dir), "--config", "Release", "--parallel"]
    )
    _run(
        [
            "cmake",
            "--install",
            str(native_build_dir),
            "--config",
            "Release",
            "--prefix",
            str(REPO_ROOT / "src"),
        ]
    )


def _copy_existing_files(
    candidates: Iterable[Path], *, dest_dir: Path, required_label: str
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file() or candidate.name in seen:
            continue
        seen.add(candidate.name)
        dest = dest_dir / candidate.name
        shutil.copy2(candidate, dest)
        copied.append(dest)
    if not copied:
        raise SystemExit(f"No {required_label} files were found.")
    return copied


def _glob_many(search_dirs: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            matches.extend(sorted(search_dir.rglob(pattern)))
    return matches


def _patch_linux_rpath(libraries: Sequence[Path]) -> None:
    if platform.system() != "Linux":
        return
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        print(
            "warning: patchelf was not found; copied libraries keep their original rpath"
        )
        return
    for library in libraries:
        if ".so" in library.name:
            _run([patchelf, "--set-rpath", "$ORIGIN", str(library)])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    *,
    bundle_dir: Path,
    ort_version: str,
    tts_cpp_ref: str,
    plugin_libraries: Sequence[Path],
    tts_libraries: Sequence[Path],
    ggml_libraries: Sequence[Path],
) -> None:
    libraries = [*plugin_libraries, *tts_libraries, *ggml_libraries]
    manifest = {
        "schema": "aivis-ggml-runtime-bundle-v1",
        "provider_name": "AivisGgmlExecutionProvider",
        "ort_version": ort_version,
        "ort_plugin_ep_api_version": 26,
        "tts_cpp_repo": DEFAULT_TTS_CPP_REPO,
        "tts_cpp_ref": tts_cpp_ref,
        "tts_cpp_runtime_abi_version": 1,
        "gguf_schema_version": 1,
        "libraries": [
            {
                "path": str(path.relative_to(bundle_dir)),
                "sha256": _file_sha256(path),
                "size": path.stat().st_size,
            }
            for path in libraries
        ],
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_bundle(
    *,
    output_dir: Path,
    tts_build_dir: Path,
    config: PlatformConfig,
    ort_version: str,
    tts_cpp_ref: str,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    package_lib_dir = REPO_ROOT / "src" / "onnxruntime_ep_aivis_ggml" / "lib"
    plugin_libraries = _copy_existing_files(
        package_lib_dir.glob("*aivis_ggml_onnx_ep*"),
        dest_dir=output_dir / "onnxruntime_ep_aivis_ggml" / "lib",
        required_label="Plugin EP",
    )

    tts_search_dirs = (
        tts_build_dir / "src",
        tts_build_dir / "ggml" / "src",
        tts_build_dir / "ggml" / "src" / "ggml-vulkan",
        tts_build_dir / "ggml" / "src" / "ggml-metal",
        tts_build_dir / "ggml" / "src" / "ggml-blas",
    )
    tts_libraries = _copy_existing_files(
        _glob_many(tts_search_dirs, config.tts_library_names),
        dest_dir=output_dir / "lib",
        required_label="TTS.cpp runtime",
    )
    ggml_libraries = _copy_existing_files(
        _glob_many(tts_search_dirs, config.ggml_patterns),
        dest_dir=output_dir / "lib",
        required_label="ggml runtime dependency",
    )
    _patch_linux_rpath([*tts_libraries, *ggml_libraries])
    _write_manifest(
        bundle_dir=output_dir,
        ort_version=ort_version,
        tts_cpp_ref=tts_cpp_ref,
        plugin_libraries=plugin_libraries,
        tts_libraries=tts_libraries,
        ggml_libraries=ggml_libraries,
    )


def _parse_args() -> argparse.Namespace:
    config = _platform_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--download-dir", type=Path, default=REPO_ROOT / "download")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "dist" / "aivis-ggml-runtime"
    )
    parser.add_argument("--ort-version", default=config.ort_version)
    parser.add_argument("--ort-include-dir", type=Path, default=None)
    parser.add_argument("--tts-cpp-repo", default=DEFAULT_TTS_CPP_REPO)
    parser.add_argument("--tts-cpp-ref", default=DEFAULT_TTS_CPP_REF)
    parser.add_argument("--ggml-repo", default=DEFAULT_GGML_REPO)
    parser.add_argument("--tts-cpp-source-dir", type=Path, default=None)
    parser.add_argument("--tts-cpp-build-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-tts-cpp-build",
        action="store_true",
        help="Use --tts-cpp-build-dir as an already-built TTS.cpp tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _platform_config()
    build_dir = args.build_dir.resolve()
    download_dir = args.download_dir.resolve()
    output_dir = args.output_dir.resolve()
    tts_source_dir = (
        args.tts_cpp_source_dir.resolve()
        if args.tts_cpp_source_dir is not None
        else build_dir / "TTS.cpp"
    )
    tts_build_dir = (
        args.tts_cpp_build_dir.resolve()
        if args.tts_cpp_build_dir is not None
        else build_dir / "TTS.cpp-build"
    )

    ort_include_dir = (
        args.ort_include_dir.resolve()
        if args.ort_include_dir is not None
        else _prepare_ort_headers(
            build_dir=build_dir,
            download_dir=download_dir,
            config=config,
            ort_version=args.ort_version,
        )
    )
    _build_plugin_ep(
        ort_include_dir=ort_include_dir,
        native_build_dir=build_dir / "onnxruntime-ep-native",
    )
    if not args.reuse_tts_cpp_build:
        _prepare_tts_cpp_source(
            source_dir=tts_source_dir,
            repo=args.tts_cpp_repo,
            ref=args.tts_cpp_ref,
            ggml_repo=args.ggml_repo,
        )
        _build_tts_cpp(
            source_dir=tts_source_dir, build_dir=tts_build_dir, config=config
        )
    _build_bundle(
        output_dir=output_dir,
        tts_build_dir=tts_build_dir,
        config=config,
        ort_version=args.ort_version,
        tts_cpp_ref=args.tts_cpp_ref,
    )
    print(f"bundle: {output_dir}")
    print(f"engine env: AIVIS_ONNX_GGML_BUNDLE_DIR={output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)

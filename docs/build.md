# Build

This repository follows the standard ONNX Runtime Plugin EP shape:

- The Plugin EP native code is built with CMake.
- The public entry point is a platform script: `build.sh` or `build.ps1`.
- The script creates a redistributable runtime bundle with a manifest.
- downstream engine consumes the bundle instead of rebuilding this repository.

## One-command build

Linux / macOS:

```bash
./build.sh
```

Windows PowerShell:

```powershell
.\build.ps1
```

Windows Command Prompt:

```bat
build.bat
```

The default output is platform-specific:

```text
dist/style-bert-vits2-ggml-runtime-linux-x64/
dist/style-bert-vits2-ggml-runtime-macos-x64/
dist/style-bert-vits2-ggml-runtime-macos-arm64/
dist/style-bert-vits2-ggml-runtime-windows-x64/
```

## Reuse an existing TTS.cpp build

```bash
./build.sh --reuse-tts-cpp-build --tts-cpp-build-dir /path/to/TTS.cpp-build
```

The Windows wrappers accept the same arguments.

## Windows BuildTools 2026 fallback

The default Windows CMake preset targets Visual Studio 2022. On a machine that
only has Visual Studio 2026 BuildTools / MSVC v145, run from an x64 developer
environment and bypass the preset with Ninja:

```bat
set CMAKE_GENERATOR=Ninja
uv run python scripts\build_runtime_bundle.py --no-cmake-preset
```

When reusing an already-built TTS.cpp tree:

```bat
set CMAKE_GENERATOR=Ninja
uv run python scripts\build_runtime_bundle.py --no-cmake-preset --reuse-tts-cpp-build --tts-cpp-build-dir build\TTS.cpp-build
```

## Build options

The wrappers pass all arguments to `scripts/build_runtime_bundle.py`.

Useful options:

```bash
./build.sh --output-dir dist/style-bert-vits2-ggml-runtime-custom
./build.sh --ort-version 1.26.0
./build.sh --tts-cpp-ref <tts-cpp-commit-or-tag>
./build.sh --build-dir build
./build.sh --download-dir download
```

`--tts-cpp-ref` の既定値は `main` です。CI と release workflow は
Myoland/TTS.cpp の最新状態を常に build し、upstream 側の build breakage や
runtime regression を早期に検出する方針です。

一方で、bundle の `manifest.json` には実際に checkout された TTS.cpp commit
と ggml submodule commit が記録されます。過去の bundle を厳密に再現したい
場合は、manifest の `tts_cpp_ref` を `--tts-cpp-ref` に渡してください。

The Plugin EP itself is configured through `CMakePresets.json`. The Python script
uses the release preset by default and only falls back to explicit `cmake -S/-B`
arguments when a custom `--build-dir` is requested.

## Bundle contents

```text
dist/style-bert-vits2-ggml-runtime-<platform>/
├── manifest.json
├── lib/
│   ├── libtts.so / libtts.dylib / tts.dll
│   └── libggml*
└── onnxruntime_ep_style_bert_vits2_ggml/
    └── lib/
        └── libstyle_bert_vits2_ggml_onnx_ep.so / .dylib / .dll
```

`manifest.json` records the ONNX Runtime version, requested TTS.cpp ref, resolved
TTS.cpp commit, resolved ggml submodule commit, ggml repository, runtime ABI
version, GGUF schema version, and library checksums.

## Related workflows

- [Downstream Engine integration](engine-integration.md)
- [Benchmark reproduction](benchmark.md)
- [JP-BERT GGUF quantization](jp-bert-gguf-quantization.md)

## macOS arm64 release workflow

The repository includes a GitHub Actions workflow for publishing the macOS
arm64 runtime bundle as a GitHub Release asset.

- Workflow: `.github/workflows/release-macos-runtime.yml`
- Runner: `macos-14`
- Bundle: `dist/style-bert-vits2-ggml-runtime-macos-arm64/`
- Assets: `style-bert-vits2-ggml-runtime-macos-arm64-<tag>.tar.gz` and
  `.sha256`

Pull requests, pushes to `main`, and manual `workflow_dispatch` runs build and
validate the runtime bundle. Push a tag matching `v*` or `runtime-macos-v*` to
publish the tarball to the corresponding GitHub Release. The release workflow
validates `manifest.json` and library checksums before publishing.

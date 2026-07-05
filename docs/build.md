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

Local Linux builds use the Vulkan headers, loader, shader compiler, and CMake
discovery paths available on the host. For release-quality Linux artifacts, use
the GitHub Actions workflow described below or provide an equivalent current
Vulkan SDK locally. Older distro Vulkan headers or shader compilers can build a
valid bundle but may miss fast Vulkan paths such as NVIDIA `NV_coopmat2`.

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

Runtime bundles configure TTS.cpp/ggml with `GGML_NATIVE=OFF`. This keeps CI
artifacts portable across x64 machines instead of baking the GitHub Actions
runner CPU instruction set into redistributed DLLs.

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

## Linux x64 runtime workflow

The repository includes a GitHub Actions workflow for building and publishing
the Linux x64 runtime bundle.

- Workflow: `.github/workflows/linux-runtime.yml`
- Runner: `ubuntu-24.04`
- Bundle: `dist/style-bert-vits2-ggml-runtime-linux-x64/`
- Workflow artifact: `linux-runtime-bundle`
- Assets: `style-bert-vits2-ggml-runtime-linux-x64-<tag>.tar.gz` and
  `.sha256`

The Linux workflow intentionally keeps the runner on `ubuntu-24.04` so the
published binary keeps a conservative glibc/libstdc++ baseline. To avoid being
limited by the distro Vulkan toolchain, the workflow downloads a pinned LunarG
Vulkan SDK and uses its headers plus shader compiler during the build.

Current Linux CI Vulkan SDK pin:

```text
VULKAN_SDK_VERSION=1.4.350.1
VULKAN_SDK_SHA256=6cce33c7e5383814150c5041820769d93c65a1fd883002e5949b067045a07daa
```

The workflow extracts only the pieces needed by the build: `glslc`,
`glslangValidator`, `glslang`, Vulkan headers, and `libshaderc_shared.so.1`.
It then sets `VULKAN_SDK`, prepends the SDK to `CMAKE_PREFIX_PATH`, and adds the
SDK shader compiler to `PATH`.

Linux CI treats Vulkan feature coverage as part of the artifact contract. The
build must show shader compiler support for:

- `GL_NV_cooperative_matrix2`
- `GL_EXT_integer_dot_product`
- `GL_EXT_bfloat16`

The built `libggml-vulkan.so` must also contain `NV_coopmat2`. This check
prevents publishing a bundle that silently falls back to a slower Vulkan matrix
core path on RTX 30-series hardware.

Only tag pushes build the Linux runtime bundle. Push a tag matching `v*` or
`runtime-linux-v*` to build, validate, package, upload the short-lived workflow
artifact, and publish the tarball to the corresponding GitHub Release.

## Windows x64 runtime workflow

The repository includes a GitHub Actions workflow for building and validating
the Windows x64 runtime bundle.

- Workflow: `.github/workflows/windows-runtime.yml`
- Runner: `windows-2022`
- Bundle: `dist/style-bert-vits2-ggml-runtime-windows-x64/`
- Assets: `style-bert-vits2-ggml-runtime-windows-x64-<tag>.zip` and
  `.sha256`

The workflow installs a pinned Vulkan SDK shader compiler, then calls the same
local build entry point as developers: `.\build.ps1`. It validates
`manifest.json`, required DLLs, library checksums, and Plugin EP exports before
uploading the runtime bundle. Only tag pushes build the Windows runtime bundle.
Push a tag matching `v*` or `runtime-windows-v*` to publish the zip to the
corresponding GitHub Release.

## macOS arm64 release workflow

The repository includes a GitHub Actions workflow for publishing the macOS
arm64 runtime bundle as a GitHub Release asset.

- Workflow: `.github/workflows/release-macos-runtime.yml`
- Runner: `macos-26`
- Bundle: `dist/style-bert-vits2-ggml-runtime-macos-arm64/`
- Assets: `style-bert-vits2-ggml-runtime-macos-arm64-<tag>.tar.gz` and
  `.sha256`

Only tag pushes build and validate the macOS runtime bundle. Push a tag matching
`v*` or `runtime-macos-v*` to publish the tarball to the corresponding GitHub
Release. The release workflow validates `manifest.json` and library checksums
before publishing.

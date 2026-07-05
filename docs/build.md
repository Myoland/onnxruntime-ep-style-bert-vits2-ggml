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

`manifest.json` records the ONNX Runtime version, TTS.cpp ref, ggml repository,
runtime ABI version, GGUF schema version, and library checksums.

## Related workflows

- [Downstream Engine integration](engine-integration.md)
- [Benchmark reproduction](benchmark.md)
- [JP-BERT GGUF quantization](jp-bert-gguf-quantization.md)

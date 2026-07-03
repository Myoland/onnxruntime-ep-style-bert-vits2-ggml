# AivisSpeech Engine integration

This repository builds the ONNX Runtime Plugin EP package and the TTS.cpp runtime
sidecars. AivisSpeech Engine consumes the generated runtime bundle.

## Build the bundle

```bash
PLUGIN_REPO_DIR=<onnxruntime-ep-aivis-ggml checkout>

cd "$PLUGIN_REPO_DIR"
uv run python scripts/build_runtime_bundle.py
```

To reuse an existing TTS.cpp build:

```bash
uv run python scripts/build_runtime_bundle.py \
  --tts-cpp-build-dir /path/to/TTS.cpp-build \
  --reuse-tts-cpp-build
```

The default output is:

```text
dist/aivis-ggml-runtime/
```

## Package Engine

```bash
ENGINE_DIR=<AivisSpeech-Engine checkout>
PLUGIN_REPO_DIR=<onnxruntime-ep-aivis-ggml checkout>

cd "$ENGINE_DIR"
export AIVIS_ONNX_GGML_REQUIRED=1
export AIVIS_ONNX_GGML_BUNDLE_DIR="$PLUGIN_REPO_DIR/dist/aivis-ggml-runtime"
uv run --group build pyinstaller --noconfirm run.spec
```

The Engine package should contain:

```text
dist/run/lib/libtts.so
dist/run/lib/libggml*
dist/run/onnxruntime_ep_aivis_ggml/lib/libaivis_ggml_onnx_ep.so
```

Use the platform equivalent names on macOS and Windows.

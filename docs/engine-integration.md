# downstream engine integration

This repository builds the ONNX Runtime Plugin EP package and the TTS.cpp runtime
sidecars. downstream engine consumes the generated runtime bundle.

## Build the bundle

```bash
PLUGIN_REPO_DIR=<onnxruntime-ep-style-bert-vits2-ggml checkout>

cd "$PLUGIN_REPO_DIR"
./build.sh
```

To reuse an existing TTS.cpp build:

```bash
./build.sh --reuse-tts-cpp-build --tts-cpp-build-dir /path/to/TTS.cpp-build
```

The default output is:

```text
dist/style-bert-vits2-ggml-runtime-linux-x64/
```

## Package Engine

```bash
ENGINE_DIR=<downstream engine checkout>
PLUGIN_REPO_DIR=<onnxruntime-ep-style-bert-vits2-ggml checkout>

cd "$ENGINE_DIR"
export STYLE_BERT_VITS2_GGML_REQUIRED=1
export STYLE_BERT_VITS2_GGML_BUNDLE_DIR="$PLUGIN_REPO_DIR/dist/style-bert-vits2-ggml-runtime-linux-x64"
uv run --group build pyinstaller --noconfirm run.spec
```

Use the matching `dist/style-bert-vits2-ggml-runtime-<platform>/` directory on macOS and Windows.

The Engine package should contain:

```text
dist/run/lib/libtts.so
dist/run/lib/libggml*
dist/run/onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.so
```

Use the platform equivalent names on macOS and Windows.

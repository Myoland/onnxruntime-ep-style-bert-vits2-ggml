# downstream engine integration

This repository builds the ONNX Runtime Plugin EP package and the TTS.cpp runtime
sidecars. downstream engine consumes the generated runtime bundle.

## Use a CI or release bundle

For reviewer and packaging workflows, prefer a runtime bundle produced by this
repository's GitHub Actions or GitHub Release. This keeps the downstream engine
change small: Engine only receives a platform bundle and does not need to carry
TTS.cpp, ggml, Vulkan, or Plugin EP build logic.

After downloading and extracting the matching platform archive, point Engine at
the extracted bundle directory:

```bash
PLUGIN_BUNDLE_DIR=/path/to/style-bert-vits2-ggml-runtime-linux-x64
```

Use the matching `style-bert-vits2-ggml-runtime-<platform>/` directory on macOS
and Windows.

Linux CI bundles are built on `ubuntu-24.04` with a pinned current LunarG Vulkan
SDK toolchain. This preserves a conservative Linux binary baseline while still
building the Vulkan feature paths needed for NVIDIA `NV_coopmat2` performance.

## Build the bundle locally

```bash
PLUGIN_REPO_DIR=<onnxruntime-ep-style-bert-vits2-ggml checkout>

cd "$PLUGIN_REPO_DIR"
./build.sh
```

Local Linux builds use the host Vulkan SDK, headers, and shader compiler. For
performance numbers comparable to the CI Linux bundle, make sure the local
toolchain exposes the same Vulkan shader extension coverage documented in
[Build](build.md).

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

Use the matching `dist/style-bert-vits2-ggml-runtime-<platform>/` directory on
macOS and Windows.

The Engine package should contain:

```text
dist/run/lib/libtts.so
dist/run/lib/libggml*
dist/run/onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.so
```

Use the platform equivalent names on macOS and Windows.

## Benchmark

After packaging or preparing a local runtime bundle, use
[Benchmark reproduction](benchmark.md) to verify provider selection, RTF, PCM
comparison, and generated review artifacts.

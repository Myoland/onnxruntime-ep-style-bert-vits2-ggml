# ONNX Runtime EP Style-Bert-VITS2 GGML

Style-Bert-VITS2 の GGML 推論を ONNX Runtime Plugin Execution Provider
として呼び出すための runtime package です。

この repository は downstream engine から分離した次の責務を持ちます。

- ONNX Runtime Plugin EP native library
- AIVMX / ONNX Style-Bert-VITS2 synthesis model から GGUF cache を作る Python helper
- TTS.cpp runtime sidecar と Plugin EP をまとめた local bundle
- bundle manifest と再現可能な build 手順

downstream engine 側は `--onnx_provider ggml` が明示された場合だけこの package を
import し、`StyleBertVits2GgmlExecutionProvider` をフォールバック Provider より前に追加します。

## Documents

- [Architecture and design policy](docs/architecture.md)
- [Build](docs/build.md)
- [Downstream Engine integration](docs/engine-integration.md)
- [Benchmark reproduction](docs/benchmark.md)
- [JP-BERT GGUF quantization](docs/jp-bert-gguf-quantization.md)

## Build runtime bundle

Linux / macOS:

```bash
./build.sh
```

配布・レビュー用の runtime bundle は GitHub Actions でも build します。Linux x64 は
`ubuntu-24.04` の binary baseline を保ちながら pinned Vulkan SDK headers /
shader compiler を使い、macOS arm64 は `macos-26` で build します。詳細は
[docs/build.md](docs/build.md) を参照してください。

Windows PowerShell:

```powershell
.\build.ps1
```

既に TTS.cpp を build 済みの場合は、Plugin EP だけを再 build して既存の TTS.cpp
runtime を bundle にできます。

```bash
./build.sh --reuse-tts-cpp-build --tts-cpp-build-dir /path/to/TTS.cpp-build
```

Windows PowerShell では同じ引数を `.\build.ps1` に渡します。

既定の出力先は platform ごとに分かれます。

```text
dist/style-bert-vits2-ggml-runtime-linux-x64/
├── manifest.json
├── lib/
│   ├── libtts.so / libtts.dylib / tts.dll
│   └── libggml*
└── onnxruntime_ep_style_bert_vits2_ggml/
    └── lib/
        └── libstyle_bert_vits2_ggml_onnx_ep.so / .dylib / .dll
```

詳細は [docs/build.md](docs/build.md) を参照してください。

## 使い方

この package は downstream の Style-Bert-VITS2 engine から利用される runtime です。
それ自体が engine の entry point を置き換えるものではありません。Engine は GGML が
明示的に選択された場合だけ、この package を import します。

### 1. runtime bundle を取得する

利用する platform に合った GitHub Release bundle を取得します。

```text
style-bert-vits2-ggml-runtime-linux-x64-<tag>.tar.gz
style-bert-vits2-ggml-runtime-macos-arm64-<tag>.tar.gz
style-bert-vits2-ggml-runtime-windows-x64-<tag>.zip
```

local development では、自分で bundle を build できます。

```bash
./build.sh
```

展開済み、または local build 済みの bundle は次の構成になります。

```text
style-bert-vits2-ggml-runtime-<platform>/
├── manifest.json
├── lib/
│   ├── libtts.so / libtts.dylib / tts.dll
│   └── libggml*
└── onnxruntime_ep_style_bert_vits2_ggml/
    └── lib/
        └── libstyle_bert_vits2_ggml_onnx_ep.so / .dylib / .dll
```

### 2. Engine に bundle の場所を渡す

downstream engine を package または実行する際に、`STYLE_BERT_VITS2_GGML_BUNDLE_DIR`
で bundle directory を渡します。

```bash
export STYLE_BERT_VITS2_GGML_BUNDLE_DIR=/path/to/style-bert-vits2-ggml-runtime-linux-x64
```

packaging 時に bundle が必須であることを検証したい場合は、次も指定します。

```bash
export STYLE_BERT_VITS2_GGML_REQUIRED=1
```

### 3. Engine 側で GGML を有効にする

GGML は opt-in です。Engine の GGML provider option を指定します。

```bash
uv run python run.py --onnx_provider ggml
```

`--onnx_provider ggml` を指定しない場合、Engine は既存の ONNX Runtime provider の
挙動を維持します。

### 4. runtime 経路を検証する

Engine checkout から benchmark reproduction script を実行します。

```bash
ENGINE_DIR=<AivisSpeech-Engine checkout>
PLUGIN_REPO_DIR=<onnxruntime-ep-style-bert-vits2-ggml checkout>
PLUGIN_BUNDLE_DIR=/path/to/style-bert-vits2-ggml-runtime-linux-x64

cd "$ENGINE_DIR"
uv run python "$PLUGIN_REPO_DIR/scripts/reproduce_onnx_ggml_benchmark.py" \
  --engine-dir "$ENGINE_DIR" \
  --ggml-native-library-path "$PLUGIN_BUNDLE_DIR/lib/libtts.so" \
  --onnx-ep-library-path "$PLUGIN_BUNDLE_DIR/onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.so" \
  --library-dir "$PLUGIN_BUNDLE_DIR/lib"
```

出力された `summary.md` と `benchmark.log` で、
`StyleBertVits2GgmlExecutionProvider` が `CPUExecutionProvider` より前にあることを
確認します。

Linux の NVIDIA cooperative matrix 対応環境では、`benchmark.log` に次の表示がある
ことを確認します。

```text
matrix cores: NV_coopmat2
```

複数の Vulkan device が見える環境では、device を明示的に固定します。例:

```bash
MESA_VK_DEVICE_SELECT=1002:1900! uv run python "$PLUGIN_REPO_DIR/scripts/reproduce_onnx_ggml_benchmark.py" \
  --engine-dir "$ENGINE_DIR" \
  --ggml-native-library-path "$PLUGIN_BUNDLE_DIR/lib/libtts.so" \
  --onnx-ep-library-path "$PLUGIN_BUNDLE_DIR/onnxruntime_ep_style_bert_vits2_ggml/lib/libstyle_bert_vits2_ggml_onnx_ep.so" \
  --library-dir "$PLUGIN_BUNDLE_DIR/lib" \
  --ggml-vulkan-device 0
```

## Python helper

Python package は次の helper を公開します。

- `get_library_path() -> str`
- `get_ep_name() -> str`
- `get_ep_names() -> list[str]`
- `get_default_provider_options() -> dict[str, str]`

Engine は `onnxruntime_ep_style_bert_vits2_ggml.cache.prepare_ggml_cache` も import します。
これは、対応している AIVMX/ONNX synthesis model を、最初の ONNX session を
開く前に GGUF キャッシュへ変換するためです。

## Native Provider

native 共有ライブラリは ONNX Runtime Plugin EP の symbol を export します。

- `CreateEpFactories`
- `ReleaseEpFactory`

Provider name:

```text
StyleBertVits2GgmlExecutionProvider
```

主な provider option:

- `backend`: `vulkan`, `metal`, `cpu`
- `device`: バックエンド内の device id
- `precision`: `accurate`, `fast`
- `gguf_path`: 変換済み Style-Bert-VITS2 synthesis GGUF
- `jp_bert_gguf_path`: 変換済み Style-Bert-VITS2 JP-BERT GGUF
- `tts_cpp_library_path`: `tts_style_bert_vits2_*` を公開する TTS.cpp shared library
- `eager_load_model`: `0`, `1`
- `claim_synthesis_graph`: `0`, `1`
- `claim_jp_bert_graph`: `0`, `1`
- `n_threads`: TTS.cpp runtime の thread 数。`0` は runtime 既定値を使う

graph claim は opt-in です。`claim_synthesis_graph=1` または
`claim_jp_bert_graph=1` を指定しない限り、Provider は登録されますが、
実行はフォールバック ONNX Provider に残ります。

## 現在のスコープ

このブランチでは統合範囲を意図的に狭くしています。

- `--onnx_provider ggml` を選ばない限り、downstream engine の起動と推論経路は変えない。
- `--onnx_provider ggml` では `StyleBertVits2GgmlExecutionProvider` を strict mode で登録する。
- AIVMX/ONNX synthesis weights は ONNX session 作成前に local GGUF キャッシュへ変換する。
- synthesis GGUF の既定は FP32 とし、レビュー基準を保守的にする。
- JP-BERT は既定の F16 `linear` GGUF を使う。既存 tokenizer、日本語 frontend、
  `word2ph` 展開は Style-Bert-VITS2 側に残す。
- native EP は既知の full synthesis graph と JP-BERT graph だけを claim する。

native sidecar/backend の追加実験、ベンチマーク生成物、EPContext packaging、
汎用 ONNX-to-GGML compile はこのブランチの対象外です。

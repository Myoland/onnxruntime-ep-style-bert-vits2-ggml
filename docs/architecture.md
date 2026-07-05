# アーキテクチャと設計方針

このドキュメントでは、ONNX Runtime GGML Plugin EP の設計目標、責務分担、
推論フロー、メンテナンス境界、現在の制限を説明します。downstream engine の
メンテナが、この backend が何を行い、なぜこの構成になっているのか、
どのように benchmark を再現できるのか、また採用した場合にどの範囲の
メンテナンス責任が発生するのかを判断できることを目的にしています。

## 設計目標

この repository は、AivisSpeech Engine の既存の CPU / CUDA / DirectML 推論経路を
置き換えるものではありません。明示的に opt-in された場合だけ使う GGML backend を
提供します。

設計目標は次の四つです。

1. Engine 側の変更を小さく保つ。
   Engine は provider selection、session creation、strict validation だけを担当し、
   GGML / TTS.cpp の native build logic を直接持ちません。
2. 既定の推論経路を変えない。
   `--onnx_provider ggml` が指定された場合だけ Plugin EP を登録します。指定されて
   いない場合、既存の CPU / CUDA / DirectML の挙動は変わりません。
3. 性能と音声結果を再現可能にする。
   benchmark では model、text、style id、AIVMX SHA-256、GGUF profile、runtime
   bundle を固定し、raw JSON、provider evidence、WAV preview を生成します。
4. 複雑性をこの repository に集約する。
   GGML/TTS.cpp integration、GGUF cache、native sidecar、runtime bundle、build
   script、benchmark script はこの repository に置き、Engine 側に散らさないようにします。

## 非目標

現在の scope には、次の内容を含めません。

- GGML backend を既定 backend にすること。
- downstream engine が GGML / TTS.cpp の C++ build details を直接メンテナンスすること。
- raw benchmark artifact や GGUF cache を Engine repository に追加すること。
- 汎用 ONNX-to-GGML compiler を作ること。
- Android / mobile integration を含めること。
- binary distribution の最終方式をこの段階で決めること。

この backend は、すぐに Engine の配布方針を変えるものではなく、まず optional runtime
として検証できる形を目指しています。

## コンポーネント境界

Engine 側が担当する責務は次の通りです。

- CLI / environment option。例: `--onnx_provider ggml`
- ONNX provider selection
- model lifecycle
- ONNX session creation
- strict provider validation

この repository が担当する責務は次の通りです。

- ONNX Runtime Plugin EP native library
- TTS.cpp runtime sidecar
- AIVMX / ONNX synthesis model から GGUF cache への変換
- JP-BERT GGUF cache の準備
- runtime bundle build
- manifest / checksum
- benchmark reproduction script
- audio preview と benchmark documentation

この境界の目的は、Engine が GGML の内部実装や複数の native repository の build details
を理解しなくてもよいようにすることです。Engine は、検証可能な runtime bundle と
thin integration layer だけを扱います。

## 推論フロー

GGML backend を有効にした場合の推論フローは次の通りです。

1. Engine が `--onnx_provider ggml` を読み取る。
2. Engine がこの package を使って Plugin EP config を生成する。
3. この package が ONNX Runtime Plugin EP native library を登録する。
4. JP-BERT session の作成前に JP-BERT GGUF cache を準備する。
5. synthesis session の作成前に、AIVMX / ONNX weights から synthesis GGUF cache を準備する。
6. Plugin EP は、既知の JP-BERT graph と Style-Bert-VITS2 synthesis graph だけを claim する。
7. claim されなかった graph は fallback ONNX provider に残す。
8. Engine が session provider を検証し、`StyleBertVits2GgmlExecutionProvider` が
   実際に使われていることを確認する。

strict validation は重要です。native runtime の登録や graph claim に失敗した場合、
実際には CPU fallback で実行されている結果を GGML benchmark として扱わないためです。

## GGUF profile

既定 profile は保守的に設計しています。

- synthesis GGUF: FP32
- JP-BERT GGUF: F16 `linear`

synthesis を既定で FP32 にするのは、benchmark の基準線を説明しやすくし、
synthesis weight quantization を追加の変数にしないためです。

JP-BERT は既定で F16 `linear` を使います。これは FP32 JP-BERT と比べて明確に
容量を削減でき、deterministic comparison でも sample count が一致し、相関も
許容範囲に収まっているためです。Q8 / Q4 などのより積極的な profile は、出力長や
音声差分への影響が大きくなる可能性があるため、現時点では既定にしません。

## Benchmark 方針

benchmark では、少なくとも次の情報を明示します。

- model source
- AIVMX SHA-256
- style id
- 三つの測定テキスト
- ONNX CPU / 既存 EP / GGML EP の RTF
- provider evidence
- WAV preview

現在の benchmark document では、AivisHub の `まお` model を使い、三つの標準文を固定
しています。macOS M1 Pro では ONNX CPU と GGML Metal の RTF を併記し、WAV preview
も提供しています。

この方針により、「特定環境で高速だった」という抽象的な主張ではなく、reviewer が
model、text、provider evidence、音声出力を個別に確認できる形にします。

## Runtime bundle

build script は platform ごとの runtime bundle を生成します。

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

`manifest.json` には、ONNX Runtime version、要求された TTS.cpp ref、実際に
checkout された TTS.cpp commit、ggml submodule commit、ggml repository、runtime
ABI、GGUF schema、library checksum を記録します。これは native bundle の出所を
見えるようにし、reviewer や packager が runtime の内容を確認できるようにするためです。

## メンテナンス方針

この backend のメンテナンスコストは、できるだけ Engine repository ではなく、
この repository に集約します。

短期的には、次の方針を推奨します。

- Engine は optional integration だけを持つ。
- native runtime はこの repository で独立して更新する。
- benchmark result は runtime の更新に合わせて更新する。
- build / package の複雑性は runtime bundle に閉じ込める。
- 正式配布に進む場合は、bundle hosting、version pinning、CI build、release policy を
  別途議論する。

そのため、現在の設計は、upstream に native runtime 全体の即時メンテナンス責任を
求めるものではありません。まず Draft / experimental backend として評価しやすい
状態を作ることを重視しています。

## ONNX Runtime Plugin EP を使う理由

AivisSpeech Engine の既存 integration は ONNX Runtime session を中心に構成されています。
Plugin EP を使うと、この境界を保ったまま GGML backend を追加できます。

- Engine は引き続き ONNX session を作成する。
- GGML backend は provider として graph execution に参加する。
- fallback provider を残せる。
- 実際に使われた provider を session から検証できる。
- 既存の provider selection logic と共存しやすい。

Engine から TTS.cpp を直接呼び出す方式と比べると、Plugin EP は侵入範囲を抑えやすく、
既存の ONNX provider と同じレイヤーで扱える点が利点です。

## 現在の判断材料

この backend の目的は、既存 backend をすべて置き換えることではありません。CUDA 以外の
cross-platform native acceleration path を検証することです。特に Apple Silicon、
AMD / Intel GPU、または CUDA runtime を導入したくない環境で価値が出る可能性があります。

正式な配布経路に進めるかどうかは、今後の次の観点で判断します。

- benchmark がより多くの実機環境で安定して既存経路より有利か。
- native bundle の配布と version pinning が十分に単純か。
- GGUF cache のディスク使用量と初回変換コストが許容できるか。
- upstream が thin integration だけを持ち、native runtime の複雑性を独立 repository に
  残す形を受け入れられるか。

この document と benchmark は、これらの論点を具体的に議論できるようにするためのものです。

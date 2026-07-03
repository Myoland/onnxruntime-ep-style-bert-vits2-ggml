"""Tests for downstream PyInstaller runtime bundle helpers."""

from pathlib import Path

from onnxruntime_ep_style_bert_vits2_ggml import pyinstaller


def test_resolve_runtime_bundle_uses_windows_documented_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pyinstaller.sys, "platform", "win32")
    bundle_dir = tmp_path / "style-bert-vits2-ggml-runtime-windows-x64"
    tts_library_path = bundle_dir / "lib" / "tts.dll"
    ep_library_path = (
        bundle_dir
        / "onnxruntime_ep_style_bert_vits2_ggml"
        / "lib"
        / "style_bert_vits2_ggml_onnx_ep.dll"
    )
    tts_library_path.parent.mkdir(parents=True)
    ep_library_path.parent.mkdir(parents=True)
    tts_library_path.write_bytes(b"tts")
    ep_library_path.write_bytes(b"ep")

    resolved = pyinstaller.resolve_runtime_bundle(bundle_dir)

    assert resolved.tts_library_path == tts_library_path
    assert resolved.onnx_ep_library_path == ep_library_path
    assert resolved.library_dirs == (tts_library_path.parent, ep_library_path.parent)


def test_copy_runtime_bundle_copies_windows_sidecars(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pyinstaller.sys, "platform", "win32")
    bundle_dir = tmp_path / "bundle"
    target_dir = tmp_path / "dist" / "run"
    (bundle_dir / "lib").mkdir(parents=True)
    (bundle_dir / "onnxruntime_ep_style_bert_vits2_ggml" / "lib").mkdir(parents=True)
    (bundle_dir / "lib" / "tts.dll").write_bytes(b"tts")
    (bundle_dir / "lib" / "ggml.dll").write_bytes(b"ggml")
    (
        bundle_dir
        / "onnxruntime_ep_style_bert_vits2_ggml"
        / "lib"
        / "style_bert_vits2_ggml_onnx_ep.dll"
    ).write_bytes(b"ep")

    pyinstaller.copy_runtime_bundle(
        target_dir=target_dir,
        required=True,
        bundle_dirs=[bundle_dir],
    )

    assert (target_dir / "lib" / "tts.dll").read_bytes() == b"tts"
    assert (target_dir / "lib" / "ggml.dll").read_bytes() == b"ggml"
    assert (
        target_dir
        / "onnxruntime_ep_style_bert_vits2_ggml"
        / "lib"
        / "style_bert_vits2_ggml_onnx_ep.dll"
    ).read_bytes() == b"ep"

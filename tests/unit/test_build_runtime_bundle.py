"""Tests for runtime bundle build helpers."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_runtime_bundle import _extract_archive  # noqa: E402


def _add_tar_file(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def test_extract_archive_keeps_direct_tar_layout_with_dot_prefix(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "onnxruntime.tgz"
    output_dir = tmp_path / "onnxruntime"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_tar_file(
            archive, "./include/onnxruntime/core/session/onnxruntime_cxx_api.h", b"h"
        )
        _add_tar_file(archive, "./lib/libonnxruntime.dylib", b"lib")

    _extract_archive(archive_path, output_dir)

    assert output_dir.exists()
    assert (output_dir / ".extracted").read_text(encoding="utf-8") == "ok\n"
    assert (
        output_dir / "include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    ).read_bytes() == b"h"
    assert not (output_dir / "lib").exists()

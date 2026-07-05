"""Tests for runtime bundle build helpers."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_runtime_bundle import (  # noqa: E402
    _extract_archive,
    _resolve_git_ref,
    _source_refs_from_existing_tree,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        check=True,
    )


def _commit_file(repo: Path, name: str, content: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", name], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", name], check=True)
    return _git(repo, "rev-parse", "HEAD")


def _add_tar_file(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def test_resolve_git_ref_prefers_remote_tracking_branch(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    clone = tmp_path / "clone"
    _init_repo(origin)
    first_commit = _commit_file(origin, "sample.txt", "first\n")
    latest_commit = _commit_file(origin, "sample.txt", "latest\n")
    subprocess.run(["git", "clone", str(origin), str(clone)], check=True)
    subprocess.run(
        ["git", "-C", str(clone), "update-ref", "refs/heads/main", first_commit],
        check=True,
    )

    assert _resolve_git_ref(clone, "main") == latest_commit


def test_source_refs_from_existing_tree_records_tts_and_ggml_heads(
    tmp_path: Path,
) -> None:
    tts_cpp = tmp_path / "TTS.cpp"
    ggml = tts_cpp / "ggml"
    _init_repo(tts_cpp)
    tts_commit = _commit_file(tts_cpp, "tts.txt", "tts\n")
    _init_repo(ggml)
    ggml_commit = _commit_file(ggml, "ggml.txt", "ggml\n")

    refs = _source_refs_from_existing_tree(tts_cpp, "main")

    assert refs.tts_cpp_requested_ref == "main"
    assert refs.tts_cpp_ref == tts_commit
    assert refs.ggml_ref == ggml_commit


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

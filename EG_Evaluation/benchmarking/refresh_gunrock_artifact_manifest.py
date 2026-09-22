#!/usr/bin/env python3
"""Bind the exact local Gunrock patches to the evaluated executables."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "gunrock_artifact_manifest.json"
HAORANDU = HERE.parents[2]
GUNROCK_ROOT = HAORANDU / "EG_Evaluation"
MAINTAINED = GUNROCK_ROOT / "gunrock_latest"
LEGACY = GUNROCK_ROOT / "gunrock_legacy_master"
MAINTAINED_BIN = MAINTAINED / "build_cuda132_a100_migrated" / "bin"
LEGACY_BIN = LEGACY / "build_lcc_result_cuda128_clean" / "bin"

COMMON_MAINTAINED = [
    "include/gunrock/framework/operators/advance/helpers.hxx",
    "include/gunrock/framework/operators/advance/merge_path_v2.hxx",
    "include/gunrock/framework/operators/configs.hxx",
    "include/gunrock/graph/csr.hxx",
]

ALIGNED_E2E_TIMER = "examples/algorithms/aligned_e2e.hxx"

RELEVANT_FILES = {
    "pr": [
        "examples/algorithms/pr/pr.cu",
        ALIGNED_E2E_TIMER,
        "include/gunrock/io/parameters.hxx",
        *COMMON_MAINTAINED,
    ],
    "mst": [
        "examples/algorithms/mst/mst.cu",
        ALIGNED_E2E_TIMER,
        *COMMON_MAINTAINED,
    ],
    "bfs": [
        "examples/algorithms/bfs/bfs.cu",
        ALIGNED_E2E_TIMER,
        "include/gunrock/io/parameters.hxx",
        *COMMON_MAINTAINED,
    ],
    "sssp": [
        "examples/algorithms/sssp/sssp.cu",
        ALIGNED_E2E_TIMER,
        "include/gunrock/io/parameters.hxx",
        *COMMON_MAINTAINED,
    ],
    "kcore": [
        "examples/algorithms/kcore/kcore.cu",
        ALIGNED_E2E_TIMER,
        *COMMON_MAINTAINED,
    ],
    "bc": [
        "examples/algorithms/bc/bc.cu",
        ALIGNED_E2E_TIMER,
        "include/gunrock/algorithms/bc.hxx",
        "include/gunrock/io/parameters.hxx",
        *COMMON_MAINTAINED,
    ],
    "lcc": [
        "CMakeLists.txt",
        "cmake/CPM.cmake",
        "examples/lcc/test_lcc.cu",
        "gunrock/app/lcc/lcc_app.cu",
        "gunrock/app/lcc/lcc_enactor.cuh",
    ],
}


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def git_text(repo: Path, *args: str) -> str:
    return git_bytes(repo, *args).decode("utf-8", errors="replace").strip()


def source_record(repo: Path, directory_name: str) -> dict:
    tracked_diff = git_bytes(
        repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--"
    )
    tracked_modified = [
        line
        for line in git_text(repo, "diff", "--name-only", "HEAD", "--").splitlines()
        if line
    ]
    untracked = [
        line
        for line in git_text(
            repo, "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        if line
        and not any(
            part.startswith("build") or part in {"cuda_overlay", "cuda_overlays"}
            for part in Path(line).parts
        )
        and (
            Path(line).name == "CMakeLists.txt"
            or Path(line).suffix
            in {".cmake", ".cu", ".cuh", ".h", ".hpp", ".hxx", ".txt"}
        )
    ]
    modified = sorted(set(tracked_modified + untracked))
    diff_digest = hashlib.sha256()
    diff_digest.update(tracked_diff)
    for relative in sorted(untracked):
        path = repo / relative
        if not path.is_file():
            continue
        diff_digest.update(b"\0UNTRACKED\0")
        diff_digest.update(relative.encode("utf-8"))
        diff_digest.update(b"\0")
        diff_digest.update(path.read_bytes())
    mtimes = [(repo / relative).stat().st_mtime for relative in modified]
    return {
        "source_directory_name": directory_name,
        "source_remote": git_text(repo, "remote", "get-url", "origin"),
        "source_commit": git_text(repo, "rev-parse", "HEAD"),
        "source_version": git_text(repo, "describe", "--tags", "--always", "--dirty"),
        "source_tracked_dirty": bool(modified),
        "source_diff_sha256": diff_digest.hexdigest(),
        "source_modified_files": modified,
        "source_modified_files_latest_mtime_epoch": max(mtimes, default=0.0),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_metadata(repo: Path, files: list[str]) -> dict:
    existing = [relative for relative in files if (repo / relative).is_file()]
    mtimes = [(repo / relative).stat().st_mtime for relative in existing]
    return {
        "source_relevant_modified_files": existing,
        "source_relevant_file_sha256": {
            relative: sha256(repo / relative) for relative in existing
        },
        "source_relevant_files_latest_mtime_epoch": max(mtimes, default=0.0),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["sources"] = {
        "maintained-v2": source_record(MAINTAINED, "gunrock_latest"),
        "legacy-v1": source_record(LEGACY, "gunrock_legacy_master"),
    }

    paths = {
        "pr": MAINTAINED_BIN / "pr",
        "mst": MAINTAINED_BIN / "mst",
        "bfs": MAINTAINED_BIN / "bfs",
        "sssp": MAINTAINED_BIN / "sssp",
        "kcore": MAINTAINED_BIN / "kcore",
        "bc": MAINTAINED_BIN / "bc",
        "lcc": LEGACY_BIN / "lcc",
    }
    executables = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        source_id = "legacy-v1" if name == "lcc" else "maintained-v2"
        repo = LEGACY if name == "lcc" else MAINTAINED
        executables[name] = {
            "source": source_id,
            "sha256": sha256(path),
            **relevant_metadata(repo, RELEVANT_FILES[name]),
        }
    manifest["executables"] = executables
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(MANIFEST)


if __name__ == "__main__":
    main()

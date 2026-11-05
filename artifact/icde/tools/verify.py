#!/usr/bin/env python3
"""Check the declared ICDE release and the complete source tree; no GPU imports."""
import argparse, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "artifact/icde/manifest.json").read_text())
    observed = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts and str(p.relative_to(root)) != "artifact/icde/manifest.json"}
    if observed != set(manifest["files"]):
        raise ValueError("Undeclared or missing release members")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Release member escapes package")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("Release member changed: " + name)
    source = root / "Easy-Graph"
    actual = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Source symlink: " + str(path.relative_to(source)))
        if path.is_file():
            actual[str(path.relative_to(source))] = sha(path)
    origin = json.loads((root / "artifact/icde/provenance/origin.json").read_text())
    omitted = json.loads((root / "artifact/icde/provenance/omitted.json").read_text())
    current = json.loads((root / "artifact/icde/provenance/current_source.json").read_text())
    delta = json.loads((root / "artifact/icde/provenance/candidate_delta.json").read_text())
    expected = {name: h for name, h in origin["files"].items() if name not in omitted}
    for row in delta["files"]:
        if expected.get(row["path"]) != row["base_sha256"]:
            raise ValueError("Patch base differs from source3 origin")
        expected[row["path"]] = row["current_sha256"]
    records = {name: {"sha256": value, "bytes": (source / name).stat().st_size} for name, value in actual.items()}
    current_digest = digest(records) if current.get("digest_scheme") == "path-to-sha256-and-bytes" else digest(actual)
    if current.get("file_records", records) != records:
        raise ValueError("Current source file-record manifest differs")
    if (actual != expected or actual != current["files"]
            or current_digest != manifest["current_source_sha256"]
            or current["source_sha256"] != manifest["current_source_sha256"]
            or delta["current_source_sha256"] != manifest["current_source_sha256"]
            or digest(origin["files"]) != manifest["base_source_sha256"]
            or origin["source_sha256"] != manifest["base_source_sha256"]):
        raise ValueError("Current source/base/patch lineage differs")
    return manifest
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = verify(args.root)
    print(json.dumps({"status": "PASS_SOURCE_AND_RELEASE_BYTES", "source_files": manifest["source_files"], "source_sha256": manifest["current_source_sha256"], "base_source_sha256": manifest["base_source_sha256"], "gpu_executed": False, "gpu_qualification": manifest["current_qualification"]["gpu_status"], "new_campaign_complete": manifest["new_campaign_complete"]}))
if __name__ == "__main__":
    main()

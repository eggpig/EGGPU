# Dataset Reproduction Policy

The benchmark suite currently uses 19 processed edge-list files under this
directory.  Their paths, sizes, SHA-256 checksums, and public source hints are
recorded in `MANIFEST_20260609.tsv`.

For a clean GitHub artifact, prefer one of these two policies:

1. Small artifact repository: track this `DATASETS.md` file, the manifest, and
   any tiny smoke-test datasets; fetch or stage the larger public datasets
   before running the full benchmark.
2. Fully self-contained artifact repository: track all processed edge-list
   files.  The current processed suite is about 212 MiB and the largest file is
   below GitHub's single-file limit, but this makes clones heavier.

The benchmark scripts expect the processed files at their listed relative
paths.  After downloading or copying files into place, verify them from
`EG_Evaluation/datasets` with:

```bash
tail -n +2 MANIFEST_20260609.tsv | while IFS=$'\t' read -r path graph_type size sha source; do
  test -f "$path" || { echo "missing $path"; exit 1; }
  actual_size="$(stat -c '%s' "$path")"
  test "$actual_size" = "$size" || { echo "size mismatch $path"; exit 1; }
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual_sha" = "$sha" || { echo "sha mismatch $path"; exit 1; }
done
```

Do not commit `benchmarking/results/`, raw compressed downloads, generated
MatrixMarket conversions, or third-party baseline source trees.  Those are
recreated by the evaluation workflow or documented as external dependencies.

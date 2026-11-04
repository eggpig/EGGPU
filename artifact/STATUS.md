# Current implementation and artifact status

Updated 2026-11-04. The current source is `Easy-Graph/`; the single current artifact entry is [icde/README.md](icde/README.md), with [machine-readable status](icde/campaign_status.json) and a [source manifest](icde/manifest.json).

The current candidate combines the source3 single/multi-GPU implementation with the reviewed Python capability-dispatch patch. The capability-dispatch patch passed 46 CPU checks, and the closure-LRU control passed a separate 10-test CPU suite. The current clean build and GPU semantic qualification are separate steps; GPU qualification is pending in this publication checkpoint. Historical source3 binary qualification does not qualify a new source/binary. No new seven-dataset performance matrix or target leading rate is claimed here.

Use `python -B artifact/icde/tools/reproduce.py status`, `verify`, and the explicit build/input/qualification commands documented by the current guide. There is not yet a complete portable E1–E6 runner. Source, input, baseline, correctness and timing identities must be bound before formal results replace historical evidence.

The older root `scripts/build_eggpu.sh`, `scripts/run_smoke.sh`, earlier artifact architecture/correctness notes, and `EG_Evaluation/` describe the VLDB-era release and evaluation source. They are retained for provenance and are not the entry points or result authority for the current ICDE campaign. Original author implementations and historical raw measurements are not deleted merely because the presentation changes.

The current suite is US/EP/LJ/OR/IN/UK/TW. Public input utilities and their inherited-data/exporter prerequisites are described in the guide; they do not yet constitute fresh-machine reproduction of all seven graphs. Mechanism and framework illustrations remain reserved slots for the collaborator's final artwork.

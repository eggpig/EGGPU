# Current artifact status

The only current entry is [artifact/icde](icde/README.md); run `python -B artifact/icde/tools/reproduce.py status` from the repository root. Source is `arena_output_ownership_v2`, SHA256 `cb14360e2dfb934b8a28f9875fb80639ce4503bc6ed27956201b18e9edec6869` (path-to-SHA256-and-byte-size digest).

Current-source SG qualification has passed 33 semantic/owned-output checks and 16 corrected CSR/global-L1 tiny configurations. Real-graph targets and other workflows are qualifying separately. All seven original-source canonical datasets are ready; see [input completion](icde/inputs/completion.json). **No formal E1–E6 measurement matrix or measured SOTA coverage is published in this checkpoint.** State-v4 and external continuous-timing corrections have separate pending GPU gates. Check [campaign_status.json](icde/campaign_status.json) and [runner pointers](icde/campaign/latest.json) for each scope.

The old `EG_Evaluation`, earlier artifact notes, and old top-level build/smoke scripts remain historical source. They do not establish the current source, dataset, timing, or correctness contract. The source3 origin and first public commit `5207c94631e191094079f94e05ee08eabd108fa8` remain in the provenance chain; earlier results must not be relabeled as current.

**PR protocol correction:** the packaged SG cuGraph adapter uses global L1 epsilon `1e-9` directly, matching the installed cuGraph implementation. Earlier SG cuGraph PR and cuGraph 16-request timing records remain historical and are not comparable under this corrected protocol. The current source and corrected CSR runner have passed all 16 SG tiny configurations; other exact runner/real-graph gates remain separate. See [the correction record](icde/provenance/pr_protocol_correction.json).

The state planner v3 and unweighted-PR control v2 preserve the selected Python environment and bind its runtime identity. They supersede the earlier planners that resolved the interpreter symlink incorrectly. Their CPU admission checks passed; this does not replace GPU or formal qualification.

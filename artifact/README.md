# EGGPU framework artifact

This directory explains how the released implementation corresponds to the
system design and how to verify its returned values.

- [`ARCHITECTURE.md`](ARCHITECTURE.md): request path and retained state.
- [`FUNCTION_SUPPORT.md`](FUNCTION_SUPPORT.md): 16 supported public functions.
- [`PAPER_CODE_MAP.md`](PAPER_CODE_MAP.md): design concept to source location.
- [`CORRECTNESS.md`](CORRECTNESS.md): deterministic smoke and correctness tests.
- [`datasets/`](datasets/): dataset provenance and release boundaries.

The artifact contains no performance results or measurement harness. Run the
root-level build and correctness commands before inspecting larger inputs.

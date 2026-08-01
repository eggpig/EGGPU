# Architecture

## Public-call path

1. A normal EasyGraph function checks `EASYGRAPH_ENABLE_GPU` and dispatches a
   supported call to `easygraph.utils.gpu_eggpu_backend`.
2. The backend obtains graph state tied to the graph's mutation generation.
   Node identities, native graph objects, and compatible sparse views remain
   reusable while that generation is unchanged.
3. The C++ binding acquires device CSR arrays for the active graph identity,
   CUDA device, relation, and weight layout. The bounded per-device registry
   reuses capacity and evicts least-recently-used entries when needed.
4. A function-specific CUDA path runs over the retained state. Traversal,
   weighted relaxation, component labeling, and neighborhood statistics share
   common storage and execution infrastructure without sharing mutable
   algorithm state.
5. Dense native output is validated and exposed through the documented
   EasyGraph return form. Dense-backed mappings defer Python object creation;
   component sets and graph results are materialized only when required.

## State validity

`Graph`, `DiGraph`, and their multigraph variants maintain a monotonic mutation
generation. Supported topology and edge-attribute mutations clear graph-derived
caches and advance the generation. Repeated read-only calls can therefore reuse
prepared state, while a mutation forces reconstruction before the next call.

For inputs too large to materialize as Python adjacency dictionaries,
`EGGPUBulkGraph` loads an immutable, zero-based CSR manifest into the same native
execution path. The bulk loader checks format, dtype, size, and optional
projection/weight metadata before constructing the native handle.

## Failure behavior

GPU execution is opt-in. With `EASYGRAPH_GPU_STRICT_ERRORS=TRUE`, a failed or
unsupported accelerated path raises an exception. This mode is used by the
correctness smoke so a CPU fallback cannot be mistaken for a GPU result.

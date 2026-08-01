# Dataset identities

The framework repository does not include raw graph files. The accompanying
`manifest.tsv` records the processed graph size, source catalog, input filename,
and SHA-256 digest used by the paper. These fields identify the evaluated inputs
without redistributing third-party data.

All inputs retain the observed vertices, remove self-loops, and deduplicate
directed arcs or undirected endpoint pairs. An undirected edge is counted once;
a directed edge is counted once per ordered pair. The two large inputs use the
following representations:

- **com-Orkut** uses the SNAP largest connected component. Labels are mapped
  deterministically to contiguous zero-based identifiers before constructing a
  symmetric compressed sparse row (CSR) representation.
- **GAP-twitter** uses the SuiteSparse/GAP matrix and retains every matrix row,
  including isolated vertices. EGGPU uses its directed sparsity pattern with
  implicit unit weights.

Download third-party graphs from the source pages in `manifest.tsv`, apply the
normalization above, and compare the resulting input digest before running the
artifact. The project-generated ER-100k graph is identified by its processed
size and digest; its graph file is not part of this framework repository.

Source catalogs:

- SNAP: <https://snap.stanford.edu/data/>
- SuiteSparse Matrix Collection: <https://sparse.tamu.edu/>

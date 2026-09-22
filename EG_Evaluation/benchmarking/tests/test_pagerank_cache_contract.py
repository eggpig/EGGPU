from pathlib import Path


PAGERANK_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "Easy-Graph"
    / "gpu_easygraph"
    / "functions"
    / "centrality"
    / "pagerank.cu"
)


def test_pagerank_transition_cache_is_parameter_aware():
    source = PAGERANK_SOURCE.read_text(encoding="utf-8")
    assert "std::uint64_t sig_alpha" in source
    assert "ws.sig_alpha != bitcast_u64(alpha)" in source
    assert "ws.sig_alpha = bitcast_u64(alpha)" in source

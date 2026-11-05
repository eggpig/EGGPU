"""Exact frozen RNG/selection recipe; scalar CSR searches preserve int32 dtype."""
from pathlib import Path
import json

def positions(root, rho, locality, seed):
    import numpy as np
    meta = json.loads((root / 'canonical.json').read_text())
    n = meta['num_nodes']
    m = meta['num_entries']
    rng = np.random.default_rng(seed)
    count = int(m * rho)
    count = count if meta['directed'] else 2 * (count // 2)
    if count == 0:
        return np.empty(0, dtype='<i8')
    if count == m:
        return np.arange(m, dtype='<i8')
    if meta['directed']:
        if locality == 'uniform':
            return np.sort(rng.choice(m, count, replace=False)).astype('<i8')
        start = int(rng.integers(m - count + 1))
        return np.arange(start, start + count, dtype='<i8')
    off = np.memmap(root / 'offsets.bin', dtype='<i4', mode='r', shape=(n + 1,))
    ind = np.memmap(root / 'indices.bin', dtype='<i4', mode='r', shape=(m,))
    chosen = set()
    cursor = int(rng.integers(m))
    while len(chosen) < count:
        pos = int(rng.integers(m)) if locality == 'uniform' else cursor % m
        cursor += 1
        if pos in chosen:
            continue
        u = int(np.searchsorted(off, np.asarray(pos, dtype=off.dtype), side='right') - 1)
        v = int(ind[pos])
        a, b = (int(off[v]), int(off[v + 1]))
        reverse = a + int(np.searchsorted(ind[a:b], np.asarray(u, dtype=ind.dtype)))
        if reverse >= b or ind[reverse] != u:
            raise ValueError('Undirected reverse arc absent')
        chosen.update((pos, reverse))
    return np.array(sorted(chosen), dtype='<i8')

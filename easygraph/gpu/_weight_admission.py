"""One bounded pass over immutable FP64 bytes for identity and value checks.

This is not a trusted-admission cache. Every call hashes the complete payload;
error reporting remains with the caller so byte-identity failures take priority.
The native importer still owns and structurally validates its later read; the
existing immutable-file contract remains. Structural import is not a second SHA.
"""
from pathlib import Path
import hashlib
import numpy as np

# Keep the previous validation's 2**18-element chunk boundary and error order.
_WEIGHT_CHUNK_BYTES = (1 << 18) * 8


def scan_weight_payload(path, *, certify=False):
    digest = hashlib.sha256()
    error = None
    maximum = 0
    count = 0
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(_WEIGHT_CHUNK_BYTES), b''):
            digest.update(payload)
            if error is not None:
                continue
            if len(payload) % 8:
                # A changed/truncated file must fail identity before a value
                # interpretation is attempted. Normal admitted sizes are / 8.
                error = 'Snapshot payload identity mismatch'
                continue
            block = np.frombuffer(payload, dtype='<f8')
            if not np.all(np.isfinite(block)) or np.any(block < 0):
                error = 'Weights must be finite nonnegative f64'
                continue
            if certify:
                if np.any(block > 255) or np.any(block != np.floor(block)):
                    error = 'Static compact SSSP requires exact integer weights in [0,255]'
                    continue
                maximum = max(maximum, int(block.max(initial=0)))
                count += len(block)
    return digest.hexdigest(), error, maximum, count

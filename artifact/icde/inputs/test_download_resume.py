"""Content-verified range resume across explicit connection-count changes."""
from pathlib import Path
from unittest.mock import patch
import io
import json

import pytest

from range_download import download_file, RangeNotSupported


def test_four_to_eight_resume_skips_completed_ranges(tmp_path):
    payload = bytes(range(256)) * 24576
    url = 'https://fixture/object'; destination = tmp_path / 'source'
    ranges = [[i, min(i + (2 << 20), len(payload))] for i in range(0, len(payload), 2 << 20)]
    partial = destination.with_name('source.partial')
    with partial.open('wb') as stream:
        stream.write(payload[:2 << 20]); stream.truncate(len(payload))
    state = dict(url=url, total_bytes=len(payload), identity={'etag': 'fixed', 'last_modified': 'fixed'},
                 initial_prefix_bytes=0, ranges=ranges, completed=[0], downloaded_bytes=2 << 20)
    destination.with_name('source.ranges.json').write_text(json.dumps(state))
    requested = []
    class Reply(io.BytesIO):
        status = 206
        def __init__(self, begin, end):
            super().__init__(payload[begin:end + 1])
            self.headers = {'Content-Range': f'bytes {begin}-{end}/{len(payload)}',
                            'ETag': 'fixed', 'Last-Modified': 'fixed'}
    def request(req, timeout=None):
        begin, end = map(int, req.headers['Range'].split('=')[1].split('-'))
        requested.append((begin, end)); return Reply(begin, end)
    with patch('urllib.request.urlopen', request):
        result = download_file(url, destination, retries=0, connections=8)
    assert destination.read_bytes() == payload and result['connections'] == 8
    assert (0, (2 << 20) - 1) not in requested
    assert requested.count((0, 0)) == 1  # The small source-identity probe remains mandatory.
    assert json.loads(destination.with_name('source.ranges.json').read_text())['status'] == 'PASS_COMPLETE_RANGES'


@pytest.mark.parametrize('connections', [0, 1, 16, True, '8', 4.0])
def test_unsupported_connection_counts_reject_before_network(tmp_path, connections):
    with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
        with pytest.raises(ValueError): download_file('https://fixture', tmp_path / 'x', connections=connections)


def test_nonrange_rejected_before_preallocation(tmp_path):
    class Reply(io.BytesIO):
        status = 200; headers = {}
    with patch('urllib.request.urlopen', return_value=Reply(b'full response')):
        with pytest.raises(RangeNotSupported): download_file('https://fixture', tmp_path / 'x')
    assert not (tmp_path / 'x.partial').exists()

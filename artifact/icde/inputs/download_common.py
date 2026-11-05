"""Fetch only the three missing frozen inputs; no GPU or normalization."""
import argparse, hashlib, http.client, json, re, shutil, time
from pathlib import Path
import urllib.error, urllib.parse, urllib.request

def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def download_file(url, dest, timeout=45, retries=4):
    """Resume only when HTTP Content-Range proves the returned start offset."""
    if dest.exists():
        return {"status": "EXISTING_REQUIRES_MD5", "url": url, "bytes": dest.stat().st_size}
    part = dest.with_name(dest.name + ".partial")
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "EGGPU-public-artifact/2", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes=" + str(offset) + "-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as source:
                status = source.status
                if status not in (200, 206):
                    raise ValueError("Unexpected download status " + str(status))
                length = source.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", source.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) < offset:
                        raise ValueError("Invalid or mismatched resume Content-Range")
                    range_length = int(match[2]) - offset + 1
                    if expected is not None and expected != range_length:
                        raise ValueError("Content-Length and Content-Range disagree")
                    expected = range_length
                    mode = "ab" if offset else "wb"
                    total = None if match[3] == "*" else int(match[3])
                else:
                    # Server may legally ignore Range: restart, never append a
                    # complete 200 response to existing partial content.
                    mode, offset, total = "wb", 0, expected
                received = 0
                with part.open(mode) as target:
                    while True:
                        block = source.read(8 << 20)
                        if not block:
                            break
                        target.write(block)
                        received += len(block)
                if expected is not None and received != expected:
                    raise OSError("Incomplete response: expected %d, received %d" % (expected, received))
                if total is not None and part.stat().st_size != total:
                    raise OSError("Incomplete object after partial response")
            part.replace(dest)
            return {"status": "DOWNLOADED_REQUIRES_MD5", "url": url,
                    "bytes": dest.stat().st_size, "attempts": attempt + 1, "resumed_from": offset}
        except urllib.error.HTTPError as error:
            match = re.fullmatch(r"bytes \*/(\d+)", error.headers.get("Content-Range", ""))
            if error.code == 416 and match and offset == int(match[1]) and offset > 0:
                part.replace(dest)
                return {"status": "COMPLETE_PARTIAL_REQUIRES_MD5", "url": url, "bytes": offset}
            if attempt == retries or error.code not in (408, 429, 500, 502, 503, 504):
                raise
        except (OSError, http.client.IncompleteRead):
            if attempt == retries:
                raise
        time.sleep(min(2 ** attempt, 8))

sequential_download_file = download_file
from range_download import download_file as checked_range_download, RangeNotSupported

def download_file(url, dest, timeout=45, retries=4, *, connections=4):
    try:
        return checked_range_download(url, dest, timeout, retries, connections=connections)
    except RangeNotSupported:
        if dest.with_name(dest.name + '.ranges.json').exists():
            raise RuntimeError('Existing range progress cannot be treated as a sequential prefix')
        return sequential_download_file(url, dest, timeout, retries)


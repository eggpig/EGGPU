#!/usr/bin/env python3
"""Compatibility entry for the current original-source downloader; see fetch.py."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('fetch.py')), run_name='__main__')

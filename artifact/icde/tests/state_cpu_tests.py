#!/usr/bin/env python3
"""Compatibility entry; current state tests live with the campaign protocol."""
from pathlib import Path
import runpy
import sys

campaign = Path(__file__).resolve().parents[1]/'campaign'
sys.path.insert(0, str(campaign))
runpy.run_path(str(campaign/'state_cpu_tests.py'), run_name='__main__')

"""Shared pytest configuration and fixtures for the football-predictor test suite."""

import os
import sys

# Ensure the repo root is on sys.path so `footy` is importable regardless of
# how pytest is invoked (e.g. from within the tests/ directory).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

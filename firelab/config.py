"""Paths and run-time settings. Everything is overridable with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# The package is installed editable, so the repo root is two levels above this file.
ROOT = Path(os.environ.get("FIRELAB_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RECORD_DIR = ROOT / "record"
PLANS_DIR = RECORD_DIR / "plans"
RUNS_DIR = RECORD_DIR / "runs"
RESULTS_DIR = ROOT / "results"
DB_PATH = RECORD_DIR / "record.db"

LANDSCAPE = os.environ.get("FIRELAB_LANDSCAPE", "synthetic_v1")

# Lab-wide simulation budget (one simulation = one plan x one ignition x one weather scenario).
MAX_SIMS = int(os.environ.get("FIRELAB_MAX_SIMS", "20000"))
# Largest single ensemble a tool call may request, so one call cannot eat the whole budget.
MAX_SIMS_PER_CALL = int(os.environ.get("FIRELAB_MAX_SIMS_PER_CALL", "6000"))
MAX_BUDGET_PCT = 30.0

# "policy": Omnigent's ASK policy is the approval step, so the tool records the approval.
# "manual": the tool records a pending recommendation; a human runs `firelab approve <id>`.
APPROVAL_MODE = os.environ.get("FIRELAB_APPROVAL_MODE", "policy")
APPROVER = os.environ.get("FIRELAB_APPROVER", "human scientist (Omnigent ASK)")

VERIFY_CITATIONS = os.environ.get("FIRELAB_VERIFY_CITATIONS", "1") == "1"
USE_MLFLOW = os.environ.get("FIRELAB_MLFLOW", "1") == "1"
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "")

SIMULATOR_VERSION = "firelab-ca-0.1"


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, RECORD_DIR, PLANS_DIR, RUNS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

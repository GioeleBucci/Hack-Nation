"""Isolate tests: temp lab root, small landscape, no network, no MLflow.

This runs before any test imports firelab, so firelab.config picks these values up.
"""

import os
import tempfile

os.environ["FIRELAB_ROOT"] = tempfile.mkdtemp(prefix="firelab-test-")
os.environ["FIRELAB_LANDSCAPE"] = "synthetic_small"
os.environ["FIRELAB_VERIFY_CITATIONS"] = "0"
os.environ["FIRELAB_MLFLOW"] = "0"
os.environ["FIRELAB_APPROVAL_MODE"] = "manual"

"""No-treatment reference maps used by the burn_prob and path_centrality strategies.

Built once per landscape from a fixed no-treatment ensemble on the ``reference`` ignition
set and the ``train`` weather set, then cached in ``data/``. It plays the role of an
existing burn-probability product: part of the frozen input data, not of the lab's
experiments, so it does not count against the lab's simulation budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from firelab import config
from firelab.landscape import load_landscape
from firelab.scenarios import ignition_points, weather_scenarios
from firelab.simulator import run_ensemble

REF_IGNITIONS = 100
REF_WEATHER = 5
REF_SEED = 12345


@dataclass(frozen=True, eq=False)
class Reference:
    burn_prob: np.ndarray    # (H, W) fraction of reference fires that burned the cell
    centrality: np.ndarray   # (H, W) in [0, 1]: how often the cell passed fire on, weighted by fire size
    n_sims: int


@lru_cache(maxsize=4)
def load_reference(landscape_name: str | None = None) -> Reference:
    ls = load_landscape(landscape_name)
    path = config.DATA_DIR / f"reference_{ls.name}_{ls.fingerprint()}.npz"
    if path.exists():
        with np.load(path) as z:
            return Reference(z["burn_prob"], z["centrality"], int(z["n_sims"]))

    out = run_ensemble(ls, np.zeros((1, *ls.shape), dtype=bool),
                       ignition_points(ls, "reference", REF_IGNITIONS),
                       weather_scenarios("train", REF_WEATHER), seed=REF_SEED, track_spread=True)
    n_sims = REF_IGNITIONS * REF_WEATHER
    burn_prob = (out.burn_count[0] / n_sims).astype(np.float32)
    centrality = out.centrality[0]
    centrality = (centrality / centrality.max() if centrality.max() > 0 else centrality).astype(np.float32)
    config.ensure_dirs()
    np.savez_compressed(path, burn_prob=burn_prob, centrality=centrality, n_sims=n_sims)
    return Reference(burn_prob, centrality, n_sims)

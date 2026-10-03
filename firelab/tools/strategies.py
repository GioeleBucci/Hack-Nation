"""generate_plan: build a treatment plan and register it in the record."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from firelab import config
from firelab.landscape import Landscape, load_landscape
from firelab.placement import STRATEGIES, build_plan
from firelab.tools._common import ToolError, check_id, get_record, tool


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(mask).tobytes()).hexdigest()[:12]


@tool
def generate_plan(strategy: str, budget_pct: float, seed: int = 0, params: dict | None = None,
                  plan_id: str | None = None) -> dict:
    """Build a plan. Identical requests return the same plan id (idempotent)."""
    ls = load_landscape()
    try:
        mask, resolved = build_plan(ls, strategy, float(budget_pct), int(seed), params)
    except ValueError as e:
        raise ToolError(str(e)) from e

    spec = {"strategy": strategy, "params": resolved, "budget_pct": float(budget_pct), "seed": int(seed),
            "landscape": ls.name, "landscape_fingerprint": ls.fingerprint()}
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:6]
    plan_id = check_id(plan_id, "plan_id") if plan_id else f"{STRATEGIES[strategy]['code']}-{digest}"

    rec = get_record()
    existing = rec.get_one("plan", plan_id)
    if existing:
        same = {k: existing["payload"].get(k) for k in spec} == json.loads(json.dumps(spec, default=str))
        if not same:
            raise ToolError(f"plan_id '{plan_id}' already exists with a different specification")
        return {**existing["payload"], "reused": True}

    target = int(round(float(budget_pct) / 100 * ls.n_burnable))
    treated = int(mask.sum())
    payload = {
        "id": plan_id, **spec,
        "treated_cells": treated,
        "treated_ha": treated * ls.cell_ha,
        "treated_pct_of_burnable": round(100 * treated / ls.n_burnable, 3),
        "under_budget": treated < target,
        "mask_hash": _mask_hash(mask),
    }
    config.ensure_dirs()
    np.save(config.PLANS_DIR / f"{plan_id}.npy", mask)
    rec.put("plan", payload, author="generate_plan")
    note = (f"Only {treated} of {target} budgeted cells were eligible for this strategy." if treated < target else None)
    return {**payload, "reused": False, "note": note}


def load_plan_mask(plan: dict, ls: Landscape) -> np.ndarray:
    """Load a plan's mask; rebuild it from its spec if the .npy is missing, and verify the hash."""
    path = config.PLANS_DIR / f"{plan['id']}.npy"
    if path.exists():
        mask = np.load(path)
    else:
        mask, _ = build_plan(ls, plan["strategy"], plan["budget_pct"], plan["seed"], plan["params"])
    if _mask_hash(mask) != plan["mask_hash"]:
        raise ToolError(f"Plan {plan['id']} does not match its recorded hash; the landscape or code changed.")
    return mask

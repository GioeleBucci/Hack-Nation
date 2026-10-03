"""Paired statistics for plan-vs-baseline comparisons (see docs/statistics.md).

All comparisons are paired by scenario: plan and baseline burned under the same ignition,
weather and random numbers, so we analyse the per-scenario differences directly.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

# Pre-registered success criterion (README.md): a plan "wins" against the baseline if mean
# burned area drops by at least MIN_EFFECT_PCT, the 95% CI of the relative change excludes
# zero, and the Holm-corrected Wilcoxon p-value is below ALPHA.
MIN_EFFECT_PCT = 10.0
ALPHA = 0.05


def paired_bootstrap(plan: np.ndarray, base: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean difference and relative change with percentile 95% CIs, resampling scenarios."""
    diff = plan - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_diff = diff[idx].mean(axis=1)
    boot_base = base[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_rel = np.where(boot_base > 0, 100 * boot_diff / boot_base, 0.0)
    base_mean = float(base.mean())
    return {
        "mean_diff": float(diff.mean()),
        "mean_diff_ci95": [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
        "rel_change_pct": float(100 * diff.mean() / base_mean) if base_mean > 0 else 0.0,
        "rel_change_ci95": [float(np.percentile(boot_rel, 2.5)), float(np.percentile(boot_rel, 97.5))],
    }


def wilcoxon_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when all differences are zero."""
    if not np.any(diff != 0):
        return 1.0
    try:
        return float(wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def verdict(rel_change_pct: float, rel_ci: list[float], p_holm: float,
            min_effect_pct: float = MIN_EFFECT_PCT, alpha: float = ALPHA) -> str:
    if rel_change_pct <= -min_effect_pct and rel_ci[1] < 0 and p_holm < alpha:
        return "wins"
    if rel_ci[0] > 0 and p_holm < alpha:
        return "worse"
    if rel_ci[1] < 0 and p_holm < alpha:
        return "better_below_threshold"
    return "no_clear_difference"


def compare_plans(values: dict[str, np.ndarray], baseline_ids: list[str], n_boot: int = 2000,
                  seed: int = 0, min_effect_pct: float = MIN_EFFECT_PCT) -> dict:
    """Compare every non-baseline plan against the per-scenario mean of the baseline plans.

    ``values`` maps plan_id -> per-scenario metric (same scenario order for all plans).
    """
    base = np.mean([values[b] for b in baseline_ids], axis=0)
    others = [p for p in values if p not in baseline_ids]
    rows = []
    for pid in others:
        row = {"plan_id": pid, "mean": float(values[pid].mean()),
               **paired_bootstrap(values[pid], base, n_boot, seed),
               "p_wilcoxon": wilcoxon_p(values[pid] - base)}
        rows.append(row)
    for row, p_adj in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
        row["p_holm"] = p_adj
        row["verdict"] = verdict(row["rel_change_pct"], row["rel_change_ci95"], p_adj, min_effect_pct)
    rows.sort(key=lambda r: r["rel_change_pct"])
    return {"baseline_ids": baseline_ids, "baseline_mean": float(base.mean()),
            "n_scenarios": int(base.size), "comparisons": rows}

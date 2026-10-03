"""Figures for the demo (optional dependency: pip install -e ".[viz]")."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from firelab import config
from firelab.landscape import load_landscape
from firelab.record import Record


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise SystemExit('Figures need matplotlib: pip install -e ".[viz]"') from e


def _plan_mask(plan_id: str) -> np.ndarray:
    return np.load(config.PLANS_DIR / f"{plan_id}.npy")


def _background(ax, ls):
    ax.imshow(np.where(ls.burnable, ls.fuel, np.nan), cmap="YlGn", vmin=0, vmax=1)
    ax.imshow(np.where(ls.burnable, np.nan, 1.0), cmap="Blues", vmin=0, vmax=1.5)
    ax.contour(ls.assets, levels=[0.5], colors="crimson", linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])


def experiment_figures(experiment_id: str, out_dir: Path | None = None) -> list[str]:
    """Plans, burn-probability maps and the effect-size chart for one experiment."""
    plt = _plt()
    rec = Record()
    ls = load_landscape()
    exp = rec.get_one("experiment", experiment_id)
    if not exp or exp["payload"].get("status") != "done":
        raise SystemExit(f"No finished experiment {experiment_id}")
    out_dir = out_dir or config.RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = {p["id"]: p["payload"] for p in rec.get("plan", exp["payload"]["plan_ids"])}
    with np.load(config.ROOT / exp["payload"]["runs_file"]) as z:
        plan_ids = [str(p) for p in z["plan_ids"]]
        burn_prob = z["burn_count"] / z["burned_ha"].shape[1]
    written = []

    n = len(plan_ids)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.2), squeeze=False)
    for k, pid in enumerate(plan_ids):
        _background(axes[0, k], ls)
        mask = _plan_mask(pid)
        axes[0, k].imshow(np.where(mask, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1)
        axes[0, k].set_title(f"{pid}\n{plans[pid]['strategy']}", fontsize=8)
        im = axes[1, k].imshow(np.where(ls.burnable, burn_prob[k], np.nan), cmap="inferno", vmin=0,
                               vmax=max(float(burn_prob.max()), 1e-6))
        axes[1, k].contour(ls.assets, levels=[0.5], colors="cyan", linewidths=1)
        axes[1, k].set_xticks([])
        axes[1, k].set_yticks([])
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="burn probability")
    fig.suptitle(f"{experiment_id}: treated cells (top, black) and burn probability (bottom)")
    path = out_dir / f"{experiment_id}_maps.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    stats = [s for s in rec.get("stats") if s["payload"]["experiment_id"] == experiment_id]
    if stats:
        res = stats[-1]["payload"]["by_metric"]["burned_ha"]
        rows = res["comparisons"]
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
        y = np.arange(len(rows))
        est = [r["rel_change_pct"] for r in rows]
        lo = [r["rel_change_pct"] - r["rel_change_ci95"][0] for r in rows]
        hi = [r["rel_change_ci95"][1] - r["rel_change_pct"] for r in rows]
        ax.errorbar(est, y, xerr=[lo, hi], fmt="o", color="black", capsize=3)
        ax.axvline(0, color="grey", lw=1)
        ax.axvline(-10, color="crimson", lw=1, ls="--", label="pre-registered -10%")
        ax.set_yticks(y, [f"{r['plan_id']} ({r['strategy']})" for r in rows], fontsize=8)
        ax.set_xlabel(f"change in mean burned area vs {', '.join(res['baseline_ids'])} (%), 95% CI")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        path = out_dir / f"{experiment_id}_effects.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written

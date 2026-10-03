"""Build the 2-minute NABC pitch as one self-contained HTML file from the research record.

    .venv/bin/python demo/build_demo.py        # writes demo/fire_lab_pitch.html

Every number on the slides is read from the record (record/record.db), the experiment run files
and results/acceleration.json, so the pitch cannot drift from what the lab actually did.
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from firelab import config  # noqa: E402
from firelab.landscape import load_landscape  # noqa: E402
from firelab.record import Record  # noqa: E402

OUT = Path(__file__).resolve().parent / "fire_lab_pitch.html"

# Plans shown in the pitch figures (all present in rounds 4 and 5).
NONE, RANDOM, SW_STRIPS, ROBUST_STRIPS = "S0-5cada3", "SR-3d4159", "S4-8ab95d", "S4-f4f566"
SW_COLOR, ROBUST_COLOR = "#e07b00", "#0f8b8d"
VERDICT_COLOR = {"wins": "#1a9850", "no_clear_difference": "#8c8c8c", "worse": "#d73027"}
VERDICT_LABEL = {"wins": "wins (meets pre-registered rule)", "no_clear_difference": "no clear difference",
                 "worse": "worse than random"}
WEATHER_LABEL = {"train": "SW wind (training)", "heldout_nw": "NW wind (held-out)",
                 "heldout_extreme": "extreme weather (held-out)"}

rec = Record()
ls = load_landscape()
e = html.escape


# ----------------------------------------------------------------------------- record helpers
def stats_for(experiment_id: str) -> dict:
    return next(s["payload"] for s in rec.get("stats") if s["payload"]["experiment_id"] == experiment_id)


def cmp(experiment_id: str, plan_id: str, metric: str = "burned_ha") -> dict:
    rows = stats_for(experiment_id)["by_metric"][metric]["comparisons"]
    return next(r for r in rows if r["plan_id"] == plan_id)


def cmp_strategy(experiment_id: str, strategy: str, metric: str = "burned_ha") -> dict:
    rows = stats_for(experiment_id)["by_metric"][metric]["comparisons"]
    return next(r for r in rows if r["strategy"] == strategy)


def pct(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}%".replace("-", "−")


def ci(r: dict) -> str:
    lo, hi = r["rel_change_ci95"]
    return f"[{lo:.1f}, {hi:.1f}]".replace("-", "−")


def mean_ba(experiment_id: str, plan_id: str) -> float:
    return rec.get_one("result", f"{experiment_id}/{plan_id}")["payload"]["metrics"]["mean_ba_ha"]


def burn_prob(experiment_id: str) -> dict[str, np.ndarray]:
    exp = rec.get_one("experiment", experiment_id)["payload"]
    with np.load(config.ROOT / exp["runs_file"]) as z:
        ids = [str(p) for p in z["plan_ids"]]
        bp = z["burn_count"] / z["burned_ha"].shape[1]
    return dict(zip(ids, bp))


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- figures
def fig_maps() -> str:
    """Plan layouts (row 1) and burn probability under SW (row 2) and NW (row 3) wind."""
    plans = [(NONE, "No treatment"), (RANDOM, "Random 5%\n(one control seed)"),
             (SW_STRIPS, "Strips built for SW wind\n(225°, round-2 leader)"),
             (ROBUST_STRIPS, "Strips built for W wind\n(270°, recommended)")]
    rows = [("EXP-R4-A", "SW wind ↗\n(training)"), ("EXP-R4-B", "NW wind ↘\n(held-out)")]
    bps = {exp: burn_prob(exp) for exp, _ in rows}
    vmax = max(float(bps[exp][pid].max()) for exp, _ in rows for pid, _ in plans)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    fuel = np.where(ls.burnable, ls.fuel, np.nan)
    water = np.where(ls.burnable, np.nan, 1.0)
    for j, (pid, title) in enumerate(plans):
        ax = axes[0, j]
        ax.imshow(fuel, cmap="YlGn", vmin=0, vmax=1)
        ax.imshow(water, cmap="Blues", vmin=0, vmax=1.5)
        ax.contour(ls.assets, levels=[0.5], colors="crimson", linewidths=1.6)
        ax.imshow(np.where(np.load(config.PLANS_DIR / f"{pid}.npy"), 1.0, np.nan), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10.5, color=ROBUST_COLOR if pid == ROBUST_STRIPS else
                     SW_COLOR if pid == SW_STRIPS else "black", fontweight="bold")
        for i, (exp, _) in enumerate(rows, start=1):
            ax = axes[i, j]
            im = ax.imshow(np.where(ls.burnable, bps[exp][pid], np.nan), cmap="inferno", vmin=0, vmax=vmax)
            ax.contour(ls.assets, levels=[0.5], colors="cyan", linewidths=1.3)
            ax.text(0.03, 0.04, f"{mean_ba(exp, pid):,.0f} ha / fire", transform=ax.transAxes, fontsize=9.5,
                    color="white", fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Where the 5%\nis treated", fontsize=10.5, fontweight="bold")
    for i, (_, label) in enumerate(rows, start=1):
        axes[i, 0].set_ylabel(f"Burn probability\n{label}", fontsize=10.5, fontweight="bold")
    cb = fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("burn probability = share of simulated fires that reach the cell\n(black = never, yellow = most often)",
                 fontsize=9.5)
    return png(fig)


def _landscape(ax):
    ax.imshow(np.where(ls.burnable, ls.fuel, np.nan), cmap="YlGn", vmin=0, vmax=1)
    ax.imshow(np.where(ls.burnable, np.nan, 1.0), cmap="Blues", vmin=0, vmax=1.5)
    ax.contour(ls.assets, levels=[0.5], colors="crimson", linewidths=1.8)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_landscape() -> str:
    """The test landscape, and where fires go with no treatment (SW training wind)."""
    bp = burn_prob("EXP-R4-A")[NONE]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))
    _landscape(axes[0])
    axes[0].set_title(f"Test landscape\n{ls.n_burnable * ls.cell_ha:,.0f} ha burnable, {ls.assets.sum() * ls.cell_ha:,.0f} ha of homes", fontsize=11, fontweight="bold")
    im = axes[1].imshow(np.where(ls.burnable, bp, np.nan), cmap="inferno", vmin=0, vmax=float(bp.max()))
    axes[1].contour(ls.assets, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(f"No treatment, SW wind\n{mean_ba('EXP-R4-A', NONE):,.0f} ha burned per fire", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.03)
    cb.set_label("burn probability", fontsize=9.5)
    return png(fig)


def fig_mini() -> str:
    """225 vs 270 strips: layout and burn probability under the NW wind where the 225 plan failed."""
    bp = burn_prob("EXP-R4-B")
    vmax = max(float(bp[NONE].max()), float(bp[SW_STRIPS].max()), float(bp[ROBUST_STRIPS].max()))
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 6.0))
    for j, (pid, title, color) in enumerate([(SW_STRIPS, "Built for SW wind\n(225°)", SW_COLOR),
                                             (ROBUST_STRIPS, "Built for W wind\n(270°)", ROBUST_COLOR)]):
        _landscape(axes[0, j])
        axes[0, j].imshow(np.where(np.load(config.PLANS_DIR / f"{pid}.npy"), 1.0, np.nan), cmap="Greys", vmin=0, vmax=1)
        axes[0, j].set_title(title, fontsize=11, fontweight="bold", color=color)
        im = axes[1, j].imshow(np.where(ls.burnable, bp[pid], np.nan), cmap="inferno", vmin=0, vmax=vmax)
        axes[1, j].contour(ls.assets, levels=[0.5], colors="cyan", linewidths=1.3)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        r = cmp("EXP-R4-B", pid)
        axes[1, j].set_title(f"NW wind (round 4)\n{pct(r['rel_change_pct'])} vs random", fontsize=10.5, color=color, fontweight="bold")
    axes[0, 0].set_ylabel("treated cells", fontsize=10)
    axes[1, 0].set_ylabel("where fire goes", fontsize=10)
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.9, pad=0.03).set_label("burn probability", fontsize=9)
    return png(fig)


def fig_transfer() -> str:
    """Change vs random for the two strip plans in every weather regime, burned area and assets."""
    regimes = [("EXP-R4-A", "SW wind\n(training)"), ("EXP-R4-B", "NW wind\n(held-out)"),
               ("EXP-R5-A", "extreme weather\n(held-out)")]
    series = [(SW_STRIPS, "strips for SW wind (225°)", SW_COLOR, +0.14),
              (ROBUST_STRIPS, "strips for W wind (270°) – recommended", ROBUST_COLOR, -0.14)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for ax, metric, title in ((axes[0], "burned_ha", "Burned area"), (axes[1], "asset_cells", "Asset (home) cells burned")):
        for pid, label, color, dy in series:
            for k, (exp, _) in enumerate(regimes):
                r = cmp(exp, pid, metric)
                lo, hi = r["rel_change_ci95"]
                win = r["verdict"] == "wins"
                ax.errorbar(r["rel_change_pct"], k + dy, xerr=[[r["rel_change_pct"] - lo], [hi - r["rel_change_pct"]]],
                            fmt="o", ms=9, color=color, mfc=color if win else "white", mew=2, capsize=4, lw=2)
        ax.axvline(0, color="grey", lw=1)
        ax.axvline(-10, color="crimson", lw=1.3, ls="--")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("change vs random placement (%), 95% CI   ← better | worse →", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(regimes)), [lbl for _, lbl in regimes], fontsize=10.5)
    axes[0].invert_yaxis()
    return png(fig)


def fig_round(experiment_id: str) -> str:
    """Effect-size chart for one experiment, coloured by verdict, with a legend."""
    s = stats_for(experiment_id)
    m = s["by_metric"]["burned_ha"]
    rows = m["comparisons"]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.6))
    for k, r in enumerate(rows):
        lo, hi = r["rel_change_ci95"]
        c = VERDICT_COLOR.get(r["verdict"], "#8c8c8c")
        ax.errorbar(r["rel_change_pct"], k, xerr=[[r["rel_change_pct"] - lo], [hi - r["rel_change_pct"]]],
                    fmt="o", color=c, ms=7, capsize=3, lw=1.8)
    ax.axvline(0, color="grey", lw=1)
    ax.axvline(-10, color="crimson", lw=1.2, ls="--")
    ax.set_yticks(range(len(rows)), [f"{r['strategy']}  ({r['plan_id']})" for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"change in mean burned area vs pooled random controls ({len(m['baseline_ids'])} seeds), %",
                  fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    return png(fig)


def fig_accel(acc: dict) -> str:
    rs = acc["random_search_sims"]
    traj = acc["agent_trajectory"]
    spent = traj[-1]["cumulative_sims"]
    best_agent = max(t["best_reduction_pct"] for t in traj)
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.barh(0, rs["median"], color="#4c72b0", xerr=[[rs["median"] - rs["iqr"][0]], [rs["iqr"][1] - rs["median"]]],
            capsize=5)
    ax.barh(1, acc["grid_search_sims"], color="#8172b2")
    ax.barh(2, spent, color="white", edgecolor=ROBUST_COLOR, hatch="//", lw=1.5)
    ax.text(rs["iqr"][1] + 250, 0, f"{rs['median']:,.0f} sims (median, IQR bar)", va="center", fontsize=9)
    ax.text(acc["grid_search_sims"] + 250, 1, f"{acc['grid_search_sims']:,} sims", va="center", fontsize=9)
    ax.text(spent / 2, 2, f"target not reached in {spent:,} sims (best {best_agent:.0f}% vs target "
            f"{acc['target_reduction_pct']:.0f}%)", va="center", ha="center", fontsize=8.8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_yticks([0, 1, 2], ["random search", "grid search", "Fire Lab agents"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("simulations to find a plan within 5% of the best (SW training weather only)", fontsize=9)
    ax.set_xlim(0, spent * 1.05)
    return png(fig)


# ----------------------------------------------------------------------------- facts from the record
acc = json.loads((config.RESULTS_DIR / "acceleration.json").read_text())
budget_used = rec.sims_used()
evidence = rec.get("evidence")
hyps = sorted(rec.get("hypothesis"), key=lambda h: int(h["id"][1:]))
options = sorted(rec.get("experiment_option"), key=lambda o: (o["payload"]["round"], o["id"]))
decisions = sorted(rec.get("decision"), key=lambda d: int(d["id"][1:]))
findings = sorted(rec.get("finding"), key=lambda f: int(f["id"][1:]))
risks = sorted(rec.get("risk_flag"), key=lambda r: int(r["id"][1:]))
recommendation = rec.get_one("recommendation", "REC1")
approval = rec.get_one("approval", "A1")
t0 = datetime.fromisoformat(min(x["created_at"] for x in evidence))
t1 = datetime.fromisoformat(approval["created_at"])
minutes = round((t1 - t0).total_seconds() / 60)

outcome: dict[str, dict] = {}
for f in findings:
    for u in f["payload"].get("hypothesis_updates", []):
        outcome[u["hypothesis_id"]] = {**u, "finding": f["id"]}

base = cmp("R0-manual-1", NONE)
r1_cent, r1_hyb, r1_strip = (cmp_strategy("EXP-R1-A", "path_centrality"), cmp_strategy("EXP-R1-A", "hybrid"),
                             cmp_strategy("EXP-R1-A", "wind_strips"))
r2 = cmp("EXP-R2-A", SW_STRIPS)
r3 = cmp("EXP-R3-A", SW_STRIPS)
r4_sw, r4_nw = cmp("EXP-R4-A", ROBUST_STRIPS), cmp("EXP-R4-B", ROBUST_STRIPS)
r4_assets_sw = cmp("EXP-R4-A", ROBUST_STRIPS, "asset_cells")
r5 = cmp("EXP-R5-A", ROBUST_STRIPS)
n_rounds = len({o["payload"]["round"] for o in options})
n_chosen = sum(bool(o["payload"].get("chosen")) for o in options)
n_exps = len([x for x in rec.get("experiment") if x["id"] != "R0-manual-1"])

print("Rendering figures…")
IMG_MAPS, IMG_TRANSFER, IMG_ACCEL = fig_maps(), fig_transfer(), fig_accel(acc)
IMG_LAND, IMG_MINI = fig_landscape(), fig_mini()
round_exps = [x["id"] for x in sorted(rec.get("experiment"), key=lambda x: x["created_at"]) if x["id"] != "R0-manual-1"]
IMG_ROUNDS = {x: fig_round(x) for x in round_exps}

# ----------------------------------------------------------------------------- slide content
LEGEND_MAPS = """
<div class="legend">
  <b>How to read the maps</b>
  <span><i class="sw" style="background:#000"></i>black = treated cells (the 5% budget)</span>
  <span><i class="sw" style="background:linear-gradient(90deg,#f7fcb9,#238443)"></i>green = fuel load (light low → dark high)</span>
  <span><i class="sw" style="background:#6baed6"></i>blue = water (cannot burn)</span>
  <span><i class="ring" style="border-color:crimson"></i>red ring = assets / homes</span>
  <span><i class="sw" style="background:linear-gradient(90deg,#000004,#bc3754,#fcffa4)"></i>burn maps: black = never burns → yellow = burns most often</span>
  <span><i class="ring" style="border-color:#00d0d0"></i>cyan ring = assets on burn maps</span>
  <span><i class="sw" style="background:#fff;border:1px solid #999"></i>white on burn maps = water</span>
</div>"""

LEGEND_CHART = """
<div class="legend">
  <b>How to read the chart</b>
  <span><i class="dot" style="background:#e07b00"></i>orange = strips built for SW wind (225°)</span>
  <span><i class="dot" style="background:#0f8b8d"></i>teal = strips built for W wind (270°, recommended)</span>
  <span><i class="dot" style="background:#777"></i>filled = wins · <i class="dot hollow"></i>hollow = does not win</span>
  <span><i class="bar"></i>bar = 95% confidence interval</span>
  <span><i class="dash"></i>red dashed = −10% bar · grey line = same as random</span>
  <span>left of 0 = better than random placement</span>
</div>"""

LEGEND_ROUND = """
<div class="legend">
  <b>How to read these charts</b>
  <span><i class="dot" style="background:#1a9850"></i>green = wins (≥10% less burned area than random, CI excludes 0, Holm p&lt;0.05)</span>
  <span><i class="dot" style="background:#8c8c8c"></i>grey = no clear difference</span>
  <span><i class="dot" style="background:#d73027"></i>red = significantly worse than random</span>
  <span><i class="bar"></i>bar = 95% CI · <i class="dash"></i>red dashed = −10% bar · grey line = same as random</span>
  <span>“none” is the no-treatment control: it should sit far to the right (worse)</span>
</div>"""

main_slides = []

main_slides.append(dict(tag="", title="", t=8, cls="title", body=f"""
<div class="hero">
  <div class="kicker">Hack-Nation · Challenge 03 · Agentic Scientific Discovery · Databricks × Omnigent</div>
  <h1>Fire Lab</h1>
  <p class="sub">An AI lab of 7 agents that learns <b>where to put prescribed burns</b> — and checks whether the answer
  still holds <b>when the wind changes</b>.</p>
  <div class="stats">
    <div><b>{n_rounds}</b><span>experiment rounds</span></div>
    <div><b>{budget_used:,}</b><span>fire simulations</span></div>
    <div><b>{len(evidence)}</b><span>cited papers</span></div>
    <div><b>{len(hyps)}</b><span>hypotheses tested</span></div>
    <div><b>{minutes} min</b><span>question → approved recommendation</span></div>
  </div>
</div>""", notes="Hi, we're Fire Lab. In two minutes: the need, how our agents work, what they found, and how we compare."))

main_slides.append(dict(tag="N", title="Need", t=20, body=f"""
<div class="cols need">
 <div>
  <ul class="big">
   <li>Fire agencies can treat only <b>~5% of a landscape</b> a year with prescribed burns.</li>
   <li><b>Where</b> matters: doing nothing burns <b>{pct(base['rel_change_pct'])}</b> more area than even <i>random</i>
       5% placement (95% CI {ci(base)}).</li>
   <li>Each plan needs <b>thousands of fire simulations</b> → analysts compare a handful, one at a time.</li>
   <li>Hidden risk: a plan tuned for <b>one wind</b> can quietly fail when the wind changes.</li>
  </ul>
  <div class="card accent"><div class="q">Our question</div>
   <p>With a fixed 5% budget, which placement <b>rule</b> minimises burned area and homes exposed —
   and <b>does it still win under a different wind?</b></p>
   <div class="q">Win rule, fixed before any agent ran</div>
   <p>≥10% less burned area than random · 95% CI excludes 0 · Holm p &lt; 0.05 · vs no-treatment and random
   controls on identical fires.</p></div>
 </div>
 <div class="figbox"><img src="{IMG_LAND}" alt="landscape">{LEGEND_MAPS}</div>
</div>""", notes="Agencies can treat about 5% a year. Placement matters — no treatment burns 51% more than even random placement. "
              "But testing a plan takes thousands of simulations, so analysts try a few. And a plan tuned to one wind can fail in another."))

agent_flow = "".join(f'<div class="agent{" human" if a == "Human" else ""}"><b>{a}</b><span>{d}</span></div>'
                     + ('<div class="arrow">→</div>' if i < 6 else "")
                     for i, (a, d) in enumerate([
                         ("Literature", "finds evidence (OpenAlex)"), ("Hypothesis", "proposes testable rules"),
                         ("Planner", "designs ≥2 rival experiments, picks one"), ("Runner", "runs fire simulations"),
                         ("Analysis", "statistics + surprises"), ("Safety", "risk flags, burn-window days"),
                         ("Human", "approves or rejects")]))
main_slides.append(dict(tag="A", title="Approach", t=22, body=f"""
<div class="flow">{agent_flow}</div>
<div class="loopnote">↺ A <b>Director</b> agent (Claude Opus) runs the loop and decides each next step; six Claude Sonnet specialists do
the work. They pass <b>record IDs, not prose</b> — every claim traces back to a stored item.</div>
<div class="cols three">
 <div class="card"><div class="q">Science guardrails</div>
  <p>Fixed 20,000-simulation budget · win rule fixed before any agent ran · no-treatment + random controls in every
  experiment · held-out ignitions and weather for transfer.</p></div>
 <div class="card"><div class="q">Integrity guardrails</div>
  <p>Evidence must cite a real paper (checked on OpenAlex) · results and statistics are written by tools, never typed by
  a model · hypotheses labelled agent-generated.</p></div>
 <div class="card"><div class="q">Human in the loop</div>
  <p>Omnigent policies: budget caps, large experiments and the final recommendation pause for a person to
  approve.</p></div>
</div>
<div class="statbar">
 <div><b>{len(evidence)}</b>papers cited</div><div><b>{len(hyps)}</b>hypotheses</div>
 <div><b>{len(options)}</b>rival designs ({n_chosen} chosen)</div><div><b>{n_exps}</b>controlled experiments</div>
 <div><b>{budget_used:,}</b>simulations</div><div><b>{len(decisions)}</b>recorded decisions</div>
 <div><b>{len(risks)}</b>safety flags</div></div>""", notes="One director agent coordinates six specialists: literature, hypothesis, planner, runner, analysis, safety. "
              "They hand each other record IDs, so every number is traceable. Guardrails: fixed budget, pre-registered rule, real citations, "
              "tools write the results, and a human approves the final recommendation."))

main_slides.append(dict(tag="A", title="How the lab thought — 5 rounds", t=25, body=f"""
<div class="timeline">
 <div class="step"><div class="rn">R1 · screen</div>
  <p>6 strategies. Path-centrality ({pct(r1_cent['rel_change_pct'])}) and hybrid ({pct(r1_hyb['rel_change_pct'])})
  made fires <b>bigger</b> → rejected. Wind strips lead ({pct(r1_strip['rel_change_pct'])}) but p = {r1_strip['p_holm']:.4f}
  → <i>confirm, don't promote</i>.</p></div>
 <div class="step"><div class="rn">R2 · confirm</div>
  <p>Strips built for the SW wind: <b>{pct(r2['rel_change_pct'])}</b> {ci(r2)} ✓. Denser strips did worse.</p></div>
 <div class="step fail"><div class="rn">R3 · transfer</div>
  <p>Same plan, <b>NW wind</b>: only {pct(r3['rel_change_pct'])}, Holm p = {r3['p_holm']:.2f} ✗<br>
  <q>Transfer test FAILED for the leader … Run one more round (round 4) to find a single FIXED placement that is robust to wind direction.</q> — D3</p></div>
 <div class="step win"><div class="rn">R4 · new idea</div>
  <p>Hypothesis H9: orient strips <b>between</b> the winds (270°). Wins SW <b>{pct(r4_sw['rel_change_pct'])}</b> and
  NW <b>{pct(r4_nw['rel_change_pct'])}</b> ✓</p></div>
 <div class="step win"><div class="rn">R5 · stress</div>
  <p>Extreme weather: <b>{pct(r5['rel_change_pct'])}</b> {ci(r5)} ✓ → safety review → human approval.</p></div>
</div>
<div class="cols think">
 <div class="stack">
 <div class="card accent"><div class="q">The moment a result changed the plan</div>
  <p>Finding F3 (transfer failed) → Decision D3 (new round, new hypothesis) → H9 → the recommended plan.
  The failure was predicted in advance by hypothesis H5 — the lab tried to falsify its own leader.</p></div>
 <div class="card"><div class="q">It checked its own work</div>
  <p>In D1 the director caught the analysis agent reading an efficiency score backwards and corrected the record;
  the planner wrote {len(options)} rival designs and justified each of the {n_chosen} it chose.</p></div>
 </div>
 <div class="figbox mini"><img src="{IMG_MINI}" alt="225 vs 270 strips under NW wind">
  <p class="cap">Black = treated cells · green = fuel (light low → dark high) · blue = water · red/cyan ring = homes ·
  burn maps: black = never burns → yellow = burns most often. Under NW wind, fire slips past the 225° strips.</p></div>
</div>""", notes="Round 1 screened six strategies; two made fires worse. Round 2 confirmed strips built for the south-west wind: minus 30 percent. "
              "Round 3 is the key moment: under a north-west wind that leader failed. The agents didn't hide it — they opened a new round "
              "and proposed strips oriented between the winds. That plan won under both winds and under extreme weather."))

main_slides.append(dict(tag="B", title="Benefits — what we found", t=25, body=f"""
<div class="cols results">
 <div class="figbox"><img src="{IMG_TRANSFER}" alt="transfer chart">{LEGEND_CHART}</div>
 <div class="side">
  <div class="card accent"><div class="q">Recommended rule (REC1, human-approved)</div>
   <p>Fire-break strips laid across a <b>westerly (270°) wind</b>, 12 cells apart: the <b>only plan that won in all three
   weather regimes</b> — SW {pct(r4_sw['rel_change_pct'])}, NW {pct(r4_nw['rel_change_pct'])},
   extreme {pct(r5['rel_change_pct'])} burned area vs random.</p></div>
  <div class="card warn"><div class="q">Honest trade-off</div>
   <p>Under SW wind it protects homes <b>worse</b> than random ({pct(r4_assets_sw['rel_change_pct'])}, flagged
   <b>high risk</b> by the safety agent). The SW-tuned plan stays stronger if the wind really is SW.</p></div>
  <div class="card"><div class="q">For a fire planner</div>
   <p>{len(hyps)} hypotheses, {n_exps} controlled experiments and a safety review in <b>{minutes} minutes</b> — every number
   traceable to a record ID, with negative results kept.</p></div>
 </div>
</div>""", notes="Here's the result. Orange is the SW-tuned plan: great under SW, fails under NW — hollow dot. Teal is the 270-degree plan: "
              "filled in all three regimes. The trade-off: under SW wind it protects homes worse than random, and the safety agent flagged that as high risk. "
              "All of this took 32 minutes, fully traceable."))

main_slides.append(dict(tag="C", title="Competition — and where we stand", t=20, body=f"""
<div class="cols">
 <table class="cmp">
  <tr><th></th><th>Analyst by hand</th><th>Grid / random search</th><th>Fire Lab agents</th></tr>
  <tr><td>Plans compared</td><td>a handful</td><td>whole fixed list</td><td>chooses what to test next</td></tr>
  <tr><td>Tests wind transfer</td><td>rarely</td><td>no — one weather set</td><td><b>yes, pre-registered</b></td></tr>
  <tr><td>Explains why</td><td>yes, slowly</td><td>no</td><td><b>yes — recorded decisions</b></td></tr>
  <tr><td>Best single-weather plan</td><td>—</td><td><b>yes</b> (random: {acc['random_search_sims']['median']:,.0f} sims)</td>
      <td>not found ✗</td></tr>
  <tr><td>Safety review + approval</td><td>manual</td><td>no</td><td><b>built in</b></td></tr>
 </table>
 <div class="figbox small"><img src="{IMG_ACCEL}" alt="acceleration benchmark">
  <p class="cap"><b>Where we lost:</b> on pure single-weather optimisation, random search found sparser strips
  (spacing 20, ~{acc['best_reduction_pct']:.0f}% less burned area than no treatment) — our agents never tried spacing &gt; 12.
  <b>Next experiment:</b> test spacing 16–20 strips for transfer, then a real landscape (LANDFIRE fuels + Cell2Fire).</p></div>
</div>""", notes="Versus an analyst, we test more and record why. Versus grid or random search: they optimise one weather and can't explain. "
              "Honestly, on that narrow benchmark random search found a better configuration — sparser strips our agents never tried. "
              "That's our next experiment, along with a real landscape. Thank you."))

# ----------------------------------------------------------------------------- appendix
def short(text: str, n: int) -> str:
    text = str(text or "")
    return e(text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…")


ev_rows = "".join(
    f"<tr><td>{x['id']}</td><td>{x['payload'].get('year') or ''}</td><td>{'gap' if x['payload'].get('gap') else x['payload'].get('confidence', '')}</td>"
    f"<td>{short(x['payload'].get('claim'), 220)}</td><td><a href=\"https://openalex.org/{x['payload']['source_id'].split(':')[-1]}\" "
    f"target=\"_blank\">{e(x['payload']['source_id'])}</a></td></tr>"
    for x in sorted(evidence, key=lambda x: int(x["id"][1:])))

hyp_rows = ""
for h in hyps:
    p, o = h["payload"], outcome.get(h["id"], {})
    verdict = o.get("verdict", p.get("status", ""))
    obs = f"{pct(o['observed_change_pct'])} ({o['finding']})" if "observed_change_pct" in o else "—"
    hyp_rows += (f"<tr class=\"v-{e(verdict)}\"><td>{h['id']}</td><td>{e(p.get('strategy', ''))}</td>"
                 f"<td>{pct(p['predicted_change_pct'], 0)} {e(p.get('metric', ''))}</td><td>{obs}</td><td><b>{e(verdict)}</b></td>"
                 f"<td>{e(p.get('parent_id') or '')}</td><td>{short(p.get('statement'), 200)}</td></tr>")

opt_rows = "".join(
    f"<tr class=\"{'chosen' if o['payload'].get('chosen') else ''}\"><td>{o['id']}</td><td>R{o['payload']['round']}</td>"
    f"<td>{'✓ chosen' if o['payload'].get('chosen') else 'rival'}</td><td>{e(o['payload'].get('design', ''))}</td>"
    f"<td>{o['payload'].get('cost_sims', 0):,}</td><td>{short(o['payload'].get('why'), 260)}</td></tr>" for o in options)

dec_cards = "".join(
    f"<div class=\"card dec\"><div class=\"q\">{d['id']} · based on {', '.join(d['payload'].get('based_on', []))}</div>"
    f"<p><b>{short(d['payload'].get('decision'), 330)}</b></p><p class=\"muted\">{short(d['payload'].get('rationale'), 380)}</p></div>"
    for d in decisions)

risk_rows = "".join(
    f"<tr class=\"sev-{e(r['payload'].get('severity', ''))}\"><td>{r['id']}</td><td>{e(r['payload'].get('category', ''))}</td>"
    f"<td><b>{e(r['payload'].get('severity', ''))}</b></td><td>{short(r['payload'].get('description'), 260)}</td>"
    f"<td>{short(r['payload'].get('mitigation'), 180)}</td></tr>" for r in risks)

round_figs = "".join(
    f"<div class=\"figbox\"><div class=\"q\">{x} · {WEATHER_LABEL.get(rec.get_one('experiment', x)['payload']['weather_set'], '')}"
    f" · {rec.get_one('experiment', x)['payload']['cost_sims']:,} sims</div><img src=\"{img}\" alt=\"{x}\"></div>"
    for x, img in IMG_ROUNDS.items())

appendix = [
    dict(title="Appendix · Maps: where the plans treat, and where fire goes", body=f"""
<div class="figbox wide"><img src="{IMG_MAPS}" alt="maps">{LEGEND_MAPS}
<p class="cap">Rounds 4 (SW, {rec.get_one('experiment', 'EXP-R4-A')['payload']['cost_sims']:,} sims) and 4b
(NW, {rec.get_one('experiment', 'EXP-R4-B')['payload']['cost_sims']:,} sims); labels give mean burned area per simulated
fire. Wind blows toward the corner the arrow points to: SW wind pushes fire to the north-east, NW wind to the south-east.</p></div>"""),
    dict(title=f"Appendix · Evidence ({len(evidence)} cards, every source checked on OpenAlex)",
         body=f"<table class=\"data\"><tr><th>id</th><th>year</th><th>conf.</th><th>claim</th><th>source</th></tr>{ev_rows}</table>"),
    dict(title=f"Appendix · Hypotheses ({len(hyps)}, all agent-generated) and what happened to them",
         body=f"<table class=\"data\"><tr><th>id</th><th>strategy</th><th>predicted</th><th>observed</th><th>verdict</th>"
              f"<th>parent</th><th>statement</th></tr>{hyp_rows}</table>"),
    dict(title=f"Appendix · Rival experiment designs ({len(options)} written, {n_chosen} chosen)",
         body=f"<table class=\"data\"><tr><th>id</th><th>round</th><th></th><th>design</th><th>sims</th><th>planner's reasoning</th></tr>{opt_rows}</table>"),
    dict(title="Appendix · Decision trail (director)", body=f"<div class=\"decs\">{dec_cards}</div>"),
    dict(title="Appendix · Every experiment's effect chart", body=f"{LEGEND_ROUND}<div class=\"grid2\">{round_figs}</div>"),
    dict(title=f"Appendix · Safety review ({len(risks)} risk flags) and approval", body=f"""
<table class="data"><tr><th>id</th><th>category</th><th>severity</th><th>risk</th><th>mitigation</th></tr>{risk_rows}</table>
<div class="card"><div class="q">Recommendation {recommendation['id']} → {e(recommendation['payload'].get('status', ''))}
 ({approval['id']}, {e(approval['payload'].get('via', ''))})</div>
<p class="muted">{short(recommendation['payload'].get('caveats'), 900)}</p></div>"""),
    dict(title="Appendix · Limitations and what we learned building it", body=f"""
<div class="cols">
<div class="card"><div class="q">Scientific limits</div><ul>
<li>Synthetic 120×120 landscape and a simplified stochastic fire model (no spotting, crown fire, suppression); treated
cells are modelled as non-burnable.</li>
<li>The 270° idea was proposed <i>after</i> seeing the round-3 failure (risk R7): it needs an independent re-test.</li>
<li>Benchmark: agents spent {acc['agent_trajectory'][-1]['cumulative_sims']:,} sims and never reached the single-weather
optimum that random search finds in ~{acc['random_search_sims']['median']:,.0f}.</li>
<li>Outputs are ranked candidate rules for expert review — not burn prescriptions.</li></ul></div>
<div class="card"><div class="q">Process notes</div><ul>
<li>Round-4 runs were sized at 4,500 sims, just under the 5,000-sim approval trigger (the trigger is per call — a policy gap
to close).</li>
<li>An earlier attempt on an open-weight model (gpt-oss-120b) was discarded: its director wrote evidence under another
agent's name and invented a report. Tool-level guards kept fake results out of the record; we switched models and
restarted from a clean record.</li>
<li>Budget: {budget_used:,} / {config.MAX_SIMS:,} simulations; {len(rec.get('stats'))} statistics items;
{len(rec.get('result'))} per-plan results.</li></ul></div>
</div>"""),
]

# ----------------------------------------------------------------------------- HTML
slides_html = ""
total = sum(s["t"] for s in main_slides)
start = 0
for i, s in enumerate(main_slides):
    end = start + s["t"]
    badge = f'<div class="badge b{s["tag"]}">{s["tag"]}</div>' if s["tag"] else ""
    head = f'<header>{badge}<h2>{e(s["title"])}</h2><div class="slot">{start // 60}:{start % 60:02d}–{end // 60}:{end % 60:02d}</div></header>' if s["title"] else ""
    slides_html += (f'<section class="slide {s.get("cls", "")}" data-end="{end}" data-notes="{e(s["notes"])}">'
                    f'{head}<div class="content">{s["body"]}</div></section>')
    start = end
slides_html += ('<section class="slide divider" data-end="0" data-notes="Appendix — use for questions.">'
                '<div class="content"><h1>Appendix</h1><p class="sub">Full results, evidence and the lab\'s reasoning — for Q&amp;A.</p>'
                '<p class="muted">Generated from the research record on ' + datetime.now().strftime("%Y-%m-%d %H:%M") + '.</p></div></section>')
for s in appendix:
    slides_html += (f'<section class="slide appendix" data-end="0" data-notes=""><header><h2>{e(s["title"])}</h2></header>'
                    f'<div class="content scroll">{s["body"]}</div></section>')

CSS = """
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#0e1013;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;color:#1d232b;overflow:hidden}
#stage{position:absolute;left:50%;top:50%;width:1280px;height:720px;transform-origin:center center}
.slide{position:absolute;inset:0;background:#fbfaf7;border-radius:14px;padding:30px 44px 46px;display:none;flex-direction:column;box-shadow:0 20px 60px #0008}
.slide.active{display:flex}
header{display:flex;align-items:center;gap:14px;margin-bottom:14px}
header h2{margin:0;font-size:32px;letter-spacing:-.5px;flex:1}
.slot{font:600 13px/1 ui-monospace,Menlo,monospace;color:#8a8f98;border:1px solid #ddd;border-radius:20px;padding:6px 10px}
.badge{width:46px;height:46px;border-radius:10px;display:grid;place-items:center;font:800 26px/1 inherit;color:#fff}
.bN{background:#c0392b}.bA{background:#e07b00}.bB{background:#0f8b8d}.bC{background:#5b4bb7}
.content{flex:1;min-height:0}.scroll{overflow:auto;padding-right:6px}
.title{background:radial-gradient(circle at 80% 20%,#ffb347 0,#e8590c 25%,#5c1a00 70%,#1a0a02 100%);color:#fff;justify-content:center}
.hero h1{font-size:96px;margin:6px 0 0;letter-spacing:-3px}.kicker{font-size:15px;opacity:.85;letter-spacing:.4px;text-transform:uppercase}
.sub{font-size:26px;line-height:1.35;max-width:980px}.title .sub b{color:#ffe08a}
.stats{display:flex;gap:16px;margin-top:34px}.stats div{background:#ffffff1f;border:1px solid #ffffff33;border-radius:12px;padding:14px 18px;min-width:150px}
.stats b{display:block;font-size:34px}.stats span{font-size:14px;opacity:.85}
.divider{background:#1d232b;color:#fff;justify-content:center;padding-left:80px}.divider h1{font-size:72px;margin:0}
.cols{display:grid;grid-template-columns:1.15fr 1fr;gap:26px;align-items:start}.cols.three{grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}
.cols.two{grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}.cols.results{grid-template-columns:1.45fr 1fr;gap:20px}
ul.big{font-size:23px;line-height:1.4;padding-left:24px;margin:0 0 14px}ul.big li{margin-bottom:10px}
.card{background:#fff;border:1px solid #e6e2da;border-radius:12px;padding:14px 18px;font-size:17px;line-height:1.45}
.card p{margin:4px 0 8px}.card.accent{border-left:6px solid #e07b00}.card.warn{border-left:6px solid #d73027}
.q{font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:#8a5a00;margin-bottom:4px}
.small .card{font-size:15px}
.flow{display:flex;align-items:stretch;gap:6px;margin-top:6px}.agent{flex:1;background:#fff;border:2px solid #e07b00;border-radius:12px;padding:12px 10px;text-align:center}
.agent{padding:18px 10px}.agent b{display:block;font-size:20px;margin-bottom:4px}.agent span{font-size:14.5px;color:#555}.agent.human{border-color:#0f8b8d;background:#eaf6f6}
.arrow{align-self:center;font-size:24px;color:#e07b00}.loopnote{margin:16px 2px 0;font-size:19px;line-height:1.45}
.timeline{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.step{background:#fff;border:1px solid #e6e2da;border-top:6px solid #9aa3ad;border-radius:12px;padding:10px 12px;font-size:15.5px;line-height:1.4}
.step p{margin:6px 0 0}.step.fail{border-top-color:#d73027;background:#fff6f5}.step.win{border-top-color:#1a9850}
.step q{display:block;font-style:italic;color:#7a1f16;margin-top:6px;font-size:13.5px}
.rn{font-weight:800;font-size:14px;text-transform:uppercase;letter-spacing:.5px}
.figbox{background:#fff;border:1px solid #e6e2da;border-radius:12px;padding:10px}.figbox img{width:100%;display:block}
.figbox.small img{max-height:260px;object-fit:contain}.figbox.wide img{max-height:500px;object-fit:contain}
.side{display:flex;flex-direction:column;gap:12px}.side .card{font-size:15px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12.5px;color:#333;border-top:1px solid #eee;margin-top:6px;padding-top:6px}
.legend b{width:100%;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#8a5a00}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:22px;height:12px;border-radius:2px}.ring{display:inline-block;width:13px;height:13px;border-radius:50%;border:2.5px solid}
.dot{display:inline-block;width:12px;height:12px;border-radius:50%}.dot.hollow{background:#fff;border:2px solid #777}
.bar{display:inline-block;width:24px;height:2px;background:#333;position:relative}.dash{display:inline-block;width:24px;border-top:2px dashed crimson}
.cap{font-size:13.5px;line-height:1.4;margin:8px 4px 2px}
table.cmp{border-collapse:collapse;font-size:17.5px;width:100%;background:#fff;border-radius:12px;overflow:hidden}
.cmp th,.cmp td{padding:10px 12px;border-bottom:1px solid #eee;text-align:left}.cmp th{background:#1d232b;color:#fff;font-size:14px}
.cmp td:last-child{background:#eaf6f6}
table.data{border-collapse:collapse;font-size:12.5px;width:100%;background:#fff}
.data th{position:sticky;top:0;background:#1d232b;color:#fff;text-align:left;padding:6px 8px}.data td{padding:6px 8px;border-bottom:1px solid #eee;vertical-align:top}
.data a{color:#0f6e8d}.v-supported td,.chosen td{background:#eef8f0}.v-rejected td{background:#fdf0ee}.v-inconclusive td{background:#f6f6f6}
.sev-high td{background:#fdf0ee}.sev-medium td{background:#fff8ec}
.decs{display:grid;grid-template-columns:1fr 1fr;gap:12px}.dec{font-size:14px}.muted{color:#5d6670}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
#hud{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);display:flex;gap:10px;align-items:center;font:600 13px ui-monospace,Menlo,monospace;color:#cfd3d8;z-index:5}
#hud button{background:#2a2f36;color:#fff;border:0;border-radius:8px;padding:7px 12px;font:inherit;cursor:pointer}
#hud button:hover{background:#3a414a}#timer.over{color:#ff6b5b}#timer.ok{color:#7ee2a8}
#prog{position:fixed;top:0;left:0;height:4px;background:#e07b00;z-index:5;transition:width .3s}
#notes{position:fixed;bottom:56px;left:50%;transform:translateX(-50%);width:min(1100px,92vw);background:#1d232bee;color:#fff;font-size:17px;line-height:1.45;padding:14px 18px;border-radius:10px;display:none;z-index:6}
.cols.need{grid-template-columns:1fr 1fr;gap:24px}.cols.need .figbox img{max-height:330px;object-fit:contain}
.cols.think{grid-template-columns:1fr 1fr;gap:16px;margin-top:14px;align-items:stretch}.stack{display:flex;flex-direction:column;gap:12px}
.stack .card{font-size:17px}.figbox.mini{display:flex;gap:10px;align-items:center}.figbox.mini img{width:58%;max-height:330px;object-fit:contain}
.figbox.mini .cap{font-size:14px}.cols.results .figbox img{max-height:420px;object-fit:contain}
.statbar{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-top:18px}
.statbar div{background:#1d232b;color:#cfd3d8;border-radius:12px;padding:14px 10px;text-align:center;font-size:14px}
.statbar b{display:block;font-size:30px;color:#ffb347;margin-bottom:2px}
#help{position:fixed;top:10px;right:14px;color:#8a8f98;font:12px ui-monospace,Menlo,monospace;z-index:5}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];let i=0,t0=null,tick=null;const LIMIT=%d;
const stage=document.getElementById('stage');
function fit(){const s=Math.min(innerWidth/1300,(innerHeight-60)/740);stage.style.transform=`translate(-50%%,-52%%) scale(${s})`}
function show(n){i=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,k)=>s.classList.toggle('active',k===i));
 document.getElementById('prog').style.width=(100*(i+1)/slides.length)+'%%';
 document.getElementById('pos').textContent=(i+1)+' / '+slides.length;
 document.getElementById('notes').textContent=slides[i].dataset.notes||'';location.hash=i+1}
function fmt(s){return Math.floor(s/60)+':'+String(Math.floor(s%%60)).padStart(2,'0')}
function startTimer(){if(t0)return;t0=Date.now();tick=setInterval(upd,250)}
function resetTimer(){clearInterval(tick);t0=null;upd()}
function upd(){const el=document.getElementById('timer');const s=t0?(Date.now()-t0)/1000:0;const end=+slides[i].dataset.end;
 el.textContent=fmt(s)+' / '+fmt(LIMIT);el.className=s>LIMIT||(end&&s>end+3)?'over':(t0?'ok':'')}
document.addEventListener('keydown',ev=>{if(['ArrowRight','PageDown',' '].includes(ev.key)){ev.preventDefault();startTimer();show(i+1)}
 else if(['ArrowLeft','PageUp'].includes(ev.key))show(i-1);else if(ev.key==='Home')show(0);else if(ev.key==='End')show(slides.length-1);
 else if(ev.key==='n'||ev.key==='N'){const n=document.getElementById('notes');n.style.display=n.style.display==='block'?'none':'block'}
 else if(ev.key==='t'||ev.key==='T'){t0?resetTimer():startTimer()}else if(ev.key==='f'||ev.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}});
stage.addEventListener('click',ev=>{if(ev.target.closest('a,.scroll'))return;startTimer();show(i+1)});
addEventListener('resize',fit);fit();show((+location.hash.slice(1)||1)-1);upd();
"""

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Fire Lab — 2-minute pitch</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>
<div id="prog"></div><div id="help">→ / space / click: next · ← back · T timer · N notes · F fullscreen</div>
<div id="stage">{slides_html}</div>
<div id="notes"></div>
<div id="hud"><button onclick="show(i-1)">◀</button><span id="pos"></span><button onclick="startTimer();show(i+1)">▶</button>
<span id="timer">0:00 / 2:00</span><button onclick="t0?resetTimer():startTimer()">timer</button></div>
<script>{JS % total}</script></body></html>"""

OUT.write_text(page, encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), main pitch {total // 60}:{total % 60:02d}, "
      f"{len(main_slides)} pitch slides + {len(appendix)} appendix slides")

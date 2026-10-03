"""Export the research record and compute the loop's timing metrics (see docs/acceleration.md)."""

from __future__ import annotations

import json
from datetime import datetime
from statistics import median

from firelab import config
from firelab.record import Record


def _t(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def cycle_times(rec: Record) -> dict:
    """Minutes from a hypothesis being proposed to the first finding that evaluates it."""
    findings = sorted(rec.get("finding"), key=lambda f: f["created_at"])
    first_eval: dict[str, str] = {}
    for f in findings:
        for u in f["payload"]["hypothesis_updates"]:
            first_eval.setdefault(u["hypothesis_id"], f["created_at"])
    cycles = {}
    for h in rec.get("hypothesis"):
        if h["id"] in first_eval:
            proposed = rec.history("hypothesis", h["id"])[0]["created_at"]
            cycles[h["id"]] = round((_t(first_eval[h["id"]]) - _t(proposed)).total_seconds() / 60, 1)
    out: dict = {"per_hypothesis_min": cycles, "n_evaluated": len(cycles)}
    if cycles:
        out["median_cycle_min"] = median(cycles.values())
        hyps = rec.get("hypothesis")
        start = min(rec.history("hypothesis", h["id"])[0]["created_at"] for h in hyps)
        hours = max((_t(findings[-1]["created_at"]) - _t(start)).total_seconds() / 3600, 1e-9)
        out["hypotheses_evaluated_per_hour"] = round(len(cycles) / hours, 2)
    return out


def _evidence_md(rec: Record) -> str:
    lines = ["# Cited evidence", "", "Exported from the research record. Source ids resolve on OpenAlex.", ""]
    for e in rec.get("evidence"):
        p = e["payload"]
        src = p["source_id"]
        url = (f"https://openalex.org/{src.split(':', 1)[1]}" if src.startswith("openalex:")
               else f"https://doi.org/{src.split(':', 1)[1]}")
        tag = "GAP" if p.get("gap") else p["confidence"]
        verified = "" if p.get("citation_verified", True) else " (unverified)"
        lines.append(f"- **{e['id']}** [{tag}] {p['claim']} - *{p.get('title') or ''}* "
                     f"({p.get('year') or 'n.d.'}) [{src}]({url}){verified}")
    return "\n".join(lines) + "\n"


def _report_md(rec: Record, cycles: dict) -> str:
    lines = ["# Lab run report", "", f"Generated {datetime.now().isoformat(timespec='seconds')}.", "",
             f"Simulations used: {rec.sims_used()} / {config.MAX_SIMS}", "", "## Hypotheses (agent-generated)", ""]
    for h in rec.get("hypothesis"):
        p = h["payload"]
        lines.append(f"- **{h['id']}** [{p['status']}] {p['statement']} "
                     f"(strategy `{p['strategy']}`, predicted {p['predicted_change_pct']}%)")
    lines += ["", "## Experiments", ""]
    for o in rec.get("experiment_option"):
        p = o["payload"]
        mark = "CHOSEN" if p["chosen"] else "rejected"
        lines.append(f"- {o['id']} (round {p['round']}, {p['design']}, {p['cost_sims']} sims) {mark}: {p['why']}")
    lines.append("")
    for s in rec.get("stats"):
        lines.append(f"### {s['id']} - {s['payload']['experiment_id']} "
                     f"(baseline {', '.join(s['payload']['baseline_plan_ids'])})")
        lines.append("")
        lines.append("| metric | plan | strategy | change vs baseline | 95% CI | Holm p | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for m, res in s["payload"]["by_metric"].items():
            for r in res["comparisons"]:
                lines.append(f"| {m} | {r['plan_id']} | {r['strategy']} | {r['rel_change_pct']:.1f}% | "
                             f"[{r['rel_change_ci95'][0]:.1f}, {r['rel_change_ci95'][1]:.1f}] | "
                             f"{r['p_holm']:.3g} | {r['verdict']} |")
        lines.append("")
    lines += ["## Findings and decisions", ""]
    for f in rec.get("finding"):
        flag = " **SURPRISE**" if f["payload"]["surprise"] else ""
        lines.append(f"- **{f['id']}**{flag} ({f['payload']['experiment_id']}): {f['payload']['summary']}")
    for d in rec.get("decision"):
        p = d["payload"]
        lines.append(f"- **{d['id']}** based on {', '.join(p['based_on'])}: {p['decision']} -> next: {p['next']}")
    lines += ["", "## Recommendations", ""]
    for r in rec.get("recommendation"):
        p = r["payload"]
        lines.append(f"- **{r['id']}** plan {p['plan_id']}: {p['status']} "
                     f"(transfer tested: {p['transfer_tested']}). Caveats: {p['caveats']}")
    lines += ["", "## Loop timing", "", "```json", json.dumps(cycles, indent=2), "```", ""]
    return "\n".join(lines)


def export(rec: Record | None = None) -> dict:
    """Write results/record_export.json, results/evidence.md and results/report.md."""
    rec = rec or Record()
    config.ensure_dirs()
    cycles = cycle_times(rec)
    paths = {
        "record": config.RESULTS_DIR / "record_export.json",
        "evidence": config.RESULTS_DIR / "evidence.md",
        "report": config.RESULTS_DIR / "report.md",
    }
    paths["record"].write_text(json.dumps(rec.all_versions(), indent=2), encoding="utf-8")
    paths["evidence"].write_text(_evidence_md(rec), encoding="utf-8")
    paths["report"].write_text(_report_md(rec, cycles), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}

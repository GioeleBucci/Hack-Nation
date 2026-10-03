"""Record tools: record_write, record_read, lab_status, publish_recommendation."""

from __future__ import annotations

from pydantic import ValidationError

from firelab import config
from firelab.landscape import load_landscape
from firelab.placement import STRATEGIES
from firelab.scenarios import IGNITION_SETS, WEATHER_SETS, describe_sets
from firelab.schemas import AGENT_KINDS, SYSTEM_KINDS
from firelab.stats import MIN_EFFECT_PCT
from firelab.tools._common import ToolError, get_record, tool
from firelab.tools.literature import resolve_source

TOOL_ADDED_FIELDS = {"citation_verified"}


def _missing(kind: str | None, ids: list[str]) -> list[str]:
    if not ids:
        return []
    found = {i["id"] for i in get_record().get(kind, ids)}
    return [i for i in ids if i not in found]


def _check_links(kind: str, p: dict) -> list[str]:
    """Referential checks that need the record. Returns warnings; raises ToolError on hard failures."""
    rec = get_record()
    warnings: list[str] = []
    if kind == "evidence" and config.VERIFY_CITATIONS:
        resolved = resolve_source(p["source_id"])
        if resolved is False:
            raise ToolError(f"Source {p['source_id']} does not exist in OpenAlex. Never invent sources.")
        p["citation_verified"] = bool(resolved)
        if resolved is None:
            warnings.append("OpenAlex unreachable; citation stored as unverified.")
    elif kind == "hypothesis":
        if missing := _missing("evidence", p["evidence_ids"]) + _missing("finding", p["finding_ids"]):
            raise ToolError(f"Hypothesis cites unknown ids {missing}")
        if p.get("parent_id") and _missing("hypothesis", [p["parent_id"]]):
            raise ToolError(f"parent_id {p['parent_id']} is not a known hypothesis")
    elif kind == "experiment_option":
        if missing := _missing("hypothesis", p["tests"]):
            raise ToolError(f"Option tests unknown hypotheses {missing}")
        if p["ignition_set"] not in IGNITION_SETS or p["weather_set"] not in WEATHER_SETS:
            raise ToolError("Unknown scenario set; see lab_status for the available sets")
        if p["chosen"]:
            rivals = [o for o in rec.get("experiment_option")
                      if o["payload"]["round"] == p["round"] and o["id"] != p.get("id")]
            if not rivals:
                raise ToolError("Write at least one rival option for this round before choosing one.")
        remaining = config.MAX_SIMS - rec.sims_used()
        if p["cost_sims"] > remaining:
            warnings.append(f"Option costs {p['cost_sims']} sims but only {remaining} remain.")
    elif kind == "finding":
        if _missing("experiment", [p["experiment_id"]]):
            raise ToolError(f"Unknown experiment {p['experiment_id']}")
        if p.get("stats_id") and _missing("stats", [p["stats_id"]]):
            raise ToolError(f"Unknown stats id {p['stats_id']}")
        if missing := _missing("hypothesis", [u["hypothesis_id"] for u in p["hypothesis_updates"]]):
            raise ToolError(f"Unknown hypotheses {missing}")
    elif kind == "decision":
        if missing := _missing(None, p["based_on"]):
            raise ToolError(f"Decision rests on unknown ids {missing}")
    elif kind == "risk_flag":
        if _missing("plan", [p["plan_id"]]):
            raise ToolError(f"Unknown plan {p['plan_id']}")
    return warnings


@tool
def record_write(kind: str, payload: dict, author: str | None = None) -> dict:
    """Validate and append an item. Reusing an id writes a new version of that item."""
    if kind in SYSTEM_KINDS:
        raise ToolError(f"'{kind}' items are written by tools only (run_ensemble, compute_stats, ...).")
    if kind not in AGENT_KINDS:
        raise ToolError(f"Unknown kind '{kind}'. Agents may write: {list(AGENT_KINDS)}")
    # Fields added by this tool may come back when an agent rewrites an item it read.
    payload = {k: v for k, v in (payload or {}).items() if k not in TOOL_ADDED_FIELDS}
    try:
        item = AGENT_KINDS[kind].model_validate(payload)
    except ValidationError as e:
        raise ToolError(f"Invalid {kind}: {e}") from None
    p = item.model_dump(mode="json")
    if not p.get("id"):
        p.pop("id", None)   # the record assigns the next id
    warnings = _check_links(kind, p)
    stored = get_record().put(kind, p, author=author)
    return {"kind": kind, "id": stored["id"], "version": stored["version"], "warnings": warnings}


@tool
def record_read(kind: str | None = None, ids: list[str] | None = None, limit: int = 30,
                history: bool = False) -> dict:
    """Latest version of matching items (or the full version history of one id with history=true)."""
    rec = get_record()
    if history:
        if not kind or not ids or len(ids) != 1:
            raise ToolError("history=true needs a kind and exactly one id")
        return {"items": rec.history(kind, ids[0])}
    items = rec.get(kind, ids, limit=int(limit) if limit else None)
    return {"n": len(items), "items": items}


@tool
def lab_status() -> dict:
    """Budget, record counts, open hypotheses, the latest decision and what the tools can do."""
    rec = get_record()
    used = rec.sims_used()
    hypotheses = [{"id": h["id"], "status": h["payload"]["status"], "strategy": h["payload"]["strategy"],
                   "statement": h["payload"]["statement"][:160]} for h in rec.get("hypothesis")]
    decisions = rec.get("decision", limit=1)
    pending = [r["payload"] for r in rec.get("recommendation") if r["payload"]["status"] == "pending_approval"]
    return {
        "landscape": load_landscape().summary(),
        "simulation_budget": {"max": config.MAX_SIMS, "used": used, "remaining": config.MAX_SIMS - used,
                              "max_per_call": config.MAX_SIMS_PER_CALL},
        "record_counts": rec.counts(),
        "hypotheses": hypotheses,
        "latest_decision": decisions[0]["payload"] if decisions else None,
        "pending_recommendations": pending,
        "strategies": {k: {"code": v["code"], "description": v["description"], "default_params": v["defaults"]}
                       for k, v in STRATEGIES.items()},
        "scenario_sets": describe_sets(),
        "success_criterion": f"wins: >= {MIN_EFFECT_PCT}% lower mean burned area than random, 95% CI excludes 0, "
                             "Holm p < 0.05",
        "approval_mode": config.APPROVAL_MODE,
    }


@tool
def publish_recommendation(plan_id: str, decision_id: str, caveats: str,
                           risk_flag_ids: list[str] | None = None) -> dict:
    """Publish a treatment recommendation. Gated by human approval (Omnigent ASK policy or `firelab approve`)."""
    rec = get_record()
    plan = rec.get_one("plan", plan_id)
    if not plan:
        raise ToolError(f"Unknown plan {plan_id}")
    if not rec.get_one("decision", decision_id):
        raise ToolError(f"Unknown decision {decision_id}")
    if not caveats or len(caveats.strip()) < 20:
        raise ToolError("State the caveats explicitly (model limits, uncertainty, validation still needed).")
    flags = [f for f in rec.get("risk_flag") if f["payload"]["plan_id"] == plan_id]
    if not flags:
        raise ToolError("Safety review missing: write at least one risk_flag for this plan first.")

    tested_in = [r["payload"]["experiment_id"] for r in rec.get("result") if r["payload"]["plan_id"] == plan_id]
    experiments = {e["id"]: e["payload"] for e in rec.get("experiment", tested_in)} if tested_in else {}
    transfer_tested = any(e["weather_set"].startswith("heldout") for e in experiments.values())
    verdicts = [{"stats_id": s["id"], "metric": m, "verdict": row["verdict"],
                 "rel_change_pct": row["rel_change_pct"], "ci95": row["rel_change_ci95"]}
                for s in rec.get("stats") if s["payload"]["experiment_id"] in experiments
                for m, res in s["payload"]["by_metric"].items()
                for row in res["comparisons"] if row["plan_id"] == plan_id]

    approved = config.APPROVAL_MODE == "policy"
    payload = {"plan_id": plan_id, "decision_id": decision_id, "caveats": caveats,
               "risk_flag_ids": risk_flag_ids or [f["id"] for f in flags],
               "experiments": sorted(experiments), "transfer_tested": transfer_tested, "evidence": verdicts,
               "status": "approved" if approved else "pending_approval"}
    item = rec.put("recommendation", payload, author="publish_recommendation")
    if approved:
        # This tool only executes after the approval_gate policy's ASK was accepted by a human.
        rec.put("approval", {"recommendation_id": item["id"], "verdict": "approved",
                             "approved_by": config.APPROVER, "via": "omnigent_ask_policy"},
                author="publish_recommendation")
    warnings = [] if transfer_tested else ["Plan was never tested on a held-out weather set."]
    return {"recommendation_id": item["id"], "status": payload["status"], "transfer_tested": transfer_tested,
            "warnings": warnings,
            "next": None if approved else f"A human must run: firelab approve {item['id']} --by <name>"}

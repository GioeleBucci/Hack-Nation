"""End-to-end through the agent-facing tools on the small landscape (no network, no MLflow)."""

from firelab.policies import approval_gate, record_integrity, simulation_budget
from firelab.tools.record import lab_status, publish_recommendation, record_read, record_write
from firelab.tools.simulate import run_ensemble
from firelab.tools.stats import compute_stats
from firelab.tools.strategies import generate_plan


def _call(name, **args):
    return {"type": "tool_call", "data": {"name": name, "arguments": args}, "session_state": {}}


def test_full_loop_through_tools():
    ev = record_write("evidence", {"claim": "c", "source_id": "openalex:W123", "summary": "s",
                                   "confidence": "medium"}, author="test")
    assert ev["ok"], ev
    h = record_write("hypothesis", {"statement": "fuel first", "label": "agent-generated", "strategy": "fuel_load",
                                    "evidence_ids": [ev["id"]], "predicted_change_pct": -15,
                                    "falsified_if": "CI overlaps 0"}, author="test")
    assert h["ok"], h

    option = {"round": 1, "tests": [h["id"]], "design": "screen",
              "plans": [{"strategy": "none"}, {"strategy": "random", "seed": 1}, {"strategy": "fuel_load"}],
              "budget_pct": 5, "n_ignitions": 5, "n_weather": 2, "cost_sims": 30,
              "expected_learning": "x", "feasibility": "y", "why": "z"}
    lonely = record_write("experiment_option", {**option, "chosen": True}, author="test")
    assert not lonely["ok"] and "rival" in lonely["error"]
    rival = record_write("experiment_option", option, author="test")
    chosen = record_write("experiment_option", {**option, "chosen": True}, author="test")
    assert rival["ok"] and chosen["ok"]

    plans = [generate_plan("none", 5, 0), generate_plan("random", 5, 1), generate_plan("fuel_load", 5, 0)]
    assert all(p["ok"] for p in plans)
    assert generate_plan("random", 5, 1)["reused"]
    ids = [p["id"] for p in plans]

    run = run_ensemble("EXP-T1", ids, "train", "train", 5, 2, seed=0, option_id=chosen["id"])
    assert run["ok"], run
    assert run["cost_sims"] == 30
    assert not run_ensemble("EXP-T1", ids, "train", "train", 5, 2, seed=0)["ok"]   # immutable

    st = compute_stats("EXP-T1", [ids[1]])
    assert st["ok"], st
    assert {r["plan_id"] for r in st["by_metric"]["burned_ha"]["comparisons"]} == {ids[0], ids[2]}

    f = record_write("finding", {"experiment_id": "EXP-T1", "stats_id": st["stats_id"], "summary": "s",
                                 "hypothesis_updates": [{"hypothesis_id": h["id"], "verdict": "inconclusive"}],
                                 "confidence": "low"}, author="test")
    d = record_write("decision", {"based_on": [f["id"]], "decision": "d", "rationale": "r", "next": "n"},
                     author="test")
    assert f["ok"] and d["ok"]

    no_review = publish_recommendation(ids[2], d["id"], "Synthetic landscape; not a burn prescription.")
    assert not no_review["ok"] and "risk_flag" in no_review["error"]
    record_write("risk_flag", {"plan_id": ids[2], "category": "model_limits", "severity": "high",
                               "description": "d", "mitigation": "m"}, author="test")
    rec = publish_recommendation(ids[2], d["id"], "Synthetic landscape; not a burn prescription.")
    assert rec["ok"] and rec["status"] == "pending_approval" and not rec["transfer_tested"]

    assert lab_status()["simulation_budget"]["used"] == 30
    assert record_read("hypothesis")["n"] == 1


def test_agents_cannot_write_system_items():
    assert not record_write("result", {"x": 1})["ok"]


def test_policies():
    assert record_integrity(_call("record_write", kind="result", payload={}))["result"] == "DENY"
    assert record_integrity(_call("record_write", kind="hypothesis", payload={"label": "x"}))["result"] == "DENY"
    assert record_integrity(_call("record_read", kind="hypothesis")) is None
    assert approval_gate()(_call("publish_recommendation", plan_id="p"))["result"] == "ASK"
    budget = simulation_budget(max_sims=10)
    assert budget(_call("run_ensemble", plan_ids=["a", "b"], n_ignitions=5, n_weather=2))["result"] == "DENY"

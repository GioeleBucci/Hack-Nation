"""Omnigent policies for the lab (declared under ``policies:`` in agents/lab_director.yaml).

A policy sees each event before it happens and returns ALLOW, DENY, ASK (pause for a human)
or None (abstain). The same rules are also enforced inside the tools, so they hold even
where a policy does not fire (e.g. a managed Omnigent route without custom policies).
See docs/policies.md.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Omnigent is only needed at run time, not to import or test this module
    from omnigent.policies.schema import PolicyEvent, PolicyResponse


def _tool_call(event: "PolicyEvent") -> tuple[str, dict[str, Any]]:
    """(tool name, arguments) for tool_call events, ("", {}) otherwise."""
    if event.get("type") != "tool_call":
        return "", {}
    data = event.get("data") or {}
    args = data.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    # Tool names may carry a prefix (e.g. "mcp__lab__run_ensemble"), so match on the suffix.
    return str(data.get("name", "")), args


def _cost(args: dict) -> int:
    return len(set(args["plan_ids"])) * int(args.get("n_ignitions", 50)) * int(args.get("n_weather", 5))


def simulation_budget(max_sims: int):
    """Factory: DENY run_ensemble calls that would push the lab past its simulation budget.

    Usage is read from the research record rather than from session state, so the count is
    correct across sub-agents, sessions and restarts.
    """
    def evaluate(event: "PolicyEvent") -> "PolicyResponse | None":
        name, args = _tool_call(event)
        if not name.endswith("run_ensemble"):
            return None
        try:
            cost = _cost(args)
        except (KeyError, TypeError, ValueError):
            return {"result": "DENY", "reason": "run_ensemble needs plan_ids, n_ignitions and n_weather."}
        from firelab.tools._common import get_record
        used = get_record().sims_used()
        if used + cost > max_sims:
            return {"result": "DENY",
                    "reason": (f"Simulation budget: {used}/{max_sims} used, request needs {cost}. "
                               "The planner must pick a cheaper experiment or the director must stop.")}
        return {"result": "ALLOW",
                "state_updates": [{"key": "sims_requested", "action": "increment", "value": cost}]}
    return evaluate


def record_integrity(event: "PolicyEvent") -> "PolicyResponse | None":
    """Agents cannot forge system items; evidence needs a source; hypotheses are labeled."""
    name, args = _tool_call(event)
    if not name.endswith("record_write"):
        return None
    from firelab.schemas import SYSTEM_KINDS
    kind, payload = args.get("kind"), args.get("payload") or {}
    if kind in SYSTEM_KINDS:
        return {"result": "DENY", "reason": f"'{kind}' items are written by tools only; agents cannot write them."}
    if kind == "evidence" and not payload.get("source_id"):
        return {"result": "DENY", "reason": "Evidence must carry an OpenAlex ID or DOI in source_id."}
    if kind == "hypothesis" and payload.get("label") != "agent-generated":
        return {"result": "DENY", "reason": "Hypotheses must be labeled 'agent-generated'."}
    return None


def approval_gate(ask_above_sims: int | None = None):
    """Factory: ASK a human before any recommendation is published, and (optionally) before
    very large simulation requests."""
    def evaluate(event: "PolicyEvent") -> "PolicyResponse | None":
        name, args = _tool_call(event)
        if name.endswith("publish_recommendation"):
            return {"result": "ASK",
                    "reason": (f"Approve publishing plan {args.get('plan_id')} as a treatment recommendation "
                               f"(decision {args.get('decision_id')})? Caveats: {args.get('caveats')}")}
        if ask_above_sims and name.endswith("run_ensemble"):
            try:
                cost = _cost(args)
            except (KeyError, TypeError, ValueError):
                return None
            if cost > ask_above_sims:
                return {"result": "ASK", "reason": f"Large experiment: {cost} simulations. Proceed?"}
        return None
    return evaluate

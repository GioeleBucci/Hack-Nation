# Guardrails: policies and tool-level checks

Omnigent evaluates every policy on every tool call **before** the call runs. A policy returns
`ALLOW`, `DENY` (the call is blocked and the reason is shown to the agent), `ASK` (the session pauses
until a human answers) or `None` (no opinion). Our custom policies are in `firelab/policies.py` and
declared under `policies:` in `agents/lab_director.yaml`.

## Defense in depth

Every rule that matters is enforced **twice**: by a policy (visible in the Omnigent session, stops
the call early) and inside the tool (holds even if a policy does not fire, e.g. on the managed
Omnigent route, which only supports built-in policies, or if parent policies turn out not to apply
to sub-agent calls).

| Rule | Policy | Tool-level guard |
|---|---|---|
| LLM spend ≤ $30, ask at $10 and $20 | `llm_budget` (built-in `cost_budget`) | — |
| ≤ 400 tool calls per session | `tool_call_cap` (built-in) | — |
| Total simulations ≤ 20,000 | `sim_budget` → DENY | `run_ensemble` refuses; also caps one call at 6,000 |
| Agents cannot write plans, results, stats, approvals | `record_integrity` → DENY | `record_write` refuses system kinds |
| Evidence cites a real work | `record_integrity` → DENY without `source_id` | `record_write` resolves the id on OpenAlex and refuses unknown works |
| Hypotheses labeled `agent-generated` | `record_integrity` → DENY | Schema requires the literal label |
| Human approves recommendations | `human_approval` → ASK on `publish_recommendation` | Tool requires a decision, ≥ 1 risk flag and explicit caveats; manual mode needs `firelab approve` |
| Human approves very large experiments | `human_approval` → ASK above 5,000 simulations | Per-call cap |

## Notes on each policy

**`simulation_budget(max_sims)`** reads usage from the research record rather than from Omnigent
session state. The count is therefore shared across sub-agents and survives restarts. Cost is
computed from the call arguments (`unique plan ids × n_ignitions × n_weather`). Keep `max_sims` in
the YAML equal to `FIRELAB_MAX_SIMS` (default 20,000).

**`record_integrity`** is a fast pre-check on the `record_write` arguments. The full schema
validation and referential checks run in the tool and come back as readable errors.

**`approval_gate(ask_above_sims)`** puts the plan id, decision id and caveats in the ASK message, so
the human sees what they are approving. Only the director has `publish_recommendation`, so the gate
is declared on the agent whose calls it is guaranteed to see.

## Approval modes

| `FIRELAB_APPROVAL_MODE` | Behaviour |
|---|---|
| `policy` (default) | The Omnigent ASK *is* the approval: the tool only runs once a human accepts, and it records the recommendation as `approved` with an `approval` item (`FIRELAB_APPROVER` names the approver). If the human rejects, the tool never runs, and the director records a decision explaining the rejection. |
| `manual` | The tool records the recommendation as `pending_approval`. A human runs `firelab approve REC1 --by "Name"` or `firelab reject REC1 --by "Name" --reason "..."`. Use this on routes where you cannot confirm that the ASK fires. |

## Tool names in policy events

Omnigent may prefix tool names (e.g. `mcp__…`). The policies match on the suffix
(`name.endswith("run_ensemble")`). Check the exact names in a session log once and tighten the match
if needed.

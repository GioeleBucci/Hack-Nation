# Agent specifications

The lab is one Omnigent orchestrator (the director) with six specialist sub-agents, all declared
in [agents/lab_director.yaml](agents/lab_director.yaml). Each agent owns one scientific decision.
Agents hand each other **record ids**, not prose; the full content lives in the shared research
record ([docs/record.md](docs/record.md)), so every decision can be reconstructed.

## Agents

| Agent | Model | Decision it owns | Tools | Input | Output (record kind) |
|---|---|---|---|---|---|
| Lab director | Opus-class | Which loop step comes next; when to stop | all sub-agents, `lab_status`, `record_read`, `record_write` (decisions), `publish_recommendation` | Human objective, record state | `decision` items, final report |
| Literature | Sonnet-class | What prior evidence says; where the gaps are | `openalex_search`, `record_write`, `record_read` | Research question | 5–15 `evidence` cards with OpenAlex ids |
| Hypothesis | Sonnet-class | Which placement strategies are worth testing | `lab_status`, `record_read`, `record_write` | Evidence ids, finding ids, surprises | 3–5 `hypothesis` items, labeled agent-generated, with quantitative predictions |
| Planner | Sonnet-class | Which experiment to run under the budget | `lab_status`, `record_read`, `record_write` | Round number, hypothesis ids, remaining budget | ≥ 2 `experiment_option` items, one marked chosen with a reason |
| Runner | Sonnet-class | Executing the chosen experiment correctly | `generate_plan`, `run_ensemble`, `record_read`, `lab_status` | Chosen option id | `plan`, `experiment`, `result` items (written by the tools) |
| Analysis | Sonnet-class | What the result means; whether it surprises | `compute_stats`, `record_read`, `record_write` | Experiment id, control plan ids | `stats` (tool) and `finding` with prediction-vs-observed and reopen flags |
| Safety | Sonnet-class | Whether a plan is safe and feasible to put before a human | `burn_window_days`, `record_read`, `record_write`, `lab_status` | Candidate plan id | `risk_flag` items, caveats draft |

The director publishes (`publish_recommendation`) itself so that the human-approval policy, which is
declared on the director, is guaranteed to apply.

## Handoffs

```
human objective
   │
director ──► literature ──E1..E12──► hypothesis ──H1..H4──► planner ──X1 (rival), X2 (chosen)──► runner
   ▲                                     ▲                                                      │
   │                                     │ F3 surprise: reopen H2                         EXP-R1-A
   │                                     └──────────── analysis ◄──────────────────────────────┘
   │                                                     │ ST1, F3
   └── D3 "refine H1 leader; reopen H2" ◄────────────────┘
   ...
director ──► safety (plan S4-…) ──R1..R3──► director ──publish_recommendation──► ASK human ──► REC1 approved / rejected
```

Schemas of every handoff item are in [firelab/schemas.py](firelab/schemas.py) and described in
[docs/record.md](docs/record.md).

## Policies (Omnigent) and the matching tool-level guards

| Policy | Verdict | Rule | Also enforced in |
|---|---|---|---|
| `llm_budget` (built-in `cost_budget`) | ASK at $10, $20; DENY at $30 | LLM spend cap | — |
| `tool_call_cap` (built-in) | DENY after 400 calls | Runaway-loop guard | — |
| `sim_budget` | DENY | A `run_ensemble` call may not push total simulations past 20,000 | `run_ensemble` (same check, plus a 6,000 per-call cap) |
| `record_integrity` | DENY | Agents cannot write plans/results/stats/approvals; evidence needs an OpenAlex/DOI source; hypotheses must be labeled `agent-generated` | `record_write` (schemas, OpenAlex resolution, referential checks) |
| `human_approval` | ASK | Before `publish_recommendation`, and before any experiment over 5,000 simulations | `publish_recommendation` requires a decision, a risk flag and caveats; manual mode needs `firelab approve` |

Details: [docs/policies.md](docs/policies.md).

## Responsibility

- Hypotheses are labeled `agent-generated` (schema and policy).
- Evidence must cite a work that exists in OpenAlex (checked on write).
- Results, statistics and approvals can only be written by tools, never typed in by a model.
- Every experiment includes no-treatment and random controls under identical scenarios.
- Nothing becomes a recommendation without a safety review and a human decision.

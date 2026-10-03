# Architecture

## Three layers

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Omnigent (agents/lab_director.yaml)                                          │
│   director (Opus) ── sub-agents (Sonnet): literature, hypothesis, planner,   │
│                      runner, analysis, safety                                │
│   policies: cost budget · tool-call cap · simulation budget ·                │
│             record integrity · human approval                                │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │ function tools (callable: firelab.tools.*)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ Tools (firelab/tools/)  JSON in → JSON out, validate, never raise            │
│   lab_status · openalex_search · generate_plan · run_ensemble ·              │
│   compute_stats · burn_window_days · record_write · record_read ·            │
│   publish_recommendation                                                     │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │ plain Python
┌───────────────▼──────────────────────────────────────────────────────────────┐
│ Science core (firelab/)                                                      │
│   landscape · scenarios · simulator · reference · placement · stats ·        │
│   schemas · record (SQLite)                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **The science core** has no knowledge of agents. It can be used from a notebook, the CLI or the
  benchmark, and it is what the tests exercise.
- **The tools** are the only way agents touch the science. Each one validates its inputs, enforces
  the lab's rules (budget, schemas, citations, immutability) and writes to the record. Errors come
  back as `{"ok": false, "error": "..."}` so the agent can read the message and correct itself
  instead of crashing the session (`firelab/tools/_common.py`).
- **Omnigent** provides orchestration: the director delegates to sub-agents (declared as
  `type: agent` tools), sub-agents inherit only the tools they need (`tools: {x: inherit}`), and
  policies inspect every tool call before it runs.

Agents have **no shell and no file access** (no `os_env` in the YAML). Everything they can do is a
typed tool call, which keeps the run auditable.

## One discovery loop

| Step | Agent | Tool calls | Record items |
|---|---|---|---|
| 0 | human | `firelab init`, `firelab baseline` | `plan`, `experiment R0-manual-1`, `result`, `stats` |
| 1 | director | `lab_status` | — |
| 2 | literature | `openalex_search` ×3–6, `record_write` | `evidence` E1…E12 (OpenAlex-verified) |
| 3 | hypothesis | `lab_status`, `record_read`, `record_write` | `hypothesis` H1…H4 (agent-generated, quantitative prediction) |
| 4 | planner | `lab_status`, `record_write` ×3 | `experiment_option` X1, X2 (rivals), X2 v2 chosen with reason |
| 5 | runner | `generate_plan` ×N, `run_ensemble` ×1 | `plan` ×N, `experiment` EXP-R1-A, `result` ×N |
| 6 | analysis | `compute_stats`, `record_read`, `record_write` | `stats` ST1, `finding` F1 (surprise? reopen?) |
| 7 | director | `record_write` | `decision` D1 → next step |
| … | | rounds 2–3: refine, then transfer test on held-out weather | |
| 8 | safety | `burn_window_days`, `record_write` | `risk_flag` R1…R3 |
| 9 | director | `publish_recommendation` → **ASK** | `recommendation` REC1, `approval` A1 |

### Where the plan adapts

The loop changes direction at step 7. The director's decision has to cite the finding it rests on
(`based_on`). The finding records, for each hypothesis tested, the observed effect next to the
predicted one, plus a `surprise` flag and a `reopen` list. Typical branches:

- **Prediction failed / control behaved unexpectedly** → the hypothesis agent rewrites the
  hypothesis with status `rejected` or `reopened`, and writes a revised hypothesis with `parent_id`
  and `revision_note`. The next planning round tests the revision.
- **Clear leader** → the planner spends budget on a parameter sweep of the leader or a deeper
  ensemble.
- **Confirmed leader** → transfer test on `heldout` ignitions and `heldout_nw` weather. If the leader
  does not transfer, that is a surprise, and it reopens the assumption that the rule is general.

### Budgeted choice between experiments

The planner must write at least two rival `experiment_option` items before it can mark one as chosen
(`record_write` rejects a chosen option without a rival in the same round). Each option states its
cost in simulations (`plans × ignitions × weather`, checked by the schema), its expected learning and
its feasibility. The remaining budget is visible through `lab_status`, and spending is enforced by
both the `sim_budget` policy and `run_ensemble` itself.

### Parallelism

Independent work can run concurrently: sub-agents can be dispatched in parallel by the director, and
the record is SQLite in WAL mode with `BEGIN IMMEDIATE` writes, so several processes can write safely.
Inside one experiment, all plans burn as a single stacked numpy computation.

## Why results are trustworthy

- **Agents cannot write results.** `plan`, `experiment`, `result`, `stats`, `recommendation` and
  `approval` items are written only by tools. The `record_integrity` policy and `record_write` both
  reject agent attempts.
- **Experiments are immutable.** Re-using an experiment id is refused.
- **Plans are content-addressed.** A plan id is derived from its spec, and its mask hash is checked
  before every run.
- **Paired, matched scenarios with common random numbers** (see [simulator.md](simulator.md)).
- **Everything is reproducible.** Seeds, landscape fingerprint, simulator version, scenario sets and
  MLflow run ids are stored with every experiment.

## Omnigent specifics and open points

- The YAML follows the Omnigent Agent YAML spec (`type: function` tools with `callable`,
  `type: agent` sub-agents with `tools: {x: inherit}`, function policies with `factory_params`).
- The Omnigent docs do not say whether policies declared on the director also fire for sub-agent
  tool calls. The design does not depend on it: every policy rule is also enforced inside the tool,
  and the approval-gated tool (`publish_recommendation`) is only given to the director.
- Policies read the simulation budget from the record, not from Omnigent session state, so the
  count is right even if session state is per sub-agent.
- Model endpoints (`databricks-claude-opus-4-7`, `databricks-claude-sonnet-4-6`) and the auth profile
  (`hackathon`) must match what your Databricks workspace exposes.

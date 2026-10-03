# The research record

`record/record.db` is an append-only, versioned SQLite store (`firelab/record.py`). It is the shared
memory of the lab: agents pass ids, and everything else is looked up here.

## Model

One table, `items(seq, kind, id, version, payload, author, created_at)`:

- **Append-only.** Nothing is updated in place or deleted.
- **Versioned.** Writing an existing `(kind, id)` adds version n+1. `record_read` returns the latest
  version; `record_read(kind, [id], history=true)` returns all versions. Example: H2 goes
  `proposed` (v1) → `rejected` (v2), and both remain visible.
- **Ids** are assigned automatically per kind (`E1`, `H1`, `X1`, `F1`, `D1`, `R1`, `ST1`, `REC1`,
  `A1`) unless the writer passes one. Plans are content-addressed (`S4-3fa2c1`); experiments use
  the runner's id (`EXP-R1-A`); results are `<experiment>/<plan>`.
- **Concurrency.** WAL mode plus `BEGIN IMMEDIATE` per write, so parallel agents cannot get the same
  id or version.

## Item kinds

Agent-written (validated by pydantic schemas in `firelab/schemas.py`, then checked against the record):

| Kind | Key fields | Extra checks on write |
|---|---|---|
| `evidence` | `claim`, `source_id` (`openalex:W…` / `doi:10.…`), `summary`, `confidence`, `gap` | Source must resolve on OpenAlex (`citation_verified` stored) |
| `hypothesis` | `statement`, `label = "agent-generated"`, `strategy`, `params`, `evidence_ids` / `finding_ids`, `predicted_change_pct`, `falsified_if`, `status`, `parent_id`, `revision_note` | Cited ids must exist; strategy must be buildable |
| `experiment_option` | `round`, `tests`, `design`, `plans`, `budget_pct`, scenario sets and sizes, `cost_sims`, `expected_learning`, `feasibility`, `chosen`, `why` | Cost = plans × ignitions × weather; controls included; a chosen option needs a rival in its round; warns if over budget |
| `finding` | `experiment_id`, `stats_id`, `summary`, `hypothesis_updates[{hypothesis_id, observed_change_pct, verdict}]`, `surprise`, `reopen`, `confidence` | Experiment, stats and hypotheses must exist |
| `decision` | `based_on`, `decision`, `rationale`, `next` | Every `based_on` id must exist |
| `risk_flag` | `plan_id`, `category`, `severity`, `description`, `mitigation` | Plan must exist |

Tool-written (agents cannot write these):

| Kind | Written by | Content |
|---|---|---|
| `plan` | `generate_plan` | Strategy, resolved params, budget, seed, treated cells/ha, mask hash, landscape fingerprint |
| `experiment` | `run_ensemble` | Spec, cost, status (`running` → `done` / `failed`), runtime, runs file, simulator version |
| `result` | `run_ensemble` | Per-plan metrics (mean/P90/SD burned ha, asset cells, P(asset loss)), seed, MLflow run id |
| `stats` | `compute_stats` | Full comparison table per metric, verdicts, efficiency |
| `recommendation` | `publish_recommendation`, `firelab approve/reject` | Plan, decision, caveats, risk flags, transfer-tested flag, supporting verdicts, status |
| `approval` | `publish_recommendation` (policy mode), `firelab approve/reject` | Who approved, how |

The simulation budget is the sum of `cost_sims` over experiments. An experiment reserves its cost
when it starts (`running`), and a failed run releases it (`cost_sims = 0`).

## Files next to the database

- `record/plans/<plan_id>.npy`: treated-cell masks (rebuilt from the spec if missing, then hash-checked).
- `record/runs/<experiment_id>.npz`: per-scenario metrics and burn-count maps.
- `mlruns/`: local MLflow store, unless `MLFLOW_TRACKING_URI` is set (e.g. `databricks`).

## Export

`firelab export` writes:

- `results/record_export.json`: every version of every item;
- `results/evidence.md`: the cited evidence list with links;
- `results/report.md`: hypotheses, options (chosen and rejected, with reasons), stats tables,
  findings, decisions, recommendations, loop timing.

`record/` is git-ignored; commit the `results/` exports with the submission. To start a fresh lab
run, move or delete `record/`.

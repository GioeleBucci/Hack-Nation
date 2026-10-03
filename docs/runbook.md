# Runbook

## 1. Install

Requires Python ≥ 3.10.

```bash
# The package, for the CLI and tests
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[all]"                                    # numpy, scipy, pydantic, requests (+ mlflow, matplotlib, pytest)

# Omnigent with the Databricks model provider, with this package importable by its tools and policies
uv tool install "omnigent[databricks]" --with-editable . --with mlflow
```

Omnigent imports `firelab.tools.*` and `firelab.policies.*` by dotted path, so the package must be
installed **into Omnigent's environment** (`--with-editable .`). The editable install also means the
tools write to this repo's `data/`, `record/` and `results/` folders (override with `FIRELAB_ROOT`).

Authenticate to Databricks with a CLI profile named `hackathon`, or change `auth.profile` in
`agents/lab_director.yaml`. Check which `databricks-claude-*` serving endpoints your workspace
exposes and edit the `model:` fields to match.

Smoke test: `omnigent run agents/lab_director.yaml -p "Call lab_status and summarize it."`

## 2. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `FIRELAB_ROOT` | repo root | Where `data/`, `record/`, `results/` live |
| `FIRELAB_LANDSCAPE` | `synthetic_v1` | Landscape name (`data/landscape_<name>.npz`) |
| `FIRELAB_MAX_SIMS` | `20000` | Lab simulation budget (keep equal to `sim_budget.max_sims` in the YAML) |
| `FIRELAB_MAX_SIMS_PER_CALL` | `6000` | Largest single `run_ensemble` |
| `FIRELAB_APPROVAL_MODE` | `policy` | `policy` (Omnigent ASK) or `manual` (`firelab approve`) |
| `FIRELAB_APPROVER` | `human scientist (Omnigent ASK)` | Name stored on policy-mode approvals |
| `FIRELAB_VERIFY_CITATIONS` | `1` | Resolve evidence source ids on OpenAlex |
| `FIRELAB_MLFLOW` | `1` | Log results to MLflow if it is installed |
| `MLFLOW_TRACKING_URI` | `./mlruns` | Set to `databricks` to log to the workspace |
| `OPENALEX_MAILTO` | — | Your email, for OpenAlex's polite pool |

## 3. Run the lab

```bash
firelab init          # ~500 simulations once: landscape + reference maps; prints a summary
firelab baseline      # round 0 by hand: none vs 5 random seeds, 50 ignitions x 5 weathers. TIME IT.
```

Check the calibration line at the end of `baseline` ([simulator.md](simulator.md#calibration-check)).

```bash
omnigent run agents/lab_director.yaml -p "Objective: find the best placement for a 5% treatment
budget on synthetic_v1, test whether it transfers to north-westerly winds, and bring me a
recommendation to approve."
```

What you should see: literature search → hypotheses → two or more options with costs and one chosen
→ plans built and an ensemble run → stats and a finding → a decision that sets the next round →
… → transfer test → safety flags → an ASK prompt asking you to approve the recommendation.

While it runs, `firelab status` shows the budget and record counts from another terminal.

## 4. After the run

```bash
firelab export                 # results/report.md, results/evidence.md, results/record_export.json
firelab figures EXP-R1-A       # results/figures/EXP-R1-A_maps.png and _effects.png (per experiment)
firelab bench                  # results/acceleration.json (~4,000 offline simulations)
```

Copy the numbers into the README's Results section, then commit `results/`.

## 5. Tests

```bash
pytest
```

The tests use a temporary root, the 60 × 60 landscape, no network and no MLflow (`tests/conftest.py`).
`tests/test_tools.py` runs one complete loop through the agent-facing tools.

## 6. Demo outline (2 minutes)

| Time | Show |
|---|---|
| 0:00–0:15 | Question and bottleneck (README) |
| 0:15–0:45 | Omnigent session: director delegating; planner choosing between two options under budget |
| 0:45–1:15 | Maps and effect chart (`firelab figures`); the finding and decision that changed the next round |
| 1:15–1:35 | Safety flags; the ASK approval prompt |
| 1:35–1:50 | Measured acceleration with its IQR (`results/acceleration.json`) |
| 1:50–2:00 | What the lab learned and the next experiment |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: firelab` inside Omnigent | Reinstall with `--with-editable .` from the repo root |
| Policies never fire / names differ | Check tool names in the session log; policies match on the name suffix |
| `Simulation budget exhausted` | Intended. Start a fresh record (move `record/`) or raise `FIRELAB_MAX_SIMS` and the YAML value together |
| `Plan … does not match its recorded hash` | The landscape or placement code changed after the plan was built; start a fresh record |
| Evidence rejected as non-existent | The agent invented or mangled an id; it must copy `source_id` from `openalex_search` |
| OpenAlex / Open-Meteo unreachable | Responses are cached in `data/cache/`; set `FIRELAB_VERIFY_CITATIONS=0` only for offline testing |
| Fires burn almost nothing or everything | Tune `SpreadParams.base_rate`, delete `data/reference_*.npz`, re-run `firelab init`, fresh record |
| Runs too slow | Use fewer ignitions for screening; keep large ensembles for confirmation and transfer |

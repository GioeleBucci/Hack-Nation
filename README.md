# Fire Lab: an agentic lab for wildfire fuel-treatment placement

Hack-Nation Challenge 03 (Agentic Scientific Discovery, Databricks × Omnigent).

A team of Omnigent agents reads the fire-science literature, proposes where to place prescribed
burns, tests those hypotheses in fire-spread simulations under a fixed simulation budget, and
changes its next experiment based on what it learns. A human approves anything that becomes a
recommendation.

```
Question → Evidence → Hypothesis → ≥2 rival experiments → chosen experiment → Result → Finding → Decision → (loop)
                                                                                                   └→ Safety review → human approval
```

## Scientific question

| Item | Definition |
|---|---|
| Question | With a fixed treatment budget (5% of the burnable area), which spatial placement pattern minimizes expected burned area and asset exposure, and does the best pattern still win under a different dominant wind? |
| Measurable outcome | Mean and P90 burned area (ha) and burned asset cells per plan, compared with no treatment and random placement at the same budget, over identical (paired) fire scenarios. |
| Bottleneck attacked | Evaluating treatment plans is slow and sequential, so analysts compare only a handful of alternatives. The lab screens more hypotheses per hour and adapts after each result. |
| Generalizable finding | A placement *rule* that transfers to held-out weather, not one optimized map. |

## Pre-registration (fixed before running agent experiments)

- **Primary metric:** mean burned area per scenario (ha). Secondary: burned asset cells.
- **Controls:** no treatment (`none`) and random placement (`random`, 3–5 seeds) at the same budget,
  in every experiment, under identical scenarios.
- **Success criterion:** a strategy *wins* if its mean burned area is ≥ 10% lower than the random
  control, the 95% paired-bootstrap CI of the relative change excludes zero, and the Holm-adjusted
  Wilcoxon signed-rank p < 0.05 (`firelab/stats.py`).
- **Transfer test:** the top strategies are re-run on held-out ignitions and the `heldout_nw` weather
  set (dominant wind rotated 90°). A rule that only wins on the training weather is reported as such.
- **Acceleration:** simulations needed to reach within 5% of the best reduction, agent vs random and
  grid search over the same strategy space (`firelab bench`); cycle time from hypothesis to evaluated
  result vs the timed manual round 0 (`docs/acceleration.md`).

## Quickstart

```bash
# 1. Omnigent with the Databricks provider, plus this package next to it
uv tool install "omnigent[databricks]" --with-editable . --with mlflow
pip install -e ".[all]"                       # same package for the CLI and tests

# 2. Frozen inputs and the manual round-0 baseline
firelab init                                  # landscape + no-treatment reference maps
firelab baseline                              # none vs random, by hand; time yourself

# 3. The agentic lab
omnigent run agents/lab_director.yaml -p "Find the best 5% treatment placement and test whether it transfers."

# 4. Results
firelab export                                # results/report.md, evidence.md, record_export.json
firelab figures EXP-R1-A                      # maps + effect-size chart for one experiment
firelab bench                                 # acceleration benchmark → results/acceleration.json
pytest                                        # unit + end-to-end tool tests
```

Full step-by-step instructions, environment variables and troubleshooting: [docs/runbook.md](docs/runbook.md).

## Repository layout

```
agents/lab_director.yaml   Omnigent spec: director, 6 specialist sub-agents, tools, policies
agents/prompts/director.md Director instructions (the loop, stopping rules)
firelab/
  landscape.py             Fuel / elevation / asset rasters; deterministic synthetic landscape
  scenarios.py             Frozen ignition and weather sets (train / held-out)
  simulator.py             Stochastic cellular-automaton fire spread with common random numbers
  reference.py             No-treatment burn-probability and spread-centrality maps
  placement.py             Placement strategies S0–S6
  stats.py                 Paired bootstrap, Wilcoxon, Holm, pre-registered verdict
  schemas.py               Pydantic handoff schemas (EvidenceCard, Hypothesis, ExperimentOption, ...)
  record.py                Append-only, versioned research record (SQLite)
  policies.py              Omnigent policies: simulation budget, record integrity, approval gate
  tools/                   Agent-facing tools (JSON in, JSON out, never raise)
  report.py, bench.py      Export, loop timing, acceleration benchmark
  viz.py, cli.py           Figures and the `firelab` command
docs/                      How everything works (start at docs/README.md)
tests/                     Unit tests and an end-to-end run through the tools
AGENTS.md                  Agent specifications and handoffs (submission artifact)
```

## Results

Filled in from `firelab export` and `firelab bench` after the run. Only numbers that are in the
exported record go here.

- Rounds run / simulations used: _TBD_
- Best strategy on training weather (change vs random, 95% CI): _TBD_
- Transfer result on `heldout_nw`: _TBD_
- The moment a result changed the plan (finding → decision ids): _TBD_
- Measured acceleration (with uncertainty): _TBD_
- Recommendation and approval status: _TBD_
- Next experiment and why: _TBD_

## Limitations and validation still needed

- The simulator is a simplified stochastic cellular automaton, not a validated fire-behaviour model
  (no spotting, crown fire, fuel moisture dynamics or suppression). See [docs/simulator.md](docs/simulator.md).
- The landscape is synthetic and the weather scenarios are simulated; results apply to this
  landscape until tested on others (e.g. LANDFIRE fuels with Cell2Fire).
- Treated cells are modelled as non-burnable. Real treatments reduce fuel partially and fuel regrows.
- Hypotheses are agent-generated. Before real-world use, results would need comparison with
  higher-fidelity simulators and historical fire perimeters, review by fire behaviour analysts,
  local fuel validation, smoke and ecological assessment, permitting and field trials.
- The lab outputs ranked candidate rules for expert review, not burn prescriptions.

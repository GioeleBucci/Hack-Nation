# Fire Lab — an agentic lab for wildfire fuel-treatment placement
Build plan for Hack-Nation Challenge 03 (Agentic Scientific Discovery, Databricks × Omnigent)

---

## 1. The pitch in one line

A team of Omnigent agents that reads the fire-science literature, proposes where to place prescribed burns, tests those hypotheses in thousands of fire simulations under a fixed budget, and changes its next experiment based on what it learns. A human approves anything that becomes a recommendation.

## 2. Question, outcome, bottleneck

| Item | Definition |
|---|---|
| Scientific question | With a fixed treatment budget (e.g. 5% of the landscape), which spatial placement pattern minimizes expected burned area and asset exposure, and does the best pattern still win under different wind and weather conditions? |
| Measurable outcome | Mean and P90 burned area (ha) and expected burned asset cells per plan, compared with no treatment and random placement at the same budget. |
| Bottleneck attacked | Evaluating treatment plans is slow and sequential, so analysts compare only a handful of alternatives. The lab screens more hypotheses per hour and adapts after each result. |
| Generalizable finding | A placement rule that transfers to held-out weather, not just one optimized map. |

Write these into `README.md` at hour 3, together with a pre-registered success criterion, before running agent experiments. For example: a strategy "wins" if it beats random placement by at least 10% in mean burned area, with a 95% paired-bootstrap CI that excludes zero.

## 3. Key technical decisions (make them in the first hour)

| Decision | Recommendation | Why |
|---|---|---|
| Omnigent route | Open-source Omnigent, with Databricks as the model provider: `uv tool install "omnigent[databricks]"` | Custom Python policies (simulation budget, citations, approval gate) are needed. Managed Omnigent on Databricks only supports built-in policies (plus CEL policies on Azure). |
| Models | Orchestrator on an Opus-class endpoint; specialists on Sonnet-class endpoints | Cost control. Check which `databricks-claude-*` endpoints your workspace exposes. |
| Simulator | Cell2Fire (open source, C++/Python, parallel, built for fuel-management studies) | Fast enough for thousands of runs. It natively evaluates "harvested cells", which you can use as treated cells. |
| Fuel models | Use the original Cell2Fire with its bundled sample instances (Canadian FBP fuels) to get the loop running. Only switch to the fire2a Cell2Fire-W fork if you want Scott & Burgan fuels from LANDFIRE. | Avoids a data-format fight in the first hours. |
| Fallback simulator | A ~150-line numpy cellular-automaton spread model (fuel, slope, wind) | Use it only if Cell2Fire hasn't produced burned-area numbers by hour 4. State its limits clearly. |
| Research record | Delta tables in Unity Catalog plus MLflow runs if you have a workspace; DuckDB plus MLflow locally otherwise | Every decision must be reconstructable. MLflow is native to Databricks, which judges will notice. |
| Weather | Open-Meteo historical API (ERA5-based, no key) | Much faster to start with than downloading from the Copernicus CDS. |
| Literature | OpenAlex API (free, no key; add your email for the polite pool) | Explicitly suggested in the brief. |

## 4. Architecture

```
Human scientist ── sets objective, approves recommendations (Omnigent ASK verdict)
        │
Lab director (Omnigent orchestrator, YAML spec)
        │  delegates via sub-agent tools, enforces policies
        ├── Literature agent    → EvidenceCards (cited)
        ├── Hypothesis agent    → Hypotheses (labeled agent-generated)
        ├── Experiment planner  → ≥2 ExperimentOptions → chosen ExperimentSpec
        ├── Experiment runner   → Results (Cell2Fire ensembles, MLflow run IDs)
        ├── Analysis agent      → Findings (effect sizes, CIs, surprises)
        └── Safety agent        → RiskFlags, burn-window feasibility, approval request
        │
Policies: LLM cost budget · tool-call cap · simulation budget · citation check · human approval
Tools:    openalex_search · generate_plan · run_ensemble · compute_stats · burn_window_days
          record_write · record_read · publish_recommendation
Record:   evidence · hypotheses · experiment_options · experiments · results · decisions · approvals
```

### Agent specifications

| Agent | Decision it owns | Tools | Input | Output |
|---|---|---|---|---|
| Lab director | Which loop step comes next, and when to stop | all sub-agents, `record_read` | Human objective, record state | Loop control, final report |
| Literature | What prior evidence says and where the gaps are | `openalex_search`, `record_write` | Research question | 5–15 EvidenceCards with OpenAlex IDs/DOIs |
| Hypothesis | Which placement strategies are worth testing | `record_read`, `record_write` | EvidenceCards, previous Findings | 3–5 Hypotheses with predicted effect and falsification criterion |
| Planner | Which experiment to run, given the budget | `record_read`, `record_write` | Hypotheses, remaining sim budget | At least 2 ExperimentOptions scored on expected learning, cost and feasibility; one chosen ExperimentSpec |
| Runner | How to execute correctly and reproducibly | `generate_plan`, `run_ensemble`, `record_write` | ExperimentSpec | Results with seeds and MLflow run IDs |
| Analysis | What the result means and whether it surprises us | `compute_stats`, `record_read`, `record_write` | Results, predictions | Findings: ranking, paired CIs, prediction-vs-observed, "reopen assumption?" flag |
| Safety | Whether a plan is safe and feasible to recommend | `burn_window_days`, `record_read`, `record_write`, `publish_recommendation` | Top plan, assets, weather | RiskFlags, feasibility score, approval request |

### Handoff schemas (validate with pydantic inside `record_write`)

```json
EvidenceCard   { "id": "E7", "claim": "...", "source_id": "openalex:W123... | doi:...",
                 "summary": "...", "confidence": "low|medium|high" }

Hypothesis     { "id": "H3", "statement": "Strips perpendicular to the dominant wind upwind of assets
                 reduce asset exposure more than fuel-load ranking at equal budget",
                 "label": "agent-generated", "evidence_ids": ["E2","E7"],
                 "predicted_effect": "≥15% lower asset exposure vs S1", "falsified_if": "CI overlaps 0",
                 "status": "proposed|testing|supported|rejected|reopened" }

ExperimentOption { "id": "X2-A", "tests": ["H1","H3","H4"], "design": "screen",
                   "plans": 6, "n_ignitions": 100, "n_weather": 5, "cost_sims": 3000,
                   "expected_learning": "...", "feasibility": "...", "chosen": true, "why": "..." }

Result         { "experiment_id": "X2-A", "plan_id": "S4-a", "metrics": { "mean_ba_ha": ..., "p90_ba_ha": ...,
                 "asset_cells_burned": ... }, "seed": 42, "mlflow_run_id": "...", "runtime_s": ... }

Decision       { "id": "D5", "based_on": ["X2-A"], "decision": "...", "rationale": "...",
                 "next": "X3-B", "approved_by": null }
```

## 5. Omnigent configuration

### Repository layout

```
fire-lab/
├── agents/
│   └── lab_director.yaml          # orchestrator + sub-agents + policies
├── prompts/
│   └── director.md                # long orchestrator instructions
├── firelab/                       # pip-installable package (pip install -e .)
│   ├── tools/
│   │   ├── literature.py          # openalex_search
│   │   ├── landscape.py           # load landscape, assets, burnable mask
│   │   ├── strategies.py          # generate_plan: S0..S6 generators
│   │   ├── simulate.py            # run_ensemble → Cell2Fire subprocess + MLflow
│   │   ├── stats.py               # compute_stats: paired bootstrap, Wilcoxon, Holm
│   │   ├── weather.py             # burn_window_days (Open-Meteo)
│   │   └── record.py              # record_write/read, publish_recommendation
│   └── policies.py                # simulation_budget, require_citations, approval_gate
├── data/                          # landscape, ignition sets, weather sets (frozen, versioned)
├── results/                       # figures, exported record
├── AGENTS.md                      # agent specifications for the submission
└── README.md                      # question, pre-registration, results, next experiment
```

Install your package next to Omnigent so the tools and policies are importable, e.g. `uv tool install "omnigent[databricks]" --with-editable .`, then smoke-test with `omnigent run agents/lab_director.yaml -p "Say hello"`.

### `agents/lab_director.yaml` (starting point; check field details against the Agent YAML spec)

```yaml
name: fire_lab_director
instructions: ../prompts/director.md

executor:
  harness: claude-sdk
  model: databricks-claude-opus-4-7        # use an endpoint your workspace exposes
  auth:
    type: databricks
    profile: hackathon

os_env:
  type: caller_process
  cwd: ..
  sandbox:
    type: auto
    write_paths: [./results, ./record]
    allow_network: true

tools:
  # ---------- domain tools ----------
  openalex_search:
    type: function
    description: Search OpenAlex. Returns work IDs, DOIs, titles, years, abstracts.
    callable: firelab.tools.literature.openalex_search
    parameters:
      type: object
      properties:
        query: {type: string}
        max_results: {type: integer}
      required: [query]

  generate_plan:
    type: function
    description: Build a treatment plan (set of treated cells) from a named strategy and parameters at a given budget.
    callable: firelab.tools.strategies.generate_plan
    parameters:
      type: object
      properties:
        strategy: {type: string}          # random | fuel_load | burn_prob | asset_buffer | wind_strips | path_centrality | hybrid
        params: {type: object}
        budget_pct: {type: number}
        seed: {type: integer}
      required: [strategy, budget_pct, seed]

  run_ensemble:
    type: function
    description: Simulate fires for each plan under the SAME ignition and weather sets. Returns per-plan metrics and MLflow run IDs.
    callable: firelab.tools.simulate.run_ensemble
    parameters:
      type: object
      properties:
        experiment_id: {type: string}
        plan_ids: {type: array, items: {type: string}}
        ignition_set: {type: string}
        weather_set: {type: string}
        n_ignitions: {type: integer}
        n_weather: {type: integer}
        seed: {type: integer}
      required: [experiment_id, plan_ids, ignition_set, weather_set, n_ignitions, n_weather, seed]

  compute_stats:
    type: function
    description: Paired comparison of plans vs controls (bootstrap CIs, Wilcoxon, Holm correction).
    callable: firelab.tools.stats.compute_stats
    parameters:
      type: object
      properties:
        experiment_id: {type: string}
        baseline_plan_id: {type: string}
      required: [experiment_id, baseline_plan_id]

  burn_window_days:
    type: function
    description: Count historical days per year meeting a burn prescription (RH, wind, temperature) for a region.
    callable: firelab.tools.weather.burn_window_days
    parameters:
      type: object
      properties:
        region_id: {type: string}
        prescription: {type: object}
      required: [region_id, prescription]

  record_write:
    type: function
    description: Append a validated item (evidence, hypothesis, experiment_option, experiment, result, decision) to the research record.
    callable: firelab.tools.record.record_write
    parameters:
      type: object
      properties:
        kind: {type: string}
        payload: {type: object}
      required: [kind, payload]

  record_read:
    type: function
    description: Query the research record by kind and/or IDs.
    callable: firelab.tools.record.record_read
    parameters:
      type: object
      properties:
        kind: {type: string}
        ids: {type: array, items: {type: string}}

  publish_recommendation:
    type: function
    description: Publish a treatment recommendation. Requires human approval.
    callable: firelab.tools.record.publish_recommendation
    parameters:
      type: object
      properties:
        plan_id: {type: string}
        decision_id: {type: string}
        caveats: {type: string}
      required: [plan_id, decision_id, caveats]

  # ---------- specialist sub-agents ----------
  literature_agent:
    type: agent
    description: Finds evidence and gaps on fuel-treatment placement. Returns cited EvidenceCards.
    prompt: |
      You are the literature specialist of a wildfire fuel-treatment lab. Search OpenAlex,
      extract claims about treatment placement effectiveness, and write each as an
      EvidenceCard with a real OpenAlex ID or DOI. Never invent sources. List open gaps.
    executor: {harness: claude-sdk, model: databricks-claude-sonnet-4-6}
    tools: {openalex_search: inherit, record_write: inherit}
    max_sessions: 2

  hypothesis_agent:
    type: agent
    description: Turns evidence and previous findings into testable placement hypotheses.
    prompt: |
      Propose 3-5 hypotheses about treatment placement. Each must name a strategy that
      generate_plan can build, cite evidence IDs, give a quantitative predicted effect and a
      falsification criterion, and be labeled "agent-generated". When the analyst flags a
      surprise, revise or reopen the relevant hypothesis explicitly.
    executor: {harness: claude-sdk, model: databricks-claude-sonnet-4-6}
    tools: {record_read: inherit, record_write: inherit}
    pass_history: true

  planner_agent:
    type: agent
    description: Chooses the next experiment under the simulation budget.
    prompt: |
      Read open hypotheses and the remaining simulation budget. Write at least TWO
      ExperimentOptions with cost in simulations (plans x ignitions x weather), expected
      learning and feasibility. Choose one, explain why, and write the ExperimentSpec.
      Always include the no-treatment and random controls under identical scenarios.
    executor: {harness: claude-sdk, model: databricks-claude-sonnet-4-6}
    tools: {record_read: inherit, record_write: inherit}

  runner_agent:
    type: agent
    description: Executes ExperimentSpecs reproducibly.
    prompt: |
      Build each plan with generate_plan, then call run_ensemble once per experiment with
      fixed seeds. Never change the ignition or weather sets between plans. Record results.
    executor: {harness: claude-sdk, model: databricks-claude-sonnet-4-6}
    os_env: inherit
    tools: {generate_plan: inherit, run_ensemble: inherit, record_write: inherit}

  analysis_agent:
    type: agent
    description: Interprets results statistically and flags surprises.
    prompt: |
      Run compute_stats against the random control. Report effect sizes with 95% CIs,
      compare observed vs predicted effects, and set reopen=true when a prediction fails
      or a control behaves unexpectedly. Preserve uncertainty; never overstate.
    executor: {harness: claude-sdk, model: databricks-claude-sonnet-4-6}
    tools: {compute_stats: inherit, record_read: inherit, record_write: inherit}

  safety_agent:
    type: agent
    description: Flags risks and requests human approval before any recommendation.
    prompt: |
      For the candidate plan, check proximity to assets and protected areas, count feasible
      burn days with burn_window_days, and list risks (escape, smoke, ecology, model limits).
      Only then call publish_recommendation with explicit caveats.
    executor: {harness: claude-sdk, model: databricks-claude-sonnet-4-6}
    tools: {burn_window_days: inherit, record_read: inherit, record_write: inherit, publish_recommendation: inherit}

# ---------- policies ----------
policies:
  llm_budget:
    type: function
    handler: omnigent.policies.builtins.cost.cost_budget
    factory_params:
      max_cost_usd: 30.00
      ask_thresholds_usd: [10.00, 20.00]
  tool_call_cap:
    type: function
    handler: omnigent.policies.builtins.safety.max_tool_calls_per_session
    factory_params:
      limit: 400
  sim_budget:
    type: function
    handler: firelab.policies.simulation_budget
    factory_params:
      max_sims: 20000
  citations:
    type: function
    handler: firelab.policies.require_citations
  human_approval:
    type: function
    handler: firelab.policies.approval_gate
```

Optional orchestration bonus: run one sub-agent on a different harness (e.g. the runner on Codex) to show Omnigent combining harnesses. Only do this if it works smoothly by hour 12.

### `firelab/policies.py`

```python
from omnigent.policies.schema import PolicyEvent, PolicyResponse


def _tool_name(event: PolicyEvent) -> str:
    if event["type"] != "tool_call":
        return ""
    return event["data"].get("name", "")   # check exact names in the session log; they may be prefixed


def simulation_budget(max_sims: int):
    """Factory: DENY run_ensemble calls that would exceed the lab's simulation budget."""
    def evaluate(event: PolicyEvent) -> PolicyResponse | None:
        if not _tool_name(event).endswith("run_ensemble"):
            return None
        args = event["data"]["arguments"]
        cost = len(args["plan_ids"]) * args["n_ignitions"] * args["n_weather"]
        used = event["session_state"].get("sims_used", 0)
        if used + cost > max_sims:
            return {
                "result": "DENY",
                "reason": (f"Simulation budget: {used}/{max_sims} used, request needs {cost}. "
                           "Planner must pick a cheaper experiment or stop."),
            }
        return {
            "result": "ALLOW",
            "state_updates": [{"key": "sims_used", "action": "increment", "value": cost}],
        }
    return evaluate


def require_citations(event: PolicyEvent) -> PolicyResponse | None:
    """Evidence needs a source; hypotheses must be labeled agent-generated."""
    if not _tool_name(event).endswith("record_write"):
        return None
    args = event["data"]["arguments"]
    kind, payload = args.get("kind"), args.get("payload", {})
    if kind == "evidence" and not payload.get("source_id"):
        return {"result": "DENY", "reason": "Evidence must carry an OpenAlex ID or DOI."}
    if kind == "hypothesis" and payload.get("label") != "agent-generated":
        return {"result": "DENY", "reason": "Hypotheses must be labeled 'agent-generated'."}
    return {"result": "ALLOW"}


def approval_gate(event: PolicyEvent) -> PolicyResponse | None:
    """Pause for a human before anything becomes a recommendation."""
    if _tool_name(event).endswith("publish_recommendation"):
        return {"result": "ASK",
                "reason": "A scientist must approve before a treatment plan is recommended."}
    return None
```

Defense in depth: `record_write` should also check that each OpenAlex ID resolves, and `run_ensemble` should refuse when the budget is exhausted. That way the guarantees still hold if you have to move to the managed route.

## 6. Experiment design (the part judges score as rigor)

Matched conditions. One frozen landscape. One frozen ignition set (N points, uniform over burnable cells, or weighted by NASA FIRMS historical detections). One frozen weather set (M fire-season scenarios, e.g. high-percentile fire weather days). Every plan is burned under exactly the same scenarios, so all comparisons are paired.

Treatment model. Treated cells become non-burnable via Cell2Fire's harvested-cells input. This is an idealization, since real prescribed burns reduce fuel rather than remove it and fuel regrows. State it as an explicit assumption.

Controls. S0 = no treatment. S-rand = random placement at the same budget, with 5 seeded replicates.

Candidate strategy generators in `strategies.py`. The hypothesis agent picks among these and tunes their parameters:

| ID | Strategy | Parameters |
|---|---|---|
| S1 | Highest fuel load / rate of spread | — |
| S2 | Highest burn probability from the S0 run | — |
| S3 | Buffer around assets | width, shape |
| S4 | Strips oriented relative to the dominant wind | angle, spacing, width |
| S5 | Cells most often on simulated spread paths (path centrality from S0) | top-k |
| S6 | Hybrids proposed by agents | weights |

Metrics. Mean and P90 burned area; asset cells burned; efficiency = (BA_S0 − BA_plan) / treated area.

Statistics. Paired bootstrap 95% CIs on the difference vs. random, Wilcoxon signed-rank, and Holm correction across plans. Report effect sizes, not only p-values.

Transfer test. Re-run the top 2–3 strategies on a held-out weather set (e.g. a different dominant wind direction). This is what turns an optimized map into a scientific claim, or refutes it.

Reproducibility. Seeds, simulator version, data hashes and MLflow run IDs are stored for every result.

## 7. The discovery loop to demonstrate

| Round | What happens | Decision it produces |
|---|---|---|
| 0 (manual, hour 2–4) | Team runs S0 vs random by hand. Time it. | Baseline numbers and the manual cycle time for the acceleration metric |
| 1 | Literature → EvidenceCards → H1–H4 → planner writes options, e.g. A: broad screen of 6 plans × 100 ignitions × 5 weather = 3,000 sims, or B: deep test of 2 plans × 400 × 5 = 4,000. Picks one and explains why. Runner → analyst. | Which strategies survive screening; which predictions failed |
| 2 | Hypothesis agent revises (parameter sweep of the leader, or a reopened assumption if the analyst flagged a surprise). Planner spends budget on the most informative test. | Refined best strategy with CIs |
| 3 | Transfer test on held-out weather. Safety agent checks assets, burn-window days, risks. Human approves or rejects via the ASK prompt. | Recommendation with caveats, or a rejection, plus the next experiment |

The demo needs one visible moment where a result changes the plan: a failed prediction, a control that behaves unexpectedly, or a strategy that doesn't transfer. Don't script the outcome; just make sure the loop can react to whatever happens.

## 8. Measuring acceleration (report what you observe)

1. Cycle time: wall-clock time from "new hypothesis" to "evaluated with CI". Manual (timed in round 0) vs the agent loop.
2. Search efficiency: number of simulations needed to reach within 5% of the best burned-area reduction found. Agent-guided vs random search vs grid search over the same strategy space, with 3–5 seeds each; report median and spread.
3. Throughput: hypotheses evaluated per hour, with controls and CIs.

State clearly which factor you measured (e.g. "3.2× fewer simulations, 95% CI …") and what would be needed to approach 10×: parallel runners on a Databricks cluster, larger landscapes, learned surrogate models of the simulator.

## 9. 24-hour timeline (aligned with the brief's 4 / 14 / 6 split)

| Hours | Goal | Checkpoint |
|---|---|---|
| 0–1 | Install Omnigent with the Databricks extra, authenticate, hello-world agent runs. Clone and build Cell2Fire (Docker if the native build fails). | `omnigent run … -p "Say hello"` works |
| 1–3 | Cell2Fire runs on a bundled sample landscape. Write `run_ensemble`. Run S0 vs random manually and time it. | Burned-area numbers per scenario |
| 3–4 | Go/no-go: simulator works or switch to the numpy fallback. Write the question, metrics and pre-registration into README. | Locked question and success criterion |
| 4–8 | Implement `generate_plan`, `compute_stats`, `record_*`, `openalex_search`. One single-agent director calling tools end to end. | One full loop, single agent |
| 8–12 | Split into the six sub-agents, add handoff schemas, policies and MLflow logging. | Multi-agent loop with policies firing |
| 12–18 | Rounds 1–2 running. Optionally swap in a real landscape. Start acceleration experiments in the background. | A result that changed the next decision, logged |
| 18–21 | Transfer test, safety review, human approval path, figures, `AGENTS.md`. | All evidence in the record |
| 21–24 | Freeze code at hour 22. Record the 2-minute demo. Write README results and next experiment. Submit. | Submission checklist complete |

## 10. Team split (merge roles if you are fewer)

| Role | Owns |
|---|---|
| Simulation lead | Cell2Fire build, `run_ensemble`, `generate_plan`, landscape and scenario data |
| Agent lead | Omnigent YAML, prompts, policies, handoffs, debugging the session |
| Science lead | Question, literature quality, experiment design, statistics, acceleration measurement |
| Demo lead | Figures (burn-probability maps before/after), record viewer, `AGENTS.md`, README, video |

## 11. Two-minute demo script

| Time | Show |
|---|---|
| 0:00–0:15 | The problem and the question; the bottleneck you attacked |
| 0:15–0:45 | The Omnigent session: director delegating to sub-agents; evidence → hypotheses → planner choosing between two options under budget |
| 0:45–1:15 | Simulation results (maps plus CI chart); the moment a result changes the next decision |
| 1:15–1:35 | Safety agent flags; the ASK approval prompt; the human approves or rejects |
| 1:35–1:50 | Measured acceleration with uncertainty |
| 1:50–2:00 | What the lab learned and the next experiment |

## 12. How the plan maps to the judging

| Criterion | Weight | Where you earn it |
|---|---|---|
| Omnigent orchestration | 30% | Six specialists with clear decision ownership, structured handoffs, four or five live policies (including a budget the planner must respect), adaptive replanning, optional multi-harness |
| Breakthrough potential | 25% | A generalizable placement rule tested for transfer, not a single map |
| Discovery acceleration | 20% | Measured cycle time and simulations-to-target vs baselines |
| Scientific rigor | 15% | Pre-registration, paired matched scenarios, controls, CIs, Holm correction, seeds and MLflow IDs |
| Creativity and responsibility | 10% | Safety agent, human approval gate, citation policy, labeled hypotheses, explicit limitations |

## 13. Submission checklist (from the brief)

- [ ] Repository with README (question, pre-registration, results, next experiment)
- [ ] Agent specifications and policies (`agents/*.yaml`, `firelab/policies.py`, `AGENTS.md`)
- [ ] Two-minute demo video
- [ ] Cited evidence (exported EvidenceCards with OpenAlex IDs/DOIs)
- [ ] Experiment code and results (exported record plus MLflow runs)
- [ ] Measured improvement with uncertainty
- [ ] Next experiment, justified by what the lab learned
- [ ] Hypotheses labeled agent-generated; uncertainty preserved; controls documented; approval gates documented
- [ ] Validation still needed before real-world use (section 15)

## 14. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Cell2Fire won't build | Use Docker; if still blocked at hour 4, use the numpy CA simulator |
| Fuel format mismatch (FBP vs Scott & Burgan) | Stay on bundled FBP instances; use Cell2Fire-W only if time allows |
| Custom Python policies unavailable on your route | Stay on open-source Omnigent; enforce budget and approval inside the tools as a backup |
| Agents invent citations | `require_citations` policy, plus `record_write` resolving every ID against OpenAlex |
| Simulations too slow | Smaller grid, fewer ignitions for screening, multiprocessing; larger ensembles only for the final transfer test |
| LLM spend | `cost_budget` policy; Sonnet-class models for specialists |

## 15. Limitations and validation still needed (state these in the README)

- Treated cells as non-burnable is an idealization. Real treatments reduce fuel partially and fuel regrows over years.
- Simulated ignitions and weather are not real fire seasons. Results apply to one landscape until tested on others.
- Before real-world use, results would need: comparison with higher-fidelity simulators and historical fire perimeters, review by fire behavior analysts, local fuel data validation, smoke and ecological impact assessment, permitting, and field trials.
- The lab outputs hypotheses and ranked candidates for expert review, not burn prescriptions.

## Useful links

- Omnigent: https://github.com/omnigent-ai/omnigent
- Agent YAML spec: https://github.com/omnigent-ai/omnigent/blob/main/docs/AGENT_YAML_SPEC.md
- Policies: https://github.com/omnigent-ai/omnigent/blob/main/docs/POLICIES.md
- Cell2Fire: https://github.com/cell2fire/Cell2Fire and https://cell2fire.readthedocs.io
- OpenAlex API: https://docs.openalex.org
- Open-Meteo historical weather: https://open-meteo.com/en/docs/historical-weather-api
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov
- LANDFIRE: https://landfire.gov

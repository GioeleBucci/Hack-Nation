# Statistics and experimental design

## Design

- **Matched conditions.** One frozen landscape, frozen ignition sets, frozen weather sets. Every plan
  in an experiment burns under exactly the same scenarios, so all comparisons are **paired**.
- **Common random numbers.** Within a scenario, the stochastic spread uses the same random field for
  every plan ([simulator.md](simulator.md#common-random-numbers)).
- **Controls in every experiment.** `none` (no treatment) and `random` (same budget, several seeds).
  The `ExperimentOption` schema refuses options without both.
- **Equal budget.** All plans treat the same share of burnable cells.
- **Held-out transfer test.** Different ignitions and a different dominant wind.
- **Reproducibility.** Seeds, scenario sets, landscape fingerprint, simulator version, plan mask
  hashes and MLflow run ids are stored with every result.

## `compute_stats` (`firelab/stats.py`, `firelab/tools/stats.py`)

For an experiment and a set of baseline plans (normally all random-control replicates):

1. **Baseline per scenario** = mean of the baseline plans in that scenario. Averaging the random
   replicates removes the luck of one particular random layout.
2. For each other plan, the paired differences `d_s = plan_s − baseline_s` over scenarios `s`.
3. **Effect size**: mean difference (ha) and relative change `100 · mean(d) / mean(baseline)`.
4. **95% paired bootstrap CI**: resample scenarios with replacement (2,000 resamples, fixed seed),
   recompute both the mean difference and the relative change, and take the 2.5/97.5 percentiles.
5. **Wilcoxon signed-rank test** on `d` (two-sided; zero differences dropped).
6. **Holm–Bonferroni correction** across all plans compared in that call.
7. **Verdict** per plan:

| Verdict | Rule |
|---|---|
| `wins` | relative change ≤ −10%, CI upper bound < 0, Holm p < 0.05 (the pre-registered criterion) |
| `better_below_threshold` | CI upper bound < 0 and Holm p < 0.05, but the effect is smaller than 10% |
| `worse` | CI lower bound > 0 and Holm p < 0.05 |
| `no_clear_difference` | anything else |

The same is computed for burned asset cells (`asset_cells`). When a no-treatment plan is present,
**efficiency** is also reported: hectares of burned area avoided per treated hectare.

## Reading results honestly

- Report the effect and its CI first; the p-value is secondary.
- A CI that spans zero is "inconclusive", not "no effect".
- The analysis agent compares each hypothesis's `predicted_change_pct` with the observed effect and
  CI. A large miss is a **surprise** and goes into the finding's `reopen` list.
- Screening on 40–60 ignitions gives wide CIs. Leaders should be confirmed on a larger ensemble
  (which reuses the same first scenarios plus new ones) before the transfer test.
- Many strategies tested across rounds means many comparisons. Holm controls this within one
  experiment, and the transfer test on held-out data is the guard against selecting a lucky winner
  across rounds.

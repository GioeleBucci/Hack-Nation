# Measuring acceleration

The brief asks for the improvement we actually observe, with its uncertainty. We measure three
things, all computed from recorded data.

## 1. Simulations to a near-best strategy (`firelab bench`, `firelab/bench.py`)

How much compute does it take to find a strategy within 5% of the best achievable reduction in
burned area?

1. **Ground truth.** Every configuration in a fixed strategy space (31 configurations: S1, S2, S5,
   6 asset-buffer variants, 18 wind-strip variants, 4 hybrids) is scored once on a common screening
   ensemble (default 40 `train` ignitions × 3 `train` weathers, seed 0). The score is the % reduction
   in mean burned area vs no treatment. This is an offline measurement; it does not touch the lab's
   record or budget.
2. **Target** = 95% of the best reduction found (in the space or by the agents).
3. **Random search**: the space in random order, repeated 20 times. Simulations until the first
   configuration that reaches the target, plus the no-treatment run. Reported as median and IQR.
4. **Grid search**: the space in its fixed enumeration order.
5. **Agent**: the lab's recorded experiments in order (excluding the manual round 0). Cumulative
   simulations the agents **actually spent**, including controls and larger ensembles, until an
   experiment contains a plan that reaches the target when scored on the same screening ensemble.

Output: `results/acceleration.json` with `speedup_vs_random_median`, its IQR over the random-search
repeats, and `speedup_vs_grid`. The agents' count is conservative: it includes controls and
confirmation runs that the search baselines do not pay for.

Run it after the lab: `firelab bench` (options `--n-ignitions`, `--n-weather`, `--repeats`).

## 2. Cycle time (`firelab export` → `results/report.md`)

Minutes from a hypothesis being proposed (its first version in the record) to the first finding
that evaluates it with a CI. The report gives the per-hypothesis values and the median.

Compare it with the **manual round 0**: run `firelab baseline` and time yourself from "we want to
compare X with random" to "we have a number with a CI", including building plans, running, and
reading the output. A realistic manual cycle without these tools (set up a simulator run, place
treatments by hand in GIS, run, collect outputs, compute statistics) is hours. Report the time you
measured, not an estimate.

## 3. Throughput

Hypotheses evaluated (with controls and CIs) per hour of lab time, from the same timestamps.

## What to say in the demo

State the factor and its spread exactly as measured, e.g. "the agents reached a near-best strategy
after N simulations vs a median of M (IQR …) for random search: M/N ×". Then state what would
move it toward 10×:

- parallel runners on a Databricks cluster (experiments are independent; the record already
  supports concurrent writers);
- a learned surrogate of the simulator to pre-screen plans before simulating;
- sequential testing that stops ensembles early once a CI is decisive;
- larger strategy spaces, where guided search gains more over random search.

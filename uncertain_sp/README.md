# Ranking shortest-path algorithms under uncertain edge weights

A small, runnable benchmark that implements the methods suggested from the survey, tests them on the
**same** uncertain graphs, and ranks them with a configurable scoring system.

```
graph_model.py   uncertain graph + sampler (lognormal costs, zone-correlated)
algorithms.py    7 methods, one per family (+ registry to add your own)
ranking.py       normalisation -> weighted score -> rank -> aggregation -> robustness check
benchmark.py     runs everything, prints tables, saves CSVs + chart
test_sanity.py   5 correctness tests (Dijkstra vs brute force, sampler, ranking, Hoeffding bound)
```

Needs: Python 3.9+, numpy, pandas, matplotlib.

## Run
```
python benchmark.py                        # 12 random graphs (30 and 50 nodes), balanced profile  (~10 s)
python benchmark.py --quick                # smoke test
python benchmark.py --profile risk_averse  # balanced | risk_averse | expected_cost | speed_first
python benchmark.py --rho 0.8 --deadline-factor 1.05   # stronger correlation, tighter budget
python benchmark.py --skip-rl              # drop Q-learning (slowest)
python test_sanity.py
```
Outputs go to `results/`: `instances.csv` (every algorithm x instance), `ranking_<profile>.csv`, `ranking_<profile>.png`.

## The algorithms
| Name | Family | Idea | Uses |
|---|---|---|---|
| Dijkstra (nominal) | classical | one crisp number per edge (median) | model |
| Expected-value Dijkstra | stochastic | Dijkstra on E[w] | model |
| Fuzzy Dijkstra | fuzzy | triangular fuzzy numbers, graded-mean ranking | model |
| Robust worst-case | robust | Dijkstra on upper bounds (min-max) | model |
| Robust regret (midpoint) | robust | Dijkstra on interval midpoints (min-max-regret heuristic) | model |
| Q-learning | learning | tabular Q-learning, reward = -observed cost | simulator only |
| Proposed (learned+bound) | proposed | learn mean/std from history, generate risk-adjusted candidate paths, pick the one with highest on-time probability, report a Hoeffding lower bound | historical samples only |

"Model" methods are handed the true uncertainty representation (as the papers assume); Q-learning and the
proposed method only see samples. All are scored on 2000 held-out sampled worlds.

**Proposed method guarantee:** with prob. >= 1 - delta (0.05), the true P(cost <= deadline) >= p_hat - eps,
eps = sqrt(ln(2m/delta) / (2n)). Candidates are generated from one half of the history and scored on the other,
so the union bound over the m candidates is valid. `test_sanity.py` checks the bound empirically.

## The ranking system
Metrics (per instance): `exp_ratio` (expected cost / mean oracle cost), `p95_ratio`, `ontime_prob`
(P(cost <= deadline)), `regret95` (95th-percentile regret / mean oracle cost), `runtime_ms` (log scale).

1. min-max normalise each metric across algorithms inside each instance (1 = best)
2. weighted sum with a profile (edit `PROFILES` in `ranking.py`) -> score
3. rank per instance, then aggregate: mean score (the league table order), mean rank, wins, Borda
4. `rank_stability`: P(#1) when the weights are drawn at random - shows whether the winner depends on your weights

## Add your own algorithm
Write `def my_method(ctx: Context) -> Route` (returns `Route(nodes, edges)`), then add it to `ALGORITHMS`
at the bottom of `algorithms.py`. It is benchmarked and ranked automatically.

## Things to know when you report results
* Fuzzy Dijkstra with a linear ranking (graded mean) picks the same path as Dijkstra on graded-mean weights, and
  here that is usually the expected-value path. Identical metrics are real, not a bug (tested in `test_sanity.py`).
* Per-instance min-max normalisation exaggerates small differences, so scores have wide error bars; look at the
  score, mean rank and the robustness panel together rather than any one of them.
* Runtime is cheap for everything except Q-learning, so runtime weight mostly penalises it.
* Results depend on the generator (`rho`, volatility, deadline). Run several settings before drawing conclusions.

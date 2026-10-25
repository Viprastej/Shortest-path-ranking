"""
ranking.py
----------
The ranking system.  Input: one row per (instance, algorithm) with the metrics below.

Pipeline
  1. Per instance, min-max normalise every metric across algorithms to [0, 1] (1 = best).
     (runtime is normalised on a log scale, because it spans orders of magnitude)
  2. Weighted sum of the normalised metrics  ->  score in [0, 1]      (weights = a "profile")
  3. Rank algorithms inside each instance (1 = best)
  4. Aggregate over instances: mean score, mean rank, #wins, Borda points
  5. Stability check: how often does each algorithm come out #1 if the weights are random?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# metric -> (direction, use_log_scale)
METRICS = {
    "exp_ratio":   ("min", False),   # expected cost / mean oracle cost
    "p95_ratio":   ("min", False),   # 95th-percentile cost / mean oracle cost
    "ontime_prob": ("max", False),   # P(cost <= deadline)
    "regret95":    ("min", False),   # 95th-percentile regret / mean oracle cost
    "runtime_ms":  ("min", True),    # compute time (incl. training / fitting)
}

PROFILES = {
    "balanced":      {"exp_ratio": .25, "p95_ratio": .25, "ontime_prob": .25, "regret95": .15, "runtime_ms": .10},
    "risk_averse":   {"exp_ratio": .10, "p95_ratio": .35, "ontime_prob": .35, "regret95": .20, "runtime_ms": .00},
    "expected_cost": {"exp_ratio": .70, "p95_ratio": .10, "ontime_prob": .10, "regret95": .10, "runtime_ms": .00},
    "speed_first":   {"exp_ratio": .20, "p95_ratio": .10, "ontime_prob": .10, "regret95": .10, "runtime_ms": .50},
}


def _normalise(values: pd.Series, direction: str, log: bool) -> pd.Series:
    v = np.log10(values.clip(lower=1e-9)) if log else values.astype(float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:                       # all algorithms tie on this metric
        return pd.Series(1.0, index=values.index)
    s = (v - lo) / (hi - lo)
    return 1.0 - s if direction == "min" else s


def normalised_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns n_<metric> (0..1, 1 = best) computed separately inside every instance."""
    out = df.copy()
    for m, (direction, log) in METRICS.items():
        out[f"n_{m}"] = out.groupby("instance")[m].transform(lambda s: _normalise(s, direction, log))
    return out


def _weights(profile) -> dict:
    w = PROFILES[profile] if isinstance(profile, str) else dict(profile)
    total = sum(w.values())
    return {m: w.get(m, 0.0) / total for m in METRICS}


def score_instances(df: pd.DataFrame, profile="balanced") -> pd.DataFrame:
    """Per-instance weighted score and rank (rank 1 = best)."""
    w = _weights(profile)
    out = normalised_metrics(df)
    out["score"] = sum(out[f"n_{m}"] * w[m] for m in METRICS)
    out["rank"] = out.groupby("instance")["score"].rank(ascending=False, method="min")
    return out


def aggregate(df: pd.DataFrame, profile="balanced") -> pd.DataFrame:
    """Final league table over all instances."""
    s = score_instances(df, profile)
    n_alg = s["algorithm"].nunique()
    g = s.groupby("algorithm")
    table = pd.DataFrame({
        "family":     g["family"].first(),
        "score":      g["score"].mean(),
        "score_std":  g["score"].std(),
        "mean_rank":  g["rank"].mean(),
        "wins":       g["rank"].apply(lambda r: int((r == 1).sum())),
        "borda":      g["rank"].apply(lambda r: float((n_alg - r).sum())),
    })
    for m in METRICS:                                                   # raw means, for reading
        table[m] = g[m].mean()
    table = table.sort_values("score", ascending=False)
    table.insert(0, "position", range(1, len(table) + 1))
    return table


def rank_stability(df: pd.DataFrame, n_draws: int = 5000, seed: int = 0) -> pd.Series:
    """P(algorithm is #1) when the metric weights are drawn at random (Dirichlet(1,...,1))."""
    norm = normalised_metrics(df)
    mean_norm = norm.groupby("algorithm")[[f"n_{m}" for m in METRICS]].mean()
    W = np.random.default_rng(seed).dirichlet(np.ones(len(METRICS)), size=n_draws)
    winners = (W @ mean_norm.values.T).argmax(axis=1)
    counts = np.bincount(winners, minlength=len(mean_norm)) / n_draws
    return pd.Series(counts, index=mean_norm.index, name="P(#1 | random weights)")

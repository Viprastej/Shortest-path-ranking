"""Sanity tests.  Run:  python test_sanity.py   (or: pytest -q)"""
import itertools

import numpy as np
import pandas as pd

from algorithms import Context, dijkstra, fuzzy_dijkstra, proposed_guarantee
from graph_model import make_random_graph, sample_worlds
from ranking import METRICS, aggregate, normalised_metrics, rank_stability


def _all_simple_path_costs(g, w, s, t):
    best = [float("inf")]

    def dfs(u, seen, cost):
        if u == t:
            best[0] = min(best[0], cost)
            return
        for v, e in g.adj[u]:
            if v not in seen:
                dfs(v, seen | {v}, cost + w[e])

    dfs(s, {s}, 0.0)
    return best[0]


def test_dijkstra_matches_brute_force():
    for seed in range(5):
        g = make_random_graph(n=9, k=3, seed=seed)
        w = np.random.default_rng(seed).uniform(1, 10, g.n_edges).tolist()
        d, nodes, edges = dijkstra(g.adj, g.n, g.source, g.target, w)
        assert abs(d - _all_simple_path_costs(g, w, g.source, g.target)) < 1e-9
        assert abs(d - sum(w[e] for e in edges)) < 1e-9


def test_fuzzy_equals_dijkstra_on_graded_mean():
    g = make_random_graph(n=25, seed=3)
    F = [tuple(map(float, r)) for r in g.fuzzy]
    _, _, e_f = fuzzy_dijkstra(g.adj, g.n, g.source, g.target, F)
    graded = [(a + 4 * b + c) / 6 for a, b, c in F]
    _, _, e_d = dijkstra(g.adj, g.n, g.source, g.target, graded)
    assert e_f == e_d      # linear ranking => same path (a known property of this ranking function)


def test_sampler_mean_and_correlation():
    g = make_random_graph(n=30, rho=0.6, seed=2)
    W = sample_worlds(g, 40000, np.random.default_rng(0))
    assert np.allclose(W.mean(axis=0) / g.mean, 1.0, atol=0.06)           # mean preserved
    c = np.corrcoef(np.log(W).T)
    same = [c[i, j] for i in range(g.n_edges) for j in range(i) if g.zone[i] == g.zone[j]]
    diff = [c[i, j] for i in range(g.n_edges) for j in range(i) if g.zone[i] != g.zone[j]]
    assert np.mean(same) > 0.3 and abs(np.mean(diff)) < 0.05              # correlated only inside a zone


def _toy_instances():
    rows = []
    for inst in ("a", "b"):
        for alg, vals in {"X": (1.0, 1.2, 0.9, 0.1, 1.0), "Y": (1.1, 1.4, 0.7, 0.3, 50.0), "Z": (1.1, 1.4, 0.7, 0.3, 5.0)}.items():
            rows.append(dict(instance=inst, algorithm=alg, family="f", **dict(zip(METRICS, vals))))
    return pd.DataFrame(rows)


def test_ranking_direction_and_ties():
    df = normalised_metrics(_toy_instances())
    x = df[df.algorithm == "X"].iloc[0]
    assert x["n_exp_ratio"] == 1.0 and x["n_ontime_prob"] == 1.0           # X best on cost & on-time
    assert df[df.algorithm == "Y"].iloc[0]["n_runtime_ms"] == 0.0           # slowest -> 0 (log scale)
    same = pd.DataFrame(dict(instance=["a", "a"], algorithm=["P", "Q"], family="f",
                             **{m: [2.0, 2.0] for m in METRICS}))
    assert (normalised_metrics(same)["n_exp_ratio"] == 1.0).all()           # all-equal metric -> no division by 0
    t = aggregate(_toy_instances(), "balanced")
    assert t.index[0] == "X" and set(t["position"]) == {1, 2, 3}
    assert abs(rank_stability(_toy_instances()).sum() - 1.0) < 1e-9


def test_hoeffding_bound_coverage():
    """The guaranteed lower bound must hold (almost) always:  lb <= true P(cost <= deadline)."""
    g = make_random_graph(n=30, seed=1)
    _, _, ev = dijkstra(g.adj, g.n, g.source, g.target, g.mean.tolist())
    deadline = 1.10 * g.mean[ev].sum()
    big = sample_worlds(g, 30000, np.random.default_rng(999))
    violations, reps = 0, 40
    for r in range(reps):
        rng = np.random.default_rng(r)
        hist = sample_worlds(g, 800, rng)
        route = proposed_guarantee(Context(g, deadline, hist, rng, lambda m: None))
        true_p = (big[:, route.edges].sum(axis=1) <= deadline).mean()
        violations += route.guarantee_lb > true_p + 0.01                   # +0.01 = MC error of true_p
    assert violations <= 2, f"bound violated in {violations}/{reps} runs"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

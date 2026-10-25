"""
algorithms.py
-------------
Seven shortest-path methods, one (or more) per family from the survey.

Every method has the same signature:   solve(ctx: Context) -> Route
so adding your own is just: write a function, add it to ALGORITHMS at the bottom.

Information each method is allowed to use
    model-based (given the uncertainty representation, as in the papers):
        Dijkstra (nominal), Expected-value, Fuzzy, Robust worst-case, Robust regret (midpoint)
    data-driven (sees only samples, never the true parameters):
        Q-learning (interacts with a simulator), Proposed (uses historical samples ctx.hist)
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from graph_model import UncertainGraph

INF = float("inf")


# ----------------------------------------------------------------------------- data classes
@dataclass
class Route:
    nodes: list
    edges: list                              # edge ids along the path
    confidence: Optional[float] = None       # method's own estimate of P(cost <= deadline)
    guarantee_lb: Optional[float] = None     # provable lower bound on that probability
    info: dict = field(default_factory=dict)


@dataclass
class Context:
    g: UncertainGraph
    deadline: float                          # user's cost budget for the source->target trip
    hist: np.ndarray                         # historical observations (n_hist x E)
    rng: np.random.Generator
    sim: Callable[[int], np.ndarray]         # environment simulator (for RL)
    cfg: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- shortest paths
def dijkstra(adj, n, s, t, w):
    """Classic Dijkstra on scalar edge weights w (list indexed by edge id)."""
    dist = [INF] * n
    prev = [None] * n
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == t:
            break
        for v, e in adj[u]:
            nd = d + w[e]
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, e)
                heapq.heappush(pq, (nd, v))
    return dist[t], *_rebuild(prev, s, t)


def _rebuild(prev, s, t):
    nodes, edges, u = [t], [], t
    while u != s:
        u, e = prev[u]
        nodes.append(u)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def fuzzy_dijkstra(adj, n, s, t, F):
    """
    Dijkstra on triangular fuzzy numbers (a, b, c).
      addition : component-wise
      ranking  : graded mean  R = (a + 4b + c) / 6   (Deng et al. style)
    Returns (fuzzy_cost, nodes, edges).
    """
    rank = lambda x: (x[0] + 4 * x[1] + x[2]) / 6.0
    dist = [None] * n
    best = [INF] * n
    prev = [None] * n
    dist[s], best[s] = (0.0, 0.0, 0.0), 0.0
    pq = [(0.0, s)]
    while pq:
        r, u = heapq.heappop(pq)
        if r > best[u]:
            continue
        if u == t:
            break
        for v, e in adj[u]:
            cand = tuple(dist[u][i] + F[e][i] for i in range(3))
            rc = rank(cand)
            if rc < best[v]:
                best[v], dist[v], prev[v] = rc, cand, (u, e)
                heapq.heappush(pq, (rc, v))
    return dist[t], *_rebuild(prev, s, t)


def _scalar_route(ctx: Context, w: np.ndarray, **info) -> Route:
    g = ctx.g
    _, nodes, edges = dijkstra(g.adj, g.n, g.source, g.target, list(map(float, w)))
    return Route(nodes, edges, info=info)


# ----------------------------------------------------------------------------- 1-5: model-based
def dijkstra_nominal(ctx: Context) -> Route:
    """Classical baseline: one crisp number per edge (the median / 'typical' value)."""
    return _scalar_route(ctx, ctx.g.median)


def expected_value(ctx: Context) -> Route:
    """Stochastic family: least-expected-weight path (Dijkstra on E[w])."""
    return _scalar_route(ctx, ctx.g.mean)


def fuzzy(ctx: Context) -> Route:
    """Fuzzy family: triangular fuzzy numbers + graded-mean ranking."""
    g = ctx.g
    F = [tuple(map(float, row)) for row in g.fuzzy]
    cost, nodes, edges = fuzzy_dijkstra(g.adj, g.n, g.source, g.target, F)
    return Route(nodes, edges, info={"fuzzy_cost": tuple(round(c, 2) for c in cost)})


def robust_worst_case(ctx: Context) -> Route:
    """Robust family: minimise the worst-case total (every edge at its upper bound)."""
    return _scalar_route(ctx, ctx.g.hi)


def robust_regret_midpoint(ctx: Context) -> Route:
    """Robust family: midpoint-scenario heuristic commonly used for min-max regret."""
    return _scalar_route(ctx, (ctx.g.lo + ctx.g.hi) / 2.0)


# ----------------------------------------------------------------------------- 6: learning
def q_learning(ctx: Context) -> Route:
    """
    Tabular Q-learning, state = current node, action = next neighbour, reward = -observed cost.
    The agent sees only the cost of the edge it actually traverses in each simulated episode.
    """
    g, rng = ctx.g, ctx.rng
    episodes = ctx.cfg.get("q_episodes", max(3000, 100 * g.n))
    alpha, gamma = ctx.cfg.get("q_alpha", 0.15), 1.0
    eps0, max_steps = 0.3, 4 * g.n
    Q = [[0.0] * len(g.adj[u]) for u in range(g.n)]

    for ep in range(episodes):
        w = ctx.sim(1)[0]
        eps = max(0.02, eps0 * (1.0 - ep / episodes))
        u, steps = g.source, 0
        while u != g.target and steps < max_steps:
            nb = g.adj[u]
            if rng.random() < eps:
                a = int(rng.integers(len(nb)))
            else:
                q = Q[u]
                m = max(q)
                a = int(rng.choice([i for i, x in enumerate(q) if x == m]))
            v, e = nb[a]
            target_val = -w[e] + (0.0 if v == g.target else gamma * max(Q[v]))
            Q[u][a] += alpha * (target_val - Q[u][a])
            u, steps = v, steps + 1

    # greedy policy extraction (never revisit a node)
    u, nodes, edges, seen = g.source, [g.source], [], {g.source}
    while u != g.target:
        order = sorted(range(len(g.adj[u])), key=lambda i: -Q[u][i])
        nxt = next(((g.adj[u][i]) for i in order if g.adj[u][i][0] not in seen), None)
        if nxt is None:                                   # policy got stuck -> fall back
            r = expected_value(ctx)
            r.info["fallback"] = True
            return r
        v, e = nxt
        nodes.append(v), edges.append(e), seen.add(v)
        u = v
    return Route(nodes, edges, info={"episodes": episodes})


# ----------------------------------------------------------------------------- 7: proposed
def proposed_guarantee(ctx: Context) -> Route:
    """
    Data-driven routing with a provable on-time guarantee.

      1. Split the historical samples H into H1 (model fitting) and H2 (validation).
      2. From H1 estimate mean_hat and std_hat per edge  ->  *learned* weight model.
      3. Generate candidate paths = Dijkstra on (mean_hat + lambda * std_hat) for several
         risk-aversion levels lambda.
      4. Score every candidate on H2 *jointly* (so spatial correlation is captured):
             p_hat(path) = fraction of H2 worlds in which cost(path) <= deadline
      5. Return the candidate with the highest p_hat, plus the Hoeffding lower bound
             P_true(cost <= deadline)  >=  p_hat - eps,   eps = sqrt( ln(2m/delta) / (2|H2|) )
         valid for all m candidates simultaneously with probability >= 1 - delta
         (the candidates depend only on H1, which is independent of H2, so the union bound applies).
    """
    g, H = ctx.g, ctx.hist
    delta = ctx.cfg.get("delta", 0.05)
    lambdas = ctx.cfg.get("lambdas", (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0))
    h = len(H) // 2
    H1, H2 = H[:h], H[h:]
    mu, sd = H1.mean(axis=0), H1.std(axis=0, ddof=1)

    cands = {}
    for lam in lambdas:
        _, nodes, edges = dijkstra(g.adj, g.n, g.source, g.target, list(map(float, mu + lam * sd)))
        cands.setdefault(tuple(edges), (nodes, edges, lam))

    best = None
    for edges_t, (nodes, edges, lam) in cands.items():
        c = H2[:, edges].sum(axis=1)
        p_hat, mean_c = float((c <= ctx.deadline).mean()), float(c.mean())
        key = (p_hat, -mean_c)
        if best is None or key > best[0]:
            best = (key, nodes, edges, lam, p_hat)

    _, nodes, edges, lam, p_hat = best
    eps = math.sqrt(math.log(2 * len(cands) / delta) / (2 * len(H2)))
    return Route(nodes, edges, confidence=p_hat, guarantee_lb=max(0.0, p_hat - eps),
                 info={"lambda": lam, "n_candidates": len(cands), "eps": round(eps, 3), "delta": delta})


# ----------------------------------------------------------------------------- registry
# name -> (function, family).  Add your own method here and it is benchmarked + ranked automatically.
ALGORITHMS = {
    "Dijkstra (nominal)":        (dijkstra_nominal,       "classical"),
    "Expected-value Dijkstra":   (expected_value,         "stochastic"),
    "Fuzzy Dijkstra":            (fuzzy,                  "fuzzy"),
    "Robust worst-case":         (robust_worst_case,      "robust"),
    "Robust regret (midpoint)":  (robust_regret_midpoint, "robust"),
    "Q-learning":                (q_learning,             "learning"),
    "Proposed (learned+bound)":  (proposed_guarantee,     "proposed"),
}

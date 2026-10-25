"""
graph_model.py
--------------
Uncertain road-network model + "world" sampler used by every algorithm and by the benchmark.

Each edge e has a true cost distribution (lognormal, mean-preserving).  Edges that fall in the
same spatial zone share a common "congestion" factor, so their costs are positively correlated
(this is the correlated-uncertainty gap from the survey).

The single underlying model can be *viewed* in the three representations the survey talks about:
    - probabilistic : mean / std                      (stochastic family)
    - interval      : [lo, hi]  (5th-95th percentile) (robust / interval family)
    - fuzzy         : triangular (lo, median, hi)     (fuzzy family)
so every algorithm is evaluated against exactly the same ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z90 = 1.6448536269514722  # 95th percentile of N(0,1)


@dataclass
class UncertainGraph:
    n: int
    edges: list            # edges[e] = (u, v)   (undirected)
    adj: list              # adj[u] = [(v, e), ...]
    mean: np.ndarray       # E[w_e]
    sigma: np.ndarray      # log-sigma of each edge
    zone: np.ndarray       # congestion zone id of each edge
    n_zones: int
    rho: float             # within-zone correlation of the log-noise
    source: int
    target: int
    coords: np.ndarray

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    # ---- three views of the same uncertainty --------------------------------------------
    @property
    def median(self) -> np.ndarray:
        return self.mean * np.exp(-self.sigma ** 2 / 2)

    @property
    def std(self) -> np.ndarray:
        return self.mean * np.sqrt(np.expm1(self.sigma ** 2))

    @property
    def lo(self) -> np.ndarray:
        return self.mean * np.exp(-self.sigma ** 2 / 2 - Z90 * self.sigma)

    @property
    def hi(self) -> np.ndarray:
        return self.mean * np.exp(-self.sigma ** 2 / 2 + Z90 * self.sigma)

    @property
    def fuzzy(self) -> np.ndarray:
        """Triangular fuzzy numbers (a, b, c) = (lo, median, hi), shape (E, 3)."""
        return np.stack([self.lo, self.median, self.hi], axis=1)


def sample_worlds(g: UncertainGraph, m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw m i.i.d. realisations of all edge costs -> array of shape (m, E)."""
    zf = rng.standard_normal((m, g.n_zones))
    ef = rng.standard_normal((m, g.n_edges))
    latent = np.sqrt(g.rho) * zf[:, g.zone] + np.sqrt(1.0 - g.rho) * ef   # N(0,1) marginals
    return g.mean * np.exp(g.sigma * latent - g.sigma ** 2 / 2)           # E[w_e] = mean_e


def make_random_graph(n: int = 40, k: int = 4, rho: float = 0.5, grid: int = 3, seed: int = 0,
                      base_sigma=(0.05, 0.35), zone_vol=(0.3, 2.0)) -> UncertainGraph:
    """Random geometric graph in the unit square; source bottom-left, target top-right."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

    edge_set = set()
    for i in range(n):
        for j in np.argsort(d[i])[1:k + 1]:
            edge_set.add((min(i, int(j)), max(i, int(j))))

    # make sure the graph is connected
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edge_set:
        parent[find(u)] = find(v)
    while len({find(x) for x in range(n)}) > 1:
        best = None
        for u in range(n):
            for v in range(u + 1, n):
                if find(u) != find(v) and (best is None or d[u, v] < best[0]):
                    best = (d[u, v], u, v)
        _, u, v = best
        edge_set.add((u, v))
        parent[find(u)] = find(v)

    edges = sorted(edge_set)
    adj = [[] for _ in range(n)]
    for e, (u, v) in enumerate(edges):
        adj[u].append((v, e))
        adj[v].append((u, e))

    mean = np.array([100.0 * d[u, v] + 2.0 for u, v in edges])
    mid = np.array([(pts[u] + pts[v]) / 2 for u, v in edges])
    gx = np.minimum((mid[:, 0] * grid).astype(int), grid - 1)
    gy = np.minimum((mid[:, 1] * grid).astype(int), grid - 1)
    zone = gx + grid * gy
    vol = rng.uniform(*zone_vol, size=grid * grid)                 # some zones are much more volatile
    sigma = np.clip(rng.uniform(*base_sigma, size=len(edges)) * vol[zone], 0.03, 0.9)

    source = int(np.argmin(np.linalg.norm(pts - np.array([0.0, 0.0]), axis=1)))
    target = int(np.argmin(np.linalg.norm(pts - np.array([1.0, 1.0]), axis=1)))
    return UncertainGraph(n, edges, adj, mean, sigma, zone, grid * grid, rho, source, target, pts)

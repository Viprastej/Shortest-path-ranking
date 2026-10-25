"""
benchmark.py
------------
Runs every algorithm in algorithms.ALGORITHMS on many random uncertain graphs, evaluates the returned
paths on HELD-OUT sampled worlds, then ranks the algorithms (see ranking.py).

Examples
    python benchmark.py                          # default benchmark, balanced profile
    python benchmark.py --quick                  # small & fast (skips nothing, fewer instances)
    python benchmark.py --profile risk_averse    # change what "best" means
    python benchmark.py --skip-rl                # drop Q-learning (the slowest method)
    python benchmark.py --rho 0.0                # independent edges  vs  --rho 0.8 strong correlation
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import ALGORITHMS, Context, dijkstra
from graph_model import make_random_graph, sample_worlds
from ranking import PROFILES, aggregate, rank_stability


def run_instance(n, seed, args):
    g = make_random_graph(n=n, rho=args.rho, seed=seed)
    rng_hist = np.random.default_rng(10_000 + seed)
    rng_test = np.random.default_rng(20_000 + seed)
    rng_env = np.random.default_rng(30_000 + seed)
    rng_algo = np.random.default_rng(40_000 + seed)

    hist = sample_worlds(g, args.n_hist, rng_hist)          # what a data-driven method may observe
    test = sample_worlds(g, args.n_test, rng_test)          # held-out worlds used only for scoring

    # the user's budget: a buffer on top of the true expected cost of the least-expected-weight path
    _, _, ev_edges = dijkstra(g.adj, g.n, g.source, g.target, g.mean.tolist())
    deadline = args.deadline_factor * float(g.mean[ev_edges].sum())

    oracle = np.array([dijkstra(g.adj, g.n, g.source, g.target, row.tolist())[0] for row in test])
    mo = oracle.mean()

    rows = []
    for name, (fn, family) in ALGORITHMS.items():
        if args.skip_rl and family == "learning":
            continue
        ctx = Context(g, deadline, hist, rng_algo, lambda m: sample_worlds(g, m, rng_env),
                      cfg={"q_episodes": args.q_episodes} if args.q_episodes else {})
        t0 = time.perf_counter()
        route = fn(ctx)
        runtime_ms = (time.perf_counter() - t0) * 1000

        cost = test[:, route.edges].sum(axis=1)
        regret = cost - oracle
        ontime = float((cost <= deadline).mean())
        rows.append(dict(
            instance=f"n{n}-s{seed}", n=n, seed=seed, algorithm=name, family=family,
            exp_ratio=cost.mean() / mo, p95_ratio=np.percentile(cost, 95) / mo,
            ontime_prob=ontime, regret95=np.percentile(regret, 95) / mo, runtime_ms=runtime_ms,
            hops=len(route.edges),
            claimed_conf=route.confidence, guarantee_lb=route.guarantee_lb,
            bound_valid=(None if route.guarantee_lb is None else route.guarantee_lb <= ontime),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[30, 50], help="graph sizes (nodes)")
    ap.add_argument("--seeds", type=int, default=6, help="random graphs per size")
    ap.add_argument("--rho", type=float, default=0.5, help="within-zone correlation of edge costs (0..1)")
    ap.add_argument("--deadline-factor", type=float, default=1.10, help="budget = factor * E[cost] of the EV path")
    ap.add_argument("--n-hist", type=int, default=800, help="historical samples given to data-driven methods")
    ap.add_argument("--n-test", type=int, default=2000, help="held-out worlds used for evaluation")
    ap.add_argument("--q-episodes", type=int, default=0, help="Q-learning episodes (0 = auto)")
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES), help="ranking weight profile")
    ap.add_argument("--skip-rl", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 sizes x 2 seeds, fewer samples")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.n_test, args.q_episodes = [25, 35], 2, 1000, 1500

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows = []
    for n in args.sizes:
        for seed in range(args.seeds):
            t0 = time.time()
            rows += run_instance(n, seed, args)
            print(f"  instance n={n} seed={seed} done ({time.time() - t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "instances.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    cols = ["position", "family", "score", "mean_rank", "wins", "exp_ratio", "p95_ratio",
            "ontime_prob", "regret95", "runtime_ms"]
    table = aggregate(df, args.profile)
    table.to_csv(out / f"ranking_{args.profile}.csv")
    print(f"\n=== FINAL RANKING  (profile: {args.profile}, {df['instance'].nunique()} instances) ===")
    print(table[cols].round(3).to_string())

    print("\n=== Position under every weight profile ===")
    matrix = pd.DataFrame({p: aggregate(df, p)["position"] for p in PROFILES}).sort_values(args.profile)
    print(matrix.to_string())

    stab = rank_stability(df)
    print("\n=== P(#1) when metric weights are random (ranking robustness) ===")
    print(stab.sort_values(ascending=False).round(3).to_string())

    prop = df[df["bound_valid"].notna()]
    if len(prop):
        print("\n=== Proposed method: confidence & guarantee check ===")
        print(f"  mean claimed on-time prob (validation) : {prop['claimed_conf'].mean():.3f}")
        print(f"  mean guaranteed lower bound            : {prop['guarantee_lb'].mean():.3f}")
        print(f"  mean true on-time prob (held-out)      : {prop['ontime_prob'].mean():.3f}")
        print(f"  bound held on {int(prop['bound_valid'].sum())}/{len(prop)} instances")

    make_chart(table, stab, args, out / f"ranking_{args.profile}.png")
    print(f"\nSaved: {out}/instances.csv, ranking_{args.profile}.csv, ranking_{args.profile}.png")


def make_chart(table, stab, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"classical": "#9aa5b1", "stochastic": "#5C8BB8", "fuzzy": "#8e7cc3",
              "robust": "#2F5D8C", "learning": "#6aa84f", "proposed": "#E8A33D"}
    t = table.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    labels = [f"#{p}  {name}" for name, p in zip(t.index, t["position"])]
    ax.barh(labels, t["score"], xerr=t["score_std"].fillna(0), color=[colors[f] for f in t["family"]],
            error_kw={"ecolor": "#16273F", "capsize": 3, "lw": 1})
    ax.set_xlim(0, 1)
    ax.set_xlabel("weighted score (higher = better)")
    ax.set_title(f"Ranking - profile: {args.profile}", fontsize=11, loc="left", fontweight="bold")
    ax2 = axes[1]
    s2 = stab.reindex(t.index)
    ax2.barh(labels, s2.values, color="#CBDDF0", edgecolor="#2F5D8C")
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("P(ranked #1) under random weights")
    ax2.set_title("Ranking robustness", fontsize=11, loc="left", fontweight="bold")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()

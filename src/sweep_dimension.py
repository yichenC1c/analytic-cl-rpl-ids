"""Sweep the random-feature dimension D.

This characterises the state-performance trade-off. It is an analysis, not a selection
procedure; D is chosen by select_hyperparams.py on validation data.

Collapsing D towards zero also reproduces on network traffic the observation from the
analytic continual-learning literature that the nonlinear lift, rather than the
least-squares solve, is what makes the head work.
"""
import argparse

import numpy as np

import config as C
import data as D
import metrics as M
import pipeline as P
from analytic import RFF, RLSHead, median_gamma

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random", choices=C.SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ds", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    domains = D.load_all(C.DOMAIN_ORDERS[a.scenario])
    enc, _ = P.build_encoder(domains, a.seed, a.device)
    Ftr = [enc.features(d[1]) for d in domains]
    Fte = [enc.features(d[3]) for d in domains]
    gamma = median_gamma(Ftr[0], seed=a.seed)

    rows = []
    for Dd in a.Ds:
        rff = RFF(Ftr[0].shape[1], Dd, gamma, seed=a.seed)
        Zte = [rff(f) for f in Fte]
        head = RLSHead(Dd, C.OUTPUT_SIZE, C.RIDGE, C.RLS_BLOCK)
        for t, f in enumerate(Ftr):
            head.partial_fit(rff(f), P.onehot(domains[t][2], C.OUTPUT_SIZE))
        pr = [M.perf(domains[i][4], head.predict_scores(Zte[i])) for i in range(len(domains))]
        r = {"D": Dd,
             "f1": float(np.nanmean([p[0] for p in pr])),
             "auc": float(np.nanmean([p[1] for p in pr])),
             "state_kb": head.state_bytes / 2**10,
             "rff_kb": rff.n_bytes / 2**10,
             "flops_per_update": 2 * Dd * Dd + Dd * Ftr[0].shape[1]}
        rows.append(r)
        print(f"  D={Dd:5d}  F1 {r['f1']:.4f}  AUC {r['auc']:.4f}  state {r['state_kb']:8.1f} KB")
    P.save({"scenario": a.scenario, "seed": a.seed, "gamma": gamma, "sweep": rows},
           f"E5_dsweep_{a.scenario}_s{a.seed}.json")

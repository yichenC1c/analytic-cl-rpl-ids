"""Exact-kernel baseline, refitted from scratch on all accumulated data per domain.

This is the classifier that hybrid sequential detectors place on top of a deep encoder.
Its Gram matrix is O(N^2), so a configured sample cutoff stops dense fitting after a few
domains. The run records the cutoff rather than implying that an allocation was attempted
or that the operating system reported an out-of-memory failure.
"""
import argparse
import time

import numpy as np

import config as C
import data as D
import metrics as M
import pipeline as P
from analytic import ExactKELM, median_gamma

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random", choices=C.SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_n", type=int, default=35000,
                    help="configured cumulative-sample cutoff for dense fitting")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    domains = D.load_all(C.DOMAIN_ORDERS[a.scenario])
    enc, _ = P.build_encoder(domains, a.seed, a.device)
    Ftr = [enc.features(d[1]) for d in domains]
    Fte = [enc.features(d[3]) for d in domains]
    gamma = median_gamma(Ftr[0], seed=a.seed)

    kelm = ExactKELM(gamma, C.RIDGE, a.max_n)
    rows, T = [], len(domains)
    Mf1 = np.full((T, T), np.nan)
    Mauc = np.full((T, T), np.nan)
    for t in range(T):
        t0 = time.perf_counter()
        ok, N = kelm.add_and_refit(Ftr[t], P.onehot(domains[t][2], C.OUTPUT_SIZE))
        dt = time.perf_counter() - t0
        if not ok:
            print(f"  domain {t+1}: cumulative N={N} exceeds {a.max_n}, stopping")
            rows.append({"t": t + 1, "n_cum": N, "status": "cutoff",
                         "kernel_gb": N * N * 8 / 2**30})
            break
        for i in range(T):
            Mf1[t, i], Mauc[t, i] = M.perf(domains[i][4], kelm.predict_scores(Fte[i]))
        rows.append({"t": t + 1, "domain": domains[t][0], "n_cum": N, "status": "ok",
                     "refit_s": dt, "state_mb": kelm.state_bytes / 2**20,
                     "kernel_gb": N * N * 8 / 2**30, "f1_avg": float(np.nanmean(Mf1[t]))})
        print(f"  domain {t+1:2d}  N={N:6d}  Gram {N*N*8/2**30:5.2f} GB  "
              f"refit {dt:6.1f}s  mean F1 {np.nanmean(Mf1[t]):.4f}")

    surv = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nsurvived {surv}/48 domains within a {a.max_n}-sample budget")
    P.save({"scenario": a.scenario, "seed": a.seed, "gamma": gamma, "max_n": a.max_n,
            "domains_survived": surv, "trace": rows,
            "matrix_f1": Mf1, "matrix_auc": Mauc},
           f"E3_kelm_{a.scenario}_s{a.seed}.json")

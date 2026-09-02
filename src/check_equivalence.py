"""Verify that the recursive update reproduces the joint-batch solution.

Tracks ||W_recursive - W_joint||_F / ||W_joint||_F across the 48-domain sequence. In
float64 this stays at rounding level, which is the empirical counterpart of the
proposition. Run this at float64; at single precision the accumulated error over 48
Woodbury updates swamps the quantity being measured.
"""
import argparse

import numpy as np

import config as C
import data as D
import pipeline as P
from analytic import RLSHead, JointHead

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random", choices=C.SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=C.RFF_D)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    domains = D.load_all(C.DOMAIN_ORDERS[a.scenario])
    enc, _ = P.build_encoder(domains, a.seed, a.device)
    rff, gamma = P.make_rff(enc, domains, a.D, a.seed)

    head = RLSHead(a.D, C.OUTPUT_SIZE, C.RIDGE, C.RLS_BLOCK)
    jh = JointHead(a.D, C.OUTPUT_SIZE, C.RIDGE)
    rows = []
    for t, (name, Xtr, ytr, _, _) in enumerate(domains):
        Z = rff(enc.features(Xtr))
        Y = P.onehot(ytr, C.OUTPUT_SIZE)
        head.partial_fit(Z, Y)
        jh.accumulate(Z, Y)
        Wj = jh.solve()
        abs_e = float(np.linalg.norm(head.W - Wj))
        rel_e = abs_e / max(float(np.linalg.norm(Wj)), 1e-300)
        rows.append({"t": t + 1, "domain": name, "abs": abs_e, "rel": rel_e,
                     "n_cum": int(sum(len(d[1]) for d in domains[:t + 1]))})
        print(f"  domain {t+1:2d}/48  {name:<26} ||dW||={abs_e:.3e}  rel={rel_e:.3e}")

    print(f"\nmax relative deviation {max(r['rel'] for r in rows):.3e} "
          f"(float64 machine epsilon is 2.2e-16)")
    P.save({"scenario": a.scenario, "seed": a.seed, "D": a.D, "gamma": gamma,
            "dtype": "float64", "curve": rows,
            "max_rel": max(r["rel"] for r in rows)},
           f"E4_equivalence_{a.scenario}_s{a.seed}_D{a.D}.json")

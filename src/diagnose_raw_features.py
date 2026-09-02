"""How far the encoder-free variant can be pushed by increasing D.

Follow-up to variant C in diagnose_encoder.py. The input dimension changes from 10 to
140, so the capacity bottleneck is different and D can usefully go higher. Each figure is
a joint-batch ceiling.
"""
import time

import numpy as np

import config as C
import data as D
import metrics as M
import pipeline as P
from analytic import RFF, JointHead, median_gamma

REPLAY_F1 = {"blackhole": .64, "disflooding": .97, "localrepair": .88, "worstparent": .63}

if __name__ == "__main__":
    domains = D.load_all(C.DOMAIN_ORDERS["random"])
    names = [d[0] for d in domains]
    print(f"  {'Experience Replay (published)':<28} mean F1 {np.mean(list(REPLAY_F1.values())):.4f} | " +
          " ".join(f"{k[:2].upper()} {v:.3f}" for k, v in REPLAY_F1.items()), flush=True)

    Xtr = [d[1].astype(np.float64) for d in domains]
    Xte = [d[3].astype(np.float64) for d in domains]
    g = median_gamma(Xtr[0], seed=0)
    print(f"  raw-140 gamma={g:.4g}\n", flush=True)

    for Dd in [1024, 2048, 4096, 8192]:
        t0 = time.perf_counter()
        rff = RFF(140, Dd, g, seed=0)
        jh = JointHead(Dd, C.OUTPUT_SIZE, C.RIDGE)
        for f, d in zip(Xtr, domains):
            jh.accumulate(rff(f), P.onehot(d[2], C.OUTPUT_SIZE))
        W = jh.solve()
        f1 = np.array([M.perf(domains[i][4], rff(Xte[i]) @ W)[0] for i in range(len(domains))])
        fam = P.by_family(names, f1)
        state = (Dd * Dd * 8 + Dd * 2 * 8) / 1048576
        print(f"  D={Dd:5d}  mean F1 {f1.mean():.4f} | " +
              " ".join(f"{k[:2].upper()} {fam[k]:.3f}" for k in P.ATTACK_FAMILIES) +
              f"  state {state:.1f} MB  {time.perf_counter()-t0:.0f}s", flush=True)

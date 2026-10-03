"""Joint-batch diagnostic for the analytic head.

Solves the ridge problem over all 48 domains at once in the frozen feature space. The
recursion converges to exactly this estimator for a fixed representation. The resulting
classification score is a diagnostic rather than a general performance upper bound.

Reference figures for Experience Replay from the benchmark, for comparison:
    Blackhole 0.64, DIS-Flooding 0.97, Local Repair 0.88, Worst Parent 0.63
"""
import argparse

import numpy as np

import config as C
import data as D
import pipeline as P

REPLAY_F1 = {"blackhole": .64, "disflooding": .97, "localrepair": .88, "worstparent": .63}
REPLAY_AUC = {"blackhole": .69, "disflooding": 1.00, "localrepair": .95, "worstparent": .70}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random", choices=C.SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=C.RFF_D)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    print(f"scenario={a.scenario} seed={a.seed} D={a.D}")
    domains = D.load_all(C.DOMAIN_ORDERS[a.scenario])
    ntr = sum(len(d[1]) for d in domains)
    print(f"  {ntr} training windows, {sum(len(d[3]) for d in domains)} test windows")

    enc, t_enc = P.build_encoder(domains, a.seed, a.device, verbose=True)
    print(f"encoder trained in {t_enc:.1f}s")
    rff, gamma = P.make_rff(enc, domains, a.D, a.seed)
    print(f"RFF d_in={rff.W.shape[0]} D={a.D} gamma={gamma:.4g}")

    f1, auc, W, jh = P.run_joint(domains, enc, rff)
    names = [d[0] for d in domains]
    fam_f1, fam_auc = P.by_family(names, f1), P.by_family(names, auc)

    print(f"\n{'family':<14}{'F1 (joint)':>12}{'Replay':>9}{'AUC (joint)':>13}{'Replay':>9}")
    for fam in P.ATTACK_FAMILIES:
        print(f"{fam:<14}{fam_f1[fam]:>12.3f}{REPLAY_F1[fam]:>9.2f}"
              f"{fam_auc[fam]:>13.3f}{REPLAY_AUC[fam]:>9.2f}")
    print(f"\nmean  F1 {np.nanmean(f1):.4f}   AUC {np.nanmean(auc):.4f}")
    print(f"W {W.shape}, accumulator state {(jh.G.nbytes + jh.A.nbytes) / 2**20:.2f} MB")

    P.save({"scenario": a.scenario, "seed": a.seed, "D": a.D, "gamma": gamma,
            "n_train": ntr, "encoder_train_s": t_enc,
            "per_domain": {n: {"f1": float(x), "auc": float(y)}
                           for n, x, y in zip(names, f1, auc)},
            "by_family_f1": fam_f1, "by_family_auc": fam_auc,
            "mean_f1": float(np.nanmean(f1)), "mean_auc": float(np.nanmean(auc)),
            "replay_reference_f1": REPLAY_F1},
           f"E0_probe_{a.scenario}_s{a.seed}_D{a.D}.json")

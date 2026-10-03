"""Select D and the ridge coefficient on a held-out validation split.

Only the first domain of the random ordering is used. Three of its 16 training runs are
held out for validation, the encoder is trained on the remaining 13, and the grid is
scored on the validation split. The shared loader also returns test arrays, but neither
test features nor test labels enter training, scoring, or selection.

sweep_dimension.py characterises the state-performance trade-off across D. That is an
analysis, not a selection criterion; the operating point reported in the paper is the one
chosen here.
"""
import argparse
import itertools

import numpy as np

import config as C
import data as D
import metrics as M
import pipeline as P
from analytic import RFF, RLSHead, median_gamma
from encoder import LSTMClassifier, train_base

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--ridges", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dom = C.DOMAIN_ORDERS["random"][0]
    Xtr, ytr, _, _, Xva, yva = D.load_domain(dom, holdout_val=True)
    print(f"selection domain {dom}: train {Xtr.shape} val {Xva.shape}")

    grid = {}
    for seed in a.seeds:
        enc = LSTMClassifier()
        train_base(enc, Xtr, ytr, device=a.device, seed=seed)
        Ftr, Fva = enc.features(Xtr), enc.features(Xva)
        gamma = median_gamma(Ftr, seed=seed)
        Ytr = P.onehot(ytr, C.OUTPUT_SIZE)
        for Dd, rg in itertools.product(a.Ds, a.ridges):
            rff = RFF(Ftr.shape[1], Dd, gamma, seed=seed)
            h = RLSHead(Dd, C.OUTPUT_SIZE, rg, C.RLS_BLOCK)
            h.partial_fit(rff(Ftr), Ytr)
            f1, auc = M.perf(yva, h.predict_scores(rff(Fva)))
            grid.setdefault((Dd, rg), []).append((f1, auc))

    rows = [{"D": k[0], "ridge": k[1],
             "val_f1": float(np.mean([v[0] for v in vs])),
             "val_f1_std": float(np.std([v[0] for v in vs])),
             "val_auc": float(np.mean([v[1] for v in vs])),
             "state_kb": k[0] * k[0] * 8 / 1024} for k, vs in grid.items()]
    rows.sort(key=lambda r: -r["val_f1"])

    print(f"\n{'D':>6}{'ridge':>9}{'val F1':>10}{'+-std':>8}{'val AUC':>10}{'state KB':>10}")
    for r in rows[:12]:
        print(f"{r['D']:>6}{r['ridge']:>9.0e}{r['val_f1']:>10.4f}{r['val_f1_std']:>8.4f}"
              f"{r['val_auc']:>10.4f}{r['state_kb']:>10.1f}")

    # One-standard-error rule: among configurations within one standard deviation of the
    # best, take the simplest. Simpler means smaller D and larger ridge; note that ridge
    # is maximised, since a smaller ridge is the weaker regulariser.
    best = rows[0]
    thr = best["val_f1"] - best["val_f1_std"]
    frugal = min([r for r in rows if r["val_f1"] >= thr], key=lambda r: (r["D"], -r["ridge"]))
    print(f"\nbest       D={best['D']} ridge={best['ridge']:.0e} val F1={best['val_f1']:.4f}")
    print(f"1-SE pick  D={frugal['D']} ridge={frugal['ridge']:.0e} "
          f"val F1={frugal['val_f1']:.4f} state {frugal['state_kb']:.1f} KB")
    print("\nUpdate RFF_D and RIDGE in config.py, then run the remaining experiments.")
    P.save({"protocol": "validation split of the first domain; test arrays returned by the shared loader but not used",
            "domain": dom, "seeds": a.seeds, "grid": rows,
            "tiebreak": "min D, then max ridge",
            "best": best, "selected_1se": frugal}, "E_select_hyperparams.json")

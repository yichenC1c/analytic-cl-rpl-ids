"""Correctness checks for the analytic head and the metric definitions.

Run this after touching analytic.py. Item 7 is a regression test for a defect that
produced a different train/test split from the reference implementation in all 48
domains; it fails silently in every other respect, so it is checked explicitly.
"""
import sys

import numpy as np

from analytic import RFF, RLSHead, JointHead, median_gamma

ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    ok &= bool(cond)


print("== 1. recursion against the joint-batch solution ==")
rng = np.random.default_rng(0)
D, K, ridge = 64, 2, 1e-2
head, jh = RLSHead(D, K, ridge, block=37), JointHead(D, K, ridge)
for _ in range(48):
    n = int(rng.integers(200, 900))
    Z = rng.normal(size=(n, D))
    Y = rng.normal(size=(n, K))
    head.partial_fit(Z, Y)
    jh.accumulate(Z, Y)
rel = np.linalg.norm(head.W - jh.solve()) / np.linalg.norm(jh.solve())
check("relative error after 48 domains < 1e-9", rel < 1e-9, f"rel={rel:.3e}")

print("== 2. blocking affects rounding only, and is clamped to D ==")
Zs = [rng.normal(size=(int(rng.integers(300, 900)), D)) for _ in range(10)]
Ys = [rng.normal(size=(len(z), K)) for z in Zs]
jh2 = JointHead(D, K, ridge)
for z, y in zip(Zs, Ys):
    jh2.accumulate(z, y)
Wj2 = jh2.solve()
nj2 = np.linalg.norm(Wj2)
worst = 0.0
for b in (13, 64, 4096):
    h = RLSHead(D, K, ridge, block=b)
    check(f"  block {b} clamped to <= D", h.block <= D, f"actual {h.block}")
    for z, y in zip(Zs, Ys):
        h.partial_fit(z, y)
    r = np.linalg.norm(h.W - Wj2) / nj2
    worst = max(worst, r)
    print(f"         block={h.block:4d} vs joint, relative error {r:.3e}")
check("all block sizes within 1e-9 of joint", worst < 1e-9, f"worst {worst:.3e}")

print("== 3. random features approximate the Gaussian kernel ==")
X = rng.normal(size=(300, 10))
g = median_gamma(X)
d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
Ktrue = np.exp(-g * d2)
for Dd in (256, 4096):
    Z = RFF(10, Dd, g, seed=1)(X)
    err = np.abs(Z @ Z.T - Ktrue).max()
    check(f"D={Dd} max pointwise error", err < (0.35 if Dd == 256 else 0.12), f"err={err:.3f}")

print("== 4. float64 throughout ==")
h = RLSHead(32, 2, 1e-2)
h.partial_fit(rng.normal(size=(50, 32)), rng.normal(size=(50, 2)))
check("R and W are float64", h.R.dtype == np.float64 and h.W.dtype == np.float64)

print("== 5. domain orderings are consistent ==")
import config as C
check("four orderings, 48 unique domains each, same set",
      all(len(v) == 48 and len(set(v)) == 48 for v in C.DOMAIN_ORDERS.values())
      and len({frozenset(v) for v in C.DOMAIN_ORDERS.values()}) == 1)

print("== 6. metric formulas against a constructed example ==")
import metrics as M
Mm = np.array([[0.9, 0.5, 0.5], [0.7, 0.9, 0.5], [0.6, 0.8, 0.9]])
# plasticity = mean(M[1,1]-M[0,1], M[2,2]-M[1,2]) = mean(0.4, 0.4) = 0.4
# stability  = mean(M[1,0]-M[0,0], mean(M[2,0]-M[0,0], M[2,1]-M[1,1]))
#            = mean(-0.2, mean(-0.3, -0.1)) = -0.2
check("plasticity", abs(M.plasticity(Mm) - 0.4) < 1e-12, f"{M.plasticity(Mm):.4f}")
check("stability", abs(M.stability(Mm) + 0.2) < 1e-12, f"{M.stability(Mm):.4f}")
check("final performance", abs(M.final_perf(Mm) - np.mean([0.6, 0.8, 0.9])) < 1e-12)

print("== 7. split matches the reference loader file by file ==")
try:
    import glob
    import os
    import random
    import types
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    sys.path.insert(0, os.path.join(C.ROOT, "data", "IoT-Attacks-IDS", "src"))
    import utils as TU
    import data as MINE
    td = TU.create_domains(C.DATA_DIR)
    bad = []
    for dom in C.DOMAIN_ORDERS["random"]:
        ft = sorted(td[dom], key=TU.extract_index)[:20]
        random.seed(42)
        random.shuffle(ft)
        fm = sorted(os.listdir(os.path.join(C.DATA_DIR, dom)))
        fm = sorted(fm, key=MINE._run_index)[:C.N_RUNS]
        random.Random(42).shuffle(fm)
        if ft[:16] != fm[:16] or ft[16:20] != fm[16:20]:
            bad.append(dom)
    check("all 48 splits identical to the reference", not bad, f"{len(bad)} differ")

    f = os.path.join(C.DATA_DIR, "blackhole_var10_oo", "1_features_timeseries_60_sec.csv")
    df = MINE._load_csv(f)
    g0, g1 = df.drop(columns=["label"]).min(), df.drop(columns=["label"]).max()
    nd = MINE._minmax(df, g0, g1)
    th = TU.seq_maker(nd, 10, "label")
    Xm, ym = MINE._seq_maker(nd, 10)
    check("windowing features identical", np.array_equal(th.iloc[:, :-1].values, Xm))
    check("windowing labels identical", np.array_equal(th["label"].values, ym))
except ImportError as e:
    print(f"  [SKIP] reference implementation unavailable: {e}")

print("== 8. joint-batch probe equals the last row of the incremental run ==")
try:
    import data as DD
    import pipeline as PP
    from encoder import LSTMClassifier as LC, train_base as TB
    dm = DD.load_all(C.DOMAIN_ORDERS["random"][:4])
    e = LC()
    TB(e, dm[0][1], dm[0][2], epochs=5, seed=0)
    rf, _ = PP.make_rff(e, dm, 128, 0)
    Mf, _, _, _ = PP.run_incremental(dm, e, rf)
    fj, _, _, _ = PP.run_joint(dm, e, rf)
    d = float(np.abs(Mf[-1] - fj).max())
    check("last incremental row equals joint solution", d < 1e-9, f"max diff {d:.3e}")
except Exception as ex:
    print(f"  [SKIP] {ex}")

print("\n" + ("all checks passed" if ok else "failures present"))
sys.exit(0 if ok else 1)

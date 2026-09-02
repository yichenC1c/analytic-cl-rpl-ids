"""Shared training and evaluation loop used by every experiment script."""
import json
import os
import time

import numpy as np

import config as C
import metrics as M
from analytic import RFF, RLSHead, JointHead, median_gamma
from encoder import LSTMClassifier, train_base

ATTACK_FAMILIES = ["blackhole", "disflooding", "localrepair", "worstparent"]


def onehot(y, k=2):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def build_encoder(domains, seed, device="cpu", verbose=False):
    """Train the encoder on the first domain of the ordering, then freeze it."""
    _, Xtr, ytr, _, _ = domains[0]
    enc = LSTMClassifier()
    t0 = time.perf_counter()
    train_base(enc, Xtr, ytr, device=device, seed=seed, verbose=verbose)
    return enc, time.perf_counter() - t0


def make_rff(enc, domains, D_dim, seed):
    """gamma is estimated on first-domain training features only; later domains are not
    consulted."""
    F0 = enc.features(domains[0][1])
    g = C.RFF_GAMMA if C.RFF_GAMMA else median_gamma(F0, seed=seed)
    return RFF(F0.shape[1], D_dim, g, seed=seed), g


def run_incremental(domains, enc, rff, ridge=C.RIDGE, block=C.RLS_BLOCK):
    """Adapt domain by domain, evaluating on all 48 test splits after each one.

    Returns (M_f1, M_auc, times, head), where M[t, i] is performance on domain i after
    training through domain t.
    """
    T = len(domains)
    Zte = [rff(enc.features(d[3])) for d in domains]
    yte = [d[4] for d in domains]
    head = RLSHead(rff.D, C.OUTPUT_SIZE, ridge, block)
    Mf1 = np.full((T, T), np.nan)
    Mauc = np.full((T, T), np.nan)
    times = []
    for t, (_, Xtr, ytr, _, _) in enumerate(domains):
        Z = rff(enc.features(Xtr))
        t0 = time.perf_counter()
        head.partial_fit(Z, onehot(ytr, C.OUTPUT_SIZE))
        times.append(time.perf_counter() - t0)
        for i in range(T):
            Mf1[t, i], Mauc[t, i] = M.perf(yte[i], head.predict_scores(Zte[i]))
    return Mf1, Mauc, times, head


def run_joint(domains, enc, rff, ridge=C.RIDGE):
    """Solve once over all domains jointly. This is the ceiling the recursion converges to."""
    jh = JointHead(rff.D, C.OUTPUT_SIZE, ridge)
    for _, Xtr, ytr, _, _ in domains:
        jh.accumulate(rff(enc.features(Xtr)), onehot(ytr, C.OUTPUT_SIZE))
    W = jh.solve()
    f1s, aucs = [], []
    for _, _, _, Xte, yte in domains:
        a, b = M.perf(yte, rff(enc.features(Xte)) @ W)
        f1s.append(a)
        aucs.append(b)
    return np.array(f1s), np.array(aucs), W, jh


def save(obj, fname):
    os.makedirs(C.RESULTS, exist_ok=True)
    p = os.path.join(C.RESULTS, fname)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"-> {p}")
    return p


def by_family(names, values):
    """Aggregate per-domain values by attack family."""
    out = {}
    for fam in ATTACK_FAMILIES:
        idx = [i for i, n in enumerate(names) if n.startswith(fam)]
        out[fam] = float(np.nanmean(np.asarray(values)[idx]))
    return out

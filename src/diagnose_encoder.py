"""Locate the bottleneck behind the detection-performance gap.

Four variants are evaluated with joint ridge fitting so that the update rule is held
constant. Only the feature space differs:

    A  encoder trained on the first domain (the deployable configuration)
    B  encoder trained on all 48 domains, then frozen (an oracle; not deployable)
    C  no encoder, random features on the raw 140-dimensional window
    D  wider encoder trained on the first domain

Variant B is an oracle diagnostic because its encoder uses all domains during training.
"""
import numpy as np

import config as C
import data as D
import metrics as M
import pipeline as P
from analytic import RFF, JointHead, median_gamma
from encoder import LSTMClassifier, train_base

REPLAY_F1 = {"blackhole": .64, "disflooding": .97, "localrepair": .88, "worstparent": .63}

domains = D.load_all(C.DOMAIN_ORDERS["random"])
names = [d[0] for d in domains]


def joint_eval(Ftr_list, Fte_list, Dd, seed=0, tag=""):
    g = median_gamma(Ftr_list[0], seed=seed)
    rff = RFF(Ftr_list[0].shape[1], Dd, g, seed=seed)
    jh = JointHead(Dd, C.OUTPUT_SIZE, C.RIDGE)
    for f, d in zip(Ftr_list, domains):
        jh.accumulate(rff(f), P.onehot(d[2], C.OUTPUT_SIZE))
    W = jh.solve()
    f1 = np.array([M.perf(domains[i][4], rff(Fte_list[i]) @ W)[0] for i in range(len(domains))])
    fam = P.by_family(names, f1)
    print(f"  {tag:<38} mean F1 {f1.mean():.4f} | " +
          " ".join(f"{k[:2].upper()} {fam[k]:.3f}" for k in P.ATTACK_FAMILIES))
    return f1.mean(), fam


if __name__ == "__main__":
    print(f"  {'Experience Replay (published)':<38} mean F1 {np.mean(list(REPLAY_F1.values())):.4f} | " +
          " ".join(f"{k[:2].upper()} {v:.3f}" for k, v in REPLAY_F1.items()))
    print()

    enc = LSTMClassifier()
    train_base(enc, domains[0][1], domains[0][2], seed=0)
    joint_eval([enc.features(d[1]) for d in domains],
               [enc.features(d[3]) for d in domains], 64,
               tag="A  first-domain encoder, D=64")

    Xall = np.concatenate([d[1] for d in domains])
    yall = np.concatenate([d[2] for d in domains])
    enc_oracle = LSTMClassifier()
    train_base(enc_oracle, Xall, yall, seed=0)
    joint_eval([enc_oracle.features(d[1]) for d in domains],
               [enc_oracle.features(d[3]) for d in domains], 64,
               tag="B  all-domain encoder (oracle), D=64")

    raw_tr = [d[1].astype(np.float64) for d in domains]
    raw_te = [d[3].astype(np.float64) for d in domains]
    for Dd in (256, 1024):
        joint_eval(raw_tr, raw_te, Dd, tag=f"C  no encoder, raw 140-d, D={Dd}")

    enc_wide = LSTMClassifier(hidden_dim=64, fc_hidden_dim=64)
    train_base(enc_wide, domains[0][1], domains[0][2], seed=0)
    joint_eval([enc_wide.features(d[1]) for d in domains],
               [enc_wide.features(d[3]) for d in domains], 256,
               tag="D  wide encoder (hidden=64), D=256")

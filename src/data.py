"""Dataset loading, reproducing the reference loader protocol file by file.

Two behaviours are deliberately preserved even though they are unusual. Changing either
would break comparability with the published baseline figures.
"""
import glob
import os
import random
import re

import numpy as np
import pandas as pd

import config as C

_DROP = ["Unnamed: 0"]


def _run_index(name):
    """Mirrors extract_index in the reference loader.

    The regular expression never matches this dataset's filenames, because the character
    before '_60_sec' is 'timeseries' rather than a digit. It therefore returns a constant
    and sorted() degenerates into a stable no-op, leaving the input order intact. What
    actually determines the order is the lexicographic sorted(os.listdir(...)) upstream:
    10, 11, ..., 19, 1, 20, (21,) 2, 3, ..., 9.

    This must be reproduced exactly. Sorting numerically instead yields a different
    train/test split in all 48 domains and silently invalidates every quoted baseline.
    """
    m = re.search(r"_(\d+)_60_sec\.csv$", os.path.basename(name))
    return int(m.group(1)) if m else 10 ** 9


def _load_csv(path):
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    for c in _DROP:
        if c in df.columns:
            df = df.drop(columns=[c])
    assert "label" in df.columns, f"'label' missing in {path}"
    return df


def _minmax(df, gmin, gmax, label_col="label"):
    cols = [c for c in df.columns if c != label_col]
    denom = (gmax - gmin).replace(0, 1)
    out = df.copy()
    out[cols] = (out[cols] - gmin) / denom
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _seq_maker(df, seq_len, label_col="label"):
    """Windowing, reproducing the reference implementation including its labelling rule.

    Labels are not taken from the final frame of each window. Everything before the first
    attack index is labelled 0 and everything after it is labelled 1. The raw labels in
    this dataset switch exactly once per run, so the rule is consistent with the data,
    but it is not the conventional choice and is kept only for comparability.
    """
    feat = df.drop(columns=[label_col])
    labels = df[label_col].astype(int).values
    idx = np.where(labels == 1)[0]
    start_attack = len(labels) + seq_len if len(idx) == 0 else max(0, idx[0] - seq_len)

    n = len(feat) - seq_len
    if n <= 0:
        return np.empty((0, feat.shape[1] * seq_len)), np.empty((0,), dtype=int)
    V = feat.values
    seqs = np.lib.stride_tricks.sliding_window_view(V, seq_len, axis=0)[:n]
    seqs = seqs.transpose(0, 2, 1).reshape(n, -1)   # time-major, matching .flatten()
    n0 = min(start_attack, n)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n - n0, dtype=int)])
    return seqs, y


def load_domain(domain, data_dir=C.DATA_DIR, holdout_val=False):
    """Return (X_train, y_train, X_test, y_test); X has shape (N, 140), float32.

    With holdout_val=True a sixth and seventh element carve a validation split out of the
    16 training runs. That path is used only for hyperparameter selection; the main
    experiments always use the full 16 so the protocol matches the benchmark exactly.
    """
    files = sorted(os.listdir(os.path.join(data_dir, domain)))
    files = sorted(files, key=_run_index)[:C.N_RUNS]
    random.Random(C.SPLIT_SEED).shuffle(files)

    tr_names = files[:C.N_TRAIN_FILES]
    va_names = []
    if holdout_val:
        tr_names, va_names = tr_names[:-C.VAL_FILES], tr_names[-C.VAL_FILES:]
    tr_f = [os.path.join(data_dir, domain, f) for f in tr_names]
    va_f = [os.path.join(data_dir, domain, f) for f in va_names]
    te_f = [os.path.join(data_dir, domain, f) for f in files[C.N_TRAIN_FILES:C.N_RUNS]]

    tr_dfs = [_load_csv(p) for p in tr_f]
    te_dfs = [_load_csv(p) for p in te_f]
    va_dfs = [_load_csv(p) for p in va_f]

    cols = [c for c in tr_dfs[0].columns if c != "label"]
    gmin = pd.concat([d[cols].min(axis=0) for d in tr_dfs], axis=1).min(axis=1)
    gmax = pd.concat([d[cols].max(axis=0) for d in tr_dfs], axis=1).max(axis=1)

    def build(dfs):
        Xs, ys = [], []
        for d in dfs:
            x, y = _seq_maker(_minmax(d, gmin, gmax), C.WINDOW_SIZE)
            if len(x):
                Xs.append(x)
                ys.append(y)
        X = np.nan_to_num(np.concatenate(Xs).astype(np.float32))
        return X, np.concatenate(ys)

    Xtr, ytr = build(tr_dfs)
    Xte, yte = build(te_dfs)
    assert Xtr.shape[1] == C.INPUT_SIZE, f"feature dim {Xtr.shape[1]} != {C.INPUT_SIZE}"
    if holdout_val:
        Xva, yva = build(va_dfs)
        return Xtr, ytr, Xte, yte, Xva, yva
    return Xtr, ytr, Xte, yte


def load_all(order, cache={}):
    """Load the 48 domains in the given order as a list of (name, Xtr, ytr, Xte, yte)."""
    out = []
    for d in order:
        if d not in cache:
            cache[d] = load_domain(d)
        out.append((d, *cache[d]))
    return out

"""Continual-learning metrics, following the definitions used by the benchmark.

Training efficiency is deliberately absent. It is defined as a geometric mean of
per-domain training times normalised by the fastest method in the comparison pool, so its
value depends on which methods happen to be included and it is not comparable across
papers. Absolute adaptation time is reported instead.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def perf(y_true, scores):
    """Return (weighted F1, AUC). scores is the (n, 2) least-squares output."""
    pred = scores.argmax(1)
    f1 = f1_score(y_true, pred, average="weighted", zero_division=0)
    s = scores[:, 1] - scores[:, 0]
    auc = roc_auc_score(y_true, s) if len(np.unique(y_true)) > 1 else float("nan")
    return f1, auc


def final_perf(M):
    """Mean performance over all domains after the last one has been seen."""
    return np.nanmean(M[-1, :])


def plasticity(M):
    """Mean gain on each domain from training on it."""
    T = M.shape[0]
    return np.nanmean([M[j, j] - M[j - 1, j] for j in range(1, T)])


def stability(M):
    """Mean backward transfer. Negative values indicate forgetting."""
    T = M.shape[0]
    return np.nanmean([np.nanmean([M[t, i] - M[i, i] for i in range(t)]) for t in range(1, T)])


def training_efficiency(times, times_min):
    """Geometric mean of TT_min(t)/TT_a(t). Pool-relative; see the module docstring."""
    r = np.asarray(times_min, float) / np.maximum(np.asarray(times, float), 1e-12)
    return float(np.exp(np.mean(np.log(np.clip(r, 1e-12, None)))))


def summarize(M, times=None, times_min=None):
    out = {
        "perf_final": float(final_perf(M)),
        "plasticity": float(plasticity(M)),
        "stability": float(stability(M)),
    }
    if times is not None and times_min is not None:
        out["train_efficiency"] = training_efficiency(times, times_min)
    if times is not None:
        out["train_time_total_s"] = float(np.sum(times))
        out["train_time_per_domain_s"] = float(np.mean(times))
    return out

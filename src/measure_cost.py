"""Deployment cost.

The primary figures are architectural counts, which are hardware independent. Wall-clock
latency is secondary and depends on the measurement environment, which is recorded
alongside it. Run this single-process; concurrent load makes the timings meaningless.
"""
import argparse
import platform
import time

import numpy as np

import config as C
import data as D
import pipeline as P
from analytic import RFF, RLSHead, median_gamma
from encoder import n_params

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=20)
    a = ap.parse_args()

    domains = D.load_all(C.DOMAIN_ORDERS["random"][:2])
    enc, _ = P.build_encoder(domains, a.seed)
    F0 = enc.features(domains[0][1])
    d_in = F0.shape[1]
    gamma = median_gamma(F0, seed=a.seed)
    enc_bytes = n_params(enc) * 4

    rows = []
    for Dd in a.Ds:
        rff = RFF(d_in, Dd, gamma, seed=a.seed)
        head = RLSHead(Dd, C.OUTPUT_SIZE, C.RIDGE, C.RLS_BLOCK)
        head.partial_fit(rff(F0), P.onehot(domains[0][2], C.OUTPUT_SIZE))
        x1 = F0[:1]
        ts = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            head.predict_scores(rff(x1))
            ts.append(time.perf_counter() - t0)
        rows.append({
            "D": Dd,
            "state_bytes_updatable_R_W": head.state_bytes,
            "state_bytes_inference_only_W": head.inference_bytes,
            "rff_bytes_W_b": rff.n_bytes,
            "encoder_bytes_fp32": enc_bytes,
            "update_macs": 2 * Dd * Dd + Dd * d_in,
            "inference_macs": Dd * d_in + Dd * C.OUTPUT_SIZE,
            "latency_ms_median": float(np.median(ts) * 1e3),
        })
        print(f"  D={Dd:5d}  inference {(head.inference_bytes + rff.n_bytes)/2**10:7.1f} KB | "
              f"updatable {(head.state_bytes + rff.n_bytes)/2**10:8.1f} KB  "
              f"update {rows[-1]['update_macs']/1e6:6.3f} MMAC  "
              f"latency {rows[-1]['latency_ms_median']:.4f} ms")
    print(f"\nencoder itself is {enc_bytes/2**10:.1f} KB ({n_params(enc)} parameters)")
    P.save({"env": {"platform": platform.platform(), "machine": platform.machine(),
                    "note": "single process; not edge hardware, secondary figure only"},
            "d_in": d_in, "encoder_params": n_params(enc), "rows": rows},
           "E6_cost.json")

"""Main experiment: the proposed detector over 4 orderings x 3 seeds."""
import argparse

import config as C
import data as D
import metrics as M
import pipeline as P

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random", choices=C.SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=C.RFF_D)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    domains = D.load_all(C.DOMAIN_ORDERS[a.scenario])
    enc, t_enc = P.build_encoder(domains, a.seed, a.device)
    rff, gamma = P.make_rff(enc, domains, a.D, a.seed)
    Mf1, Mauc, times, head = P.run_incremental(domains, enc, rff)

    names = [d[0] for d in domains]
    # Training efficiency is omitted here: it is relative to a method pool, so comparing
    # a method against itself is vacuous. Raw adaptation time is recorded instead.
    res = {"scenario": a.scenario, "seed": a.seed, "D": a.D, "gamma": gamma,
           "encoder_train_s": t_enc,
           "f1": M.summarize(Mf1, times),
           "auc": M.summarize(Mauc, times),
           "by_family_f1_final": P.by_family(names, Mf1[-1]),
           "by_family_auc_final": P.by_family(names, Mauc[-1]),
           "state_bytes_updatable": head.state_bytes,
           "state_bytes_inference_only": head.inference_bytes,
           "rff_bytes": rff.n_bytes,
           "matrix_f1": Mf1, "matrix_auc": Mauc, "times_s": times}
    for k in ("f1", "auc"):
        s = res[k]
        print(f"{k.upper():4s}  perf {s['perf_final']:.4f}  plasticity {s['plasticity']:+.4f}  "
              f"stability {s['stability']:+.4f}  adapt {s['train_time_total_s']:.2f}s")
    print(f"state: inference {(head.inference_bytes + rff.n_bytes) / 2**10:.1f} KB | "
          f"updatable {(head.state_bytes + rff.n_bytes) / 2**10:.1f} KB")
    P.save(res, f"E1_ours_{a.scenario}_s{a.seed}_D{a.D}.json")

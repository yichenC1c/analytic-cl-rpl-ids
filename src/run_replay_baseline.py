"""Reproduce the Experience Replay baseline using the benchmark authors' own code.

The point of this run is to establish that our harness reproduces the published setting
rather than merely resembling it, so the reference implementation is invoked unmodified.
Reimplementing it would introduce degrees of freedom and defeat the purpose.

Note that the reference entry point does not take a domain ordering; it reads the order
from the insertion order of the loader dictionary. The dictionary is therefore built in
scenario order and restricted to the 48 domains used by the benchmark, since the
repository also ships a fifth attack family that the published results exclude.
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch


class _WandbStub:
    """The reference code touches config.update, watch, log and summary only."""

    def __init__(self):
        self.summary = {}
        self.config = self

    def update(self, *a, **k):
        pass

    def watch(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import config as C
    reference = os.path.join(C.ROOT, "data", "IoT-Attacks-IDS", "src")
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    sys.path.insert(0, reference)
    import utils as TU
    import models as TM
    import tdim_replay as TR

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    class Args:
        architecture = "LSTM"
        algorithm = "replay"
        scenario = a.scenario
        exp_no = a.seed
        window_size = C.WINDOW_SIZE
        step_size = 3
        batch_size = C.BATCH_SIZE
        input_size = C.INPUT_SIZE
        hidden_size = C.HIDDEN_SIZE
        output_size = C.OUTPUT_SIZE
        num_layers = C.NUM_LAYERS
        dropout = 0.3
        bidirectional = False
        patience = C.PATIENCE
        learning_rate = C.LEARNING_RATE
        epochs = a.epochs
        weight_decay = 0.0
        forgetting_threshold = 0.01

    args = Args()
    all_domains = TU.create_domains(C.DATA_DIR)
    tr_loader, te_loader = {}, {}
    for k in C.DOMAIN_ORDERS[a.scenario]:
        tr_loader[k], te_loader[k] = TU.load_data(
            C.DATA_DIR, k, all_domains[k], window_size=args.window_size,
            step_size=args.step_size, batch_size=args.batch_size)
    print(f"scenario={a.scenario} seed={a.seed} domains={len(tr_loader)}")

    dev = torch.device(a.device)   # the reference _sync() expects a device object
    model = TM.LSTMClassifier(input_dim=args.input_size, hidden_dim=args.hidden_size,
                              output_dim=args.output_size, num_layers=args.num_layers,
                              fc_hidden_dim=10).to(dev)
    # tdim_replay has no return statement; it writes its results to disk.
    TR.tdim_replay(args, _WandbStub(), tr_loader, te_loader, dev,
                   model, args.exp_no, num_epochs=args.epochs,
                   learning_rate=args.learning_rate, patience=args.patience)
    saved = os.path.join(
        "results",
        f"{args.exp_no}_experiment_results_{args.architecture}_{args.algorithm}"
        f"_REPLAY_{args.scenario}.json")
    res = json.load(open(saved))

    bwt = res["BWT_values_f1"]
    pl = res["plasticity_values_f1"]
    out = {"scenario": a.scenario, "seed": a.seed,
           "their_S_bar_f1": float(np.mean(bwt)) if len(bwt) else None,
           "their_P_bar_f1": float(pl[-1]) if len(pl) else None,
           "their_S_bar_auc": float(np.mean(res["BWT_values_auc"])) if res["BWT_values_auc"] else None,
           "domain_training_cost": res["domain_training_cost"],
           "replay_config": res["replay_config"]}
    print(f"  S_bar(F1) {out['their_S_bar_f1']}   P_bar(F1) {out['their_P_bar_f1']}")
    print("  published reference: S_bar ~ -0.11, P_bar ~ 0.31")
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(os.path.join(C.RESULTS, f"E2_replay_{a.scenario}_s{a.seed}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"-> E2_replay_{a.scenario}_s{a.seed}.json")

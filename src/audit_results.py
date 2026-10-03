"""Recompute the manuscript's principal aggregates from the saved JSON logs."""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCENARIOS = ("random", "b2w", "w2b", "toggle")


def load(name):
    return json.loads((RESULTS / name).read_text())


def metrics(matrix):
    m = np.asarray(matrix, dtype=float)
    t = len(m)
    diag = np.diag(m)
    return {
        "final_f1": float(np.mean(m[-1])),
        "plasticity": float(np.mean([m[j, j] - m[j - 1, j] for j in range(1, t)])),
        "bwt": float(np.mean([np.mean(m[j, :j] - diag[:j]) for j in range(1, t)])),
        "frozen_f1": float(np.mean(m[0])),
    }


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main():
    by_order = {}
    all_runs = []
    family = {k: [] for k in ("blackhole", "disflooding", "localrepair", "worstparent")}
    for scenario in SCENARIOS:
        runs = []
        for seed in range(3):
            record = load(f"E1_ours_{scenario}_s{seed}_D64.json")
            item = metrics(record["matrix_f1"])
            item["head_time_s"] = float(sum(record["times_s"]))
            runs.append(item)
            all_runs.append(item)
            for key, value in record["by_family_f1_final"].items():
                family[key].append(value)
        by_order[scenario] = {key: mean_sd([r[key] for r in runs]) for key in runs[0]}

    overall = {key: float(np.mean([r[key] for r in all_runs])) for key in all_runs[0]}
    family_mean = {key: float(np.mean(values)) for key, values in family.items()}

    equivalence = [
        point["rel"]
        for scenario in SCENARIOS
        for point in load(f"E4_equivalence_{scenario}_s0_D64.json")["curve"]
    ]
    kernel = [
        row
        for scenario in SCENARIOS
        for row in load(f"E3_kelm_{scenario}_s0.json")["trace"]
        if row["status"] == "ok"
    ]

    assert round(overall["final_f1"], 3) == 0.603
    assert round(overall["frozen_f1"], 3) == 0.494
    assert round(overall["plasticity"], 3) == 0.055
    assert round(overall["bwt"], 3) == -0.030
    assert round(overall["head_time_s"], 3) == 1.027
    assert [round(family_mean[k], 3) for k in family] == [0.378, 0.956, 0.679, 0.399]
    assert len(equivalence) == 192
    assert f"{min(equivalence):.1e}" == "5.8e-13"
    assert f"{max(equivalence):.1e}" == "7.7e-11"
    assert round(max(row["kernel_gb"] for row in kernel), 2) == 8.61
    assert round(max(row["refit_s"] for row in kernel)) == 456

    print("D=64 results by ordering, mean ± sample SD")
    for scenario, values in by_order.items():
        print(
            f"{scenario:7s}  frozen {values['frozen_f1'][0]:.3f} ± {values['frozen_f1'][1]:.3f}"
            f"  final {values['final_f1'][0]:.3f} ± {values['final_f1'][1]:.3f}"
            f"  P {values['plasticity'][0]:.3f} ± {values['plasticity'][1]:.3f}"
            f"  BWT {values['bwt'][0]:.3f} ± {values['bwt'][1]:.3f}"
        )
    print("\nOverall")
    for key, value in overall.items():
        print(f"  {key}: {value:.9f}")
    print("\nAttack-family F1")
    for key, value in family_mean.items():
        print(f"  {key}: {value:.9f}")
    print(f"\nEquivalence: {len(equivalence)} points, {min(equivalence):.3e} to {max(equivalence):.3e}")
    print("All stored-result checks passed.")


if __name__ == "__main__":
    main()

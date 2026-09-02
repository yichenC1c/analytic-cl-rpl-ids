# Forgetting-Free Continual Intrusion Detection for RPL Networks

A closed-form decision head for domain-incremental intrusion detection. A temporal
encoder is trained once and frozen; from then on only a random-feature ridge head is
updated, by recursive least squares. The update is provably identical to solving over all
data jointly, so forgetting is eliminated structurally rather than regularised, and the
resident state is 33 KB regardless of how many domains have been seen.

Code and experiments for a paper under submission.

## What is here

```
src/          implementation and experiment drivers
docs/         environment notes and implementation caveats
```

Running the scripts creates `data/` (the benchmark, fetched below), `results/` (one JSON
per run) and `figures/`. None of the three is tracked: the dataset belongs to its authors,
and results and figures are reproduced by running the code rather than shipped with it.

## Setup

The system Python is not usable here; PyTorch has no wheel for it. Use 3.11 or 3.12.

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Fetch the benchmark dataset (about 165 MB, includes the reference implementation used by
the Replay baseline):

```bash
git clone --depth 1 https://github.com/uu-core/IoT-Attacks-IDS data/IoT-Attacks-IDS
```

## Running

All scripts are run from `src/`.

```bash
cd src
../venv/bin/python test_analytic.py          # correctness checks, seconds
```

Then, in order:

| Script | Purpose | Cost |
|---|---|---|
| `select_hyperparams.py` | choose `D` and ridge on a validation split | ~5 min |
| `probe_upper_bound.py` | joint-batch ceiling; decides whether to continue | ~2 min |
| `check_equivalence.py` | recursion against the joint solution | ~2 min |
| `run_analytic.py` | main results, 4 orderings x 3 seeds | ~1 h total |
| `run_replay_baseline.py` | reproduce the Replay baseline for validation | ~3 h total |
| `run_exact_kelm.py` | exact-kernel baseline and its memory wall | ~1 h |
| `sweep_dimension.py` | state against performance across `D` | ~2 h |
| `measure_cost.py` | byte and MAC counts; run single-process | ~1 h |
| `make_figures.py` | build the figures from `results/` | seconds |

`run_analytic.py`, `check_equivalence.py`, `run_exact_kelm.py` and
`run_replay_baseline.py` take `--scenario {random,b2w,w2b,toggle}` and `--seed`.

`select_hyperparams.py` writes its choice to stdout; copy it into `RFF_D` and `RIDGE` in
`config.py` before running anything else.

Two diagnostics support the ceiling analysis in the paper and are not part of the main
sequence: `diagnose_encoder.py` compares four feature spaces at their joint-batch
ceilings, and `diagnose_raw_features.py` pushes the encoder-free variant to larger `D`.

## Two constraints the code depends on

**The analytic head runs in float64 on the CPU.** The MPS backend does not support
float64, and the equivalence check is meaningless at single precision.

**The RLS block size must not exceed `D`.** Woodbury inverts a `block x block` matrix; once
that exceeds `D` it is the identity plus a low-rank correction, and inverting it is both
slower and ill-conditioned. `RLSHead` clamps the value. Measured relative error against
the joint solution at `D=256`: 2.6e-12 at block 16 against 2.4e-07 at block 512.

`docs/NOTES.md` records the dataset and reference-implementation details that the
experiments depend on. Read it before changing anything in `data.py`.

## Attribution

Experiments run on the public RPL attack benchmark of Banerjee et al. (IEEE ICC 2026),
<https://github.com/uu-core/IoT-Attacks-IDS>, under its published protocol. The Experience
Replay baseline is that repository's own implementation, invoked unmodified, so the
reproduction validates the harness rather than a reimplementation of it. The dataset and
that code remain the property of their authors and are not redistributed here.

## Licence

MIT, see `LICENSE`. The vendored benchmark is covered by its own licence.

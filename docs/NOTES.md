# Implementation notes

Facts about the dataset and the reference implementation that the experiments depend on.
Read this before changing `data.py` or `config.py`.

## The encoder is small, and `D` cannot be large

`main.py` in the reference implementation passes `fc_hidden_dim=10` explicitly,
overriding the default of 64. Together with `hidden_size=10` that gives:

| | |
|---|---|
| encoder parameters | 6,212 (24.3 KB in float32) |
| frozen feature dimension | 10 |

The persistent random map and updatable head use `8D(D+d+3)` bytes, where `d=10`:

| D | state | relative to the model |
|---|---|---|
| 128 | 141 KiB | 5.8x |
| 256 | 538 KiB | 22x |
| 1024 | 8.10 MiB | 342x |

An exemplar buffer of 4,000 windows holds 2.14 MiB of retained traffic before labels and
model state. At `D=64`, the random map and updatable head occupy 38.5 KiB and retain no
traffic windows. These quantities have different roles and are reported separately.

## The exact-kernel wall is in training, not in inference state

Because the frozen features are only 10-dimensional, storing support vectors is cheap
(80 bytes each, 3.4 MB at N=35k). The argument that the exact kernel does not scale rests
on two other quantities:

1. **Training memory.** The Gram matrix is `N x N` in float64. The experiment uses a
   configured cutoff of 35,000 cumulative samples before attempting the next allocation.
   The largest completed fit formed an 8.61 GiB Gram matrix at N=33,996. The baseline
   completes two to four of the 48 domains depending on the ordering.
2. **Inference cost.** Each prediction requires N kernel evaluations. After 48 domains
   N is about 600k, against D cosines and one `D x 2` product for the analytic head.

## The RLS block size must not exceed D

Woodbury inverts `(I + Z R Z^T)`, which is `block x block`. Once `block > D` that matrix is
the identity plus a correction of rank at most D, and a general inverse of it is both
slower and ill-conditioned. Measured relative error against the joint solution at `D=256`:

| block | 16 | 64 | 256 | 512 | 4096 |
|---|---|---|---|---|---|
| relative error | 2.6e-12 | 7.3e-11 | 7.6e-09 | 2.4e-07 | 3.6e-07 |

`RLSHead` clamps the value to `min(block, D)` and defaults to 16. The equivalence figure
needs a small block; at block 512 the curve sits at 1e-07 rather than 1e-11.

## Where the code and the published text disagree

| | paper text | code | we follow |
|---|---|---|---|
| batch size | 256 | 128 | 128, with a footnote |
| epochs | 100 | default 3 | 100, but early stopping fires at 25-40 |

## Two reference behaviours that must not be "fixed"

**Window labelling.** `seq_maker` does not take the label of the final frame. Everything
before the first attack index is labelled 0 and everything after it is labelled 1. The raw
labels switch exactly once per run in this dataset, so the rule is consistent with the
data, but it is unconventional and is kept only for comparability.

**File ordering.** `extract_index` uses a regular expression that never matches these
filenames, because the character before `_60_sec` is `timeseries` rather than a digit. It
therefore returns a constant and `sorted()` degenerates into a stable no-op. What actually
orders the files is the lexicographic `sorted(os.listdir(...))` upstream: 10, 11, ..., 19,
1, 20, (21,) 2, 3, ..., 9.

This must be reproduced exactly. Sorting numerically instead produces a different
train/test split in all 48 domains, which silently invalidates every quoted baseline
figure without raising an error or degrading any metric. `test_analytic.py` item 7 is a
regression test for it.

## Dataset

Data ships inside the benchmark repository under `src/attack_data/`; no simulator run is
required.

| | |
|---|---|
| domains used | 48 (a fifth attack family, `versionnumber`, is present but unused by the benchmark) |
| detection windows | 749,263 |
| per domain | 7,874 to 18,228, mean 15,828 |
| columns | `rank, disr, diss, dior, dios, diar, tots` x (mean, std), plus `label` |
| class balance | 49.6% attack; the classes are balanced, so no imbalance handling is warranted |

Two irregularities in the released data. `disflooding_var15_base` holds only 19 runs, so
its test split has 3 rather than 4. `disflooding_var10_oo` holds 21, and the lexicographic
ordering above discards run 9 rather than run 21.

## Sequence length

The loader flattens a 10-frame window into a 140-dimensional vector and feeds it as a
length-1 sequence, so the LSTM never sees a time axis. This is the reference design and is
kept unchanged, but the model should not be described as capturing temporal dependencies.

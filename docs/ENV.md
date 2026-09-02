# Environment

## Setup

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The system Python on the development machine is 3.14, for which PyTorch has no wheel.
Use 3.11 or 3.12.

## Verified configuration

| | |
|---|---|
| Python | 3.11.12 |
| torch | 2.13.0, MPS available |
| numpy | 2.4.6, linked against Accelerate |
| scipy / scikit-learn / pandas / matplotlib | 1.17.1 / 1.9.0 / 3.0.5 / 3.11.1 |
| hardware | Apple M3 Pro, 11 cores (5P + 6E), 18 GB |
| measured float64 throughput | about 70 GFLOPS (8000x8000 solve in 2.3 s) |

## Precision

The MPS backend does not support float64; `torch.randn(..., dtype=float64, device='mps')`
raises. The equivalence check accumulates 48 successive Woodbury updates, and at single
precision the rounding error swamps the quantity being measured. Split the precision:

- encoder: float32, CPU or MPS
- analytic head: float64, numpy, CPU

Do not expect MPS to accelerate the encoder. At roughly 6k parameters with a length-1
sequence, kernel launch overhead dominates and the CPU is usually faster.

## Memory ceiling

The exact-kernel baseline determines the memory requirement. Its Gram matrix is `N x N` in
float64:

| cumulative N | Gram matrix | solve time |
|---|---|---|
| 8k | 0.5 GB | 2.3 s (measured) |
| 16k | 2 GB | 18 s |
| 32k | 8 GB | 2.5 min |
| 50k | 20 GB | exceeds 18 GB |

The hard ceiling on 18 GB is around N = 35k. `run_exact_kelm.py` stops there and records
the overrun rather than subsampling to continue.

## Parallelism

The full suite runs overnight in a single process. If parallelising, use four processes
with `OMP_NUM_THREADS=2`; measured speedup is about 3x, since the efficiency cores are
roughly a third the speed of the performance cores and sustained load throttles.

`measure_cost.py` must run single-process. Concurrent load makes its timings meaningless.

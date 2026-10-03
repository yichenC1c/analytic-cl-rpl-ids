# Result logs

These JSON files are the compact experiment records used by the manuscript. They are
included so the reported aggregates can be audited without retraining the encoder or
rerunning the benchmark.

| Prefix | Contents |
|---|---|
| `E0_probe` | Joint-fitting diagnostic for one frozen representation |
| `E1_ours` | Main analytic-head runs for four orderings, three runs, and two feature widths |
| `E2_replay` | Metrics extracted from the released Experience Replay implementation |
| `raw_replay` | Unmodified output records produced by that implementation |
| `E3_kelm` | Dense exact-kernel runs stopped at the configured sample cutoff |
| `E4_equivalence` | Recursive weights compared with joint ridge fitting after every domain |
| `E5_dsweep` | Random-feature width sweep under the Random ordering |
| `E6_cost` | Parameter, state, operation, and latency measurements |
| `E_select_hyperparams` | Validation-grid output used to choose width and ridge coefficient |

Run the consistency summary from the repository root:

```bash
python src/audit_results.py
```

Timing fields are measurements from the platform recorded in each relevant file. They
should not be interpreted as edge-device latency. The `cutoff` status in `E3_kelm` means
the next dense fit exceeded the configured 35,000-sample limit. It is not an operating
system out-of-memory report.

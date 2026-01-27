# Refactor bundle (runner architecture)

This bundle demonstrates the recommended split:

- `ops/a2a_core.py`: **SSOT** for kernel composition (Stage-1 + Stage-2 launches)
- `layers/all_to_all.py`: thin façade to preserve `layers.all_to_all` import workflow
- `runtime/`:
  - `executor.py`: executes custom/baseline once (no timing, no checking)
  - `timer.py`: CUDA-event timing wrapper
  - `sync.py`: reset + (optional) spin-wait helpers
  - `checker.py`: correctness comparisons/report
  - `orchestrator.py`: wires the above together for a `run_case(...)` API

Entry scripts:
- `benchmark_refactored.py`: spawn + run_case (perf/both)
- `unit_tests_refactored.py`: spawn + run_case (correctness)

## Integration plan

1. Copy `runtime/`, `ops/`, and `layers/all_to_all.py` into your repo.
2. Update your existing `benchmark.py` / `unit_tests.py` to call `runtime.orchestrator.run_case`.
3. Gradually delete duplicated code from `benchmark.py` / `unit_tests.py` once you trust the new paths.

Notes:
- This bundle keeps your existing `utils.py`, `kernels.py`, `baseline.py` as-is (copied).
- Next step is to **remove internal timing/sync** from `baseline.py` and measure baseline via `runtime.timer.time_cuda_ms`.

# Benchmarking Guide — Memory Layer AI

This guide explains how to run the built-in performance benchmark suite and record reproducible baseline numbers.

---

## What It Measures

The benchmark suite currently measures two high-signal operations:

- `save` latency and throughput
- `recall` latency and throughput

It runs against an in-memory benchmark harness so results are reproducible and not dominated by external storage/network variability.

---

## Run The Suite

From the repository root:

```bash
python scripts/benchmark/run_benchmark_suite.py
```

On Windows with the project virtual environment:

```powershell
d:/github/Memory-Layer-AI/.venv/Scripts/python.exe scripts/benchmark/run_benchmark_suite.py
```

---

## Common Options

```bash
python scripts/benchmark/run_benchmark_suite.py \
  --save-count 1000 \
  --recall-count 500 \
  --warmup-saves 100 \
  --warmup-recalls 50 \
  --top-k 5 \
  --token-budget 512 \
  --seed 42 \
  --format json \
  --output-file ./benchmark-results/local-baseline.json
```

Key flags:

- `--save-count`: Number of measured save operations.
- `--recall-count`: Number of measured recall operations.
- `--warmup-saves`: Save operations executed before timing starts.
- `--warmup-recalls`: Recall operations executed before timing starts.
- `--top-k`: Recall `top_k` used in measured runs.
- `--token-budget`: Recall token budget used in measured runs.
- `--seed`: Deterministic random seed for synthetic workload generation.
- `--format`: `text` or `json` output.
- `--output-file`: Optional path to write the report.

---

## Report Fields

For each operation (`save`, `recall`), the report includes:

- `count`
- `duration_seconds`
- `throughput_ops_per_second`
- `latency_ms`:
  - `min_ms`
  - `mean_ms`
  - `p50_ms`
  - `p95_ms`
  - `p99_ms`
  - `max_ms`

Use `p95_ms` as your primary latency KPI and throughput as a secondary scaling signal.

---

## Baseline Workflow

1. Run the benchmark 3-5 times on the same machine and environment.
2. Store JSON outputs under `benchmark-results/` (gitignored or attached in CI artifacts).
3. Compare current `p95_ms` and throughput against your baseline.
4. Investigate any regression greater than 10% before merging.

---

## Notes

- This suite is designed for regression tracking and relative comparisons.
- Absolute production performance depends on storage backend, embedding model, dataset shape, and hardware.

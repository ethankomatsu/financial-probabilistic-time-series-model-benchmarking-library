# FinProbTS-Bench

FinProbTS-Bench is a research-grade benchmarking library for probabilistic financial time series forecasting. It provides a common data contract, reproducible experiment runner, model registry, synthetic financial stress datasets, and evaluation stack for comparing probabilistic forecasting models on real CRSP-style panels and controlled simulated data.

The design follows the blueprint in the project slides: split-safe preprocessing, train-only standardization, rolling windows, sample-based probabilistic outputs, finance-aware diagnostics, and CLI-driven runs similar in spirit to the Time-Series-Library workflow.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,torch,metrics,parquet]
```

Use a smaller install when needed:

```bash
python -m pip install -e .[dev]
python -m pip install -e .[torch]
python -m pip install -e .[metrics]
```

## Repository Layout

```text
finprobts/data          canonical loaders, schemas, preprocessing, windows
finprobts/models        model interface, registry, native adapters, references
finprobts/evaluation    metric wrappers and finance diagnostics
finprobts/experiment    runners for single runs and sweeps
finprobts/simulators    financial data-generating processes
finprobts/synthetic     five-level synthetic benchmark generation
configs                 dataset, task, model, and run YAMLs
tests                   unit and smoke tests
```

## Data Contract

All datasets are loaded into `FinancialDataset`:

```text
values:    [time, assets]
dates:     [time]
asset_ids: [assets]
```

Rolling windows use:

```text
x_context: [num_windows, context_length, num_assets]
y_target:  [num_windows, prediction_length, num_assets]
```

Forecast outputs use:

```text
samples: [num_windows, num_samples, prediction_length, num_assets]
y_true:  [num_windows, prediction_length, num_assets]
```

Supported input formats:

```text
wide CSV/Parquet: date, asset_1, asset_2, ...
long CSV/Parquet: date, asset_id, target, optional_feature_1, ...
```

CRSP data is treated as private local data. The repo includes loader/config support, but raw licensed data should stay untracked. See `configs/dataset/crsp_local.yaml` and `configs/runs/crsp_deepvar.yaml`.

## Synthetic Data

The benchmark includes six simulator families, each with five difficulty levels:

| Case | Simulator |
| --- | --- |
| `case1_garch` | factor/idiosyncratic GARCH volatility clustering |
| `case2_har` | HAR multi-scale volatility memory |
| `case3_heavy_tail` | heavy tails and rare outlier contamination |
| `case4_regime` | market-wide block Markov regimes |
| `case5_hawkes` | market-wide self-exciting jumps |
| `case6_zip_panel` | zero-inflated Poisson jumps with panel exposure |

Generate all datasets:

```bash
finprobts generate-synthetic \
  --case all \
  --levels 1,2,3,4,5 \
  --out-dir data/simulated \
  --base-seed 123 \
  --T 20000 \
  --n-firms 50 \
  --formats csv
```

For a quick smoke run:

```bash
finprobts generate-synthetic --case case1_garch --levels 1 --T 300 --n-firms 5 --out-dir data/simulated
```

## Running Experiments

Config-first workflow:

```bash
finprobts run --config configs/example_crypto_naive.yaml
finprobts run --config configs/runs/crsp_deepvar.yaml
```

Time-Series-Library-style flag workflow:

```bash
finprobts run \
  --task_name probabilistic_forecast \
  --is_training 1 \
  --model DeepVAR \
  --data crsp \
  --root_path data/crsp \
  --data_path crsp_returns.parquet \
  --data_format long \
  --date_column date \
  --asset_id_column permno \
  --target_column ret \
  --seq_len 96 \
  --pred_len 1 \
  --num_samples 100 \
  --train_epochs 10 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --device auto
```

Run a generated synthetic suite:

```bash
finprobts run-synthetic-suite \
  --manifest data/simulated/manifest.json \
  --models deepvar,timegrad,quantileformer \
  --output-dir outputs/synthetic_suite \
  --context-length 96 \
  --prediction-length 1 \
  --num-samples 100 \
  --device auto
```

Dry-run config generation:

```bash
finprobts run-synthetic-suite --models all --dry-run
```

Re-evaluate saved forecasts:

```bash
finprobts evaluate --run-dir outputs/example_crypto_naive
```

## Model Roster And Provenance

| Model | Status |
| --- | --- |
| DeepVAR 2019 | Native PyTorch adapter aligned with PyTorchTS/GluonTS DeepVAR and seeded by the old repo implementation |
| TimeGrad 2021 | Native adapter targeting PyTorchTS TimeGrad architecture |
| TimeMCL 2025 ICML | Native adapter based on official TimeMCL design notes |
| TSFlow 2025 ICLR | Paper/repo-faithful native adapter with documented dependency/licensing deviations |
| RATD 2024 NeurIPS | Native retrieval-augmented diffusion adapter based on official RATD |
| QuantileFormer 2025 IJCAI | Paper-faithful native quantile Transformer; no official source code identified |
| Naive | Sanity-check baseline, not intended as a headline leaderboard model |

Every deep model includes a `REFERENCE.md` under `finprobts/models/<model>/` documenting papers, official repositories, licensing notes, implemented components, and deviations.

## Metrics

The benchmark prefers established metric packages for probabilistic scoring:

```text
scoringrules -> preferred sample CRPS backend when installed
properscoring -> fallback sample CRPS backend when installed
local fallback -> deterministic CRPS estimator if neither package is available
```

Reported metrics include:

```text
Point:          MAE, RMSE, MAPE, ND, volatility-normalized MAE
Probabilistic: CRPS, CRPS-Sum, normalized CRPS-Sum, quantile loss, coverage
Optional:      energy score
Finance:       VaR violation rate, expected shortfall, volatility forecast error, correlation forecast error
```

Metric backend versions are written to `run_metadata.json`.

## Output Artifacts

Each run writes:

```text
config.yaml
forecast_samples.npz
forecast_metrics.json
run_metadata.json
```

`run_metadata.json` records the resolved config path, seed, git commit, package version, platform, metric backend versions, dataset fingerprint, elapsed time, and output location.

## Adding A Model

1. Create `finprobts/models/<model>/model.py`.
2. Implement `BaseProbForecastModel.fit` and `predict`.
3. Return `ForecastResult` with samples shaped `[windows, samples, horizon, assets]`.
4. Add `finprobts/models/<model>/REFERENCE.md`.
5. Register the model in `finprobts/models/registry.py`.
6. Add `configs/model/<model>.yaml`.
7. Add a tiny CPU smoke test.

Keep adapters thin around the common contract. Model-specific data transformations are allowed internally, but experiment runners and metrics should not need model-specific branches.

## Reproducibility Checklist

- Pin seeds in run configs.
- Record exact configs and run metadata.
- Keep raw CRSP/local data out of git.
- Prefer package-backed probabilistic metrics.
- Document model deviations in `REFERENCE.md`.
- Run tests before reporting results:

```bash
python -m pytest -q
```

## Current Test Status

The test suite covers data loading, preprocessing, rolling windows, metrics, synthetic generation, CLI runs, and CPU smoke tests for native torch models.

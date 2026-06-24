"""Configurable experiment runner for FinProbTS-Bench."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import finprobts
from finprobts.config import load_yaml_config, save_yaml_config
from finprobts.data import (
    concatenate_financial_datasets,
    DatasetNormalizer,
    FinancialDataset,
    generate_boundary_aware_rolling_windows,
    generate_rolling_windows,
    get_default_dataset_registry,
    handle_missing_values_split_safe,
    price_to_log_return,
    time_train_val_test_split,
)
from finprobts.evaluation import evaluate_forecasts, metric_backend_versions
from finprobts.models import ForecastResult, get_default_model_registry


@dataclass
class ExperimentResult:
    """Paths and metrics produced by an experiment run."""

    output_dir: Path
    forecast_result: ForecastResult
    forecast_metrics: Dict[str, Any]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, allow_nan=False)


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _dataset_fingerprint(dataset: FinancialDataset) -> Dict[str, Any]:
    values = dataset.values
    return {
        "num_timesteps": int(values.shape[0]),
        "num_assets": int(values.shape[1]),
        "asset_ids_head": list(dataset.asset_ids[:5]),
        "date_start": str(dataset.dates[0]) if len(dataset.dates) else None,
        "date_end": str(dataset.dates[-1]) if len(dataset.dates) else None,
        "metadata": dict(dataset.metadata),
    }


def _run_metadata(
    *,
    config_path: str,
    config: Dict[str, Any],
    dataset: FinancialDataset,
    output_dir: Path,
    started_at: float,
) -> Dict[str, Any]:
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "run_id": config.get("run", {}).get("run_id"),
        "seed": config.get("run", {}).get("seed"),
        "package_version": getattr(finprobts, "__version__", None),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "metric_backend_versions": metric_backend_versions(),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "elapsed_seconds": float(time.time() - started_at),
    }


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    run_config = config.get("run", {})
    run_id = run_config.get("run_id") or "finprobts_run"
    output_root = Path(run_config.get("output_dir", "outputs"))
    return output_root / str(run_id)


def _load_dataset(config: Dict[str, Any]) -> FinancialDataset:
    dataset_config = dict(config.get("dataset", {}))
    if not dataset_config:
        raise ValueError("Config must include a dataset section.")

    dataset_name = dataset_config.pop("name", None)
    if not dataset_name:
        raise ValueError("dataset.name is required.")

    registry = get_default_dataset_registry()
    return registry.load(dataset_name, **dataset_config)


def _preprocess_dataset(dataset: FinancialDataset, config: Dict[str, Any]) -> FinancialDataset:
    preprocessing = config.get("preprocessing", {})
    value_kind = preprocessing.get("value_kind", dataset.metadata.get("value_kind", "returns"))
    if preprocessing.get("price_to_log_return", False) or value_kind == "prices":
        dataset = price_to_log_return(dataset)
    return dataset


def _make_windows(config: Dict[str, Any], dataset: FinancialDataset):
    split_config = config.get("split", {})
    task_config = config.get("task", {})
    split = time_train_val_test_split(
        dataset,
        train_size=float(split_config.get("train_size", 0.6)),
        val_size=float(split_config.get("val_size", 0.2)),
        test_size=split_config.get("test_size"),
    )

    missing_method = config.get("preprocessing", {}).get("missing_method", "ffill")
    split = handle_missing_values_split_safe(split, method=missing_method)

    standardize = bool(config.get("preprocessing", {}).get("standardize", True))
    normalizer: Optional[DatasetNormalizer] = None
    if standardize:
        normalizer = DatasetNormalizer.fit(split.train)
        split = type(split)(
            train=normalizer.transform_dataset(split.train),
            val=normalizer.transform_dataset(split.val),
            test=normalizer.transform_dataset(split.test),
        )

    context_length = int(task_config.get("context_length", 96))
    prediction_length = int(task_config.get("prediction_length", 1))
    stride = int(task_config.get("stride", 1))

    train_windows = generate_rolling_windows(split.train, context_length, prediction_length, stride)
    val_windows = generate_boundary_aware_rolling_windows(
        split.train,
        split.val,
        context_length,
        prediction_length,
        stride,
        metadata={"split_role": "validation"},
    )
    test_history = concatenate_financial_datasets(
        [split.train, split.val],
        metadata={"split_role": "test_history"},
    )
    test_windows = generate_boundary_aware_rolling_windows(
        test_history,
        split.test,
        context_length,
        prediction_length,
        stride,
        metadata={"split_role": "test"},
    )
    return train_windows, val_windows, test_windows, normalizer


def _make_model(config: Dict[str, Any]):
    model_config = dict(config.get("model", {}))
    model_name = model_config.pop("name", None) or model_config.pop("type", None)
    if not model_name:
        raise ValueError("model.name is required.")
    params = dict(model_config.pop("params", {}))
    params.update(model_config)

    run_seed = config.get("run", {}).get("seed")
    params.setdefault("seed", run_seed)
    registry = get_default_model_registry()
    return registry.create(model_name, **params)


def _inverse_result(result: ForecastResult, normalizer: Optional[DatasetNormalizer]) -> ForecastResult:
    if normalizer is None:
        return result
    metadata = dict(result.metadata)
    metadata["inverse_transformed"] = True
    return ForecastResult(
        samples=normalizer.inverse_transform_samples(result.samples),
        y_true=normalizer.inverse_transform_targets(result.y_true),
        start_dates=result.start_dates,
        item_ids=list(result.item_ids),
        metadata=metadata,
    )


def run_experiment(config_path: str) -> ExperimentResult:
    """Run a full forecasting experiment from a YAML config."""

    started_at = time.time()
    config = load_yaml_config(config_path)
    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(config)
    dataset = _preprocess_dataset(dataset, config)
    train_windows, val_windows, test_windows, normalizer = _make_windows(config, dataset)

    model = _make_model(config)
    model.fit(train_windows, val_windows)

    forecast_config = config.get("forecast", {})
    num_samples = int(forecast_config.get("num_samples", 100))
    forecast_result = model.predict(test_windows, num_samples=num_samples)
    eval_result = _inverse_result(forecast_result, normalizer)

    metrics_config = config.get("evaluation", {})
    forecast_metrics = evaluate_forecasts(
        eval_result,
        quantiles=metrics_config.get("quantiles", (0.1, 0.5, 0.9)),
        coverage_levels=metrics_config.get("coverage_levels", (0.5, 0.9)),
        var_alpha=float(metrics_config.get("var_alpha", 0.05)),
        include_multivariate_scores=bool(metrics_config.get("include_multivariate_scores", False)),
    )

    eval_result.save_npz(str(output_dir / "forecast_samples.npz"))
    _write_json(forecast_metrics, output_dir / "forecast_metrics.json")
    _write_json(
        _run_metadata(
            config_path=config_path,
            config=config,
            dataset=dataset,
            output_dir=output_dir,
            started_at=started_at,
        ),
        output_dir / "run_metadata.json",
    )
    save_yaml_config(config, str(output_dir / "config.yaml"))

    return ExperimentResult(
        output_dir=output_dir,
        forecast_result=eval_result,
        forecast_metrics=forecast_metrics,
    )

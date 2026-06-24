"""Native QuantileFormer-style probabilistic forecaster.

This implementation follows the paper-level design of direct multi-horizon
quantile forecasting with a Transformer backbone and pinball loss. No official
source code was available during implementation, so the model card records this
as a paper-faithful native adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from finprobts.data.schema import RollingWindowDataset
from finprobts.models.base import BaseProbForecastModel, ForecastResult
from finprobts.models.torch_utils import (
    iter_torch_batches,
    make_torch_data_loader,
    require_torch,
    resolve_torch_device,
    set_torch_seed,
)


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def _parse_quantiles(quantiles: Optional[Iterable[float]]) -> list[float]:
    values = list(quantiles if quantiles is not None else (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    values = sorted({float(q) for q in values})
    if not values or values[0] <= 0.0 or values[-1] >= 1.0:
        raise ValueError("quantiles must be inside (0, 1).")
    return values


class QuantileFormerNetwork(nn.Module if nn is not None else object):
    """Transformer encoder with direct horizon/asset quantile heads."""

    def __init__(
        self,
        num_assets: int,
        context_length: int,
        prediction_length: int,
        quantiles: list[float],
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        self.num_assets = int(num_assets)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.quantiles = [float(q) for q in quantiles]
        self.input_proj = nn.Linear(self.num_assets, int(d_model))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.context_length, int(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(
            int(d_model),
            self.prediction_length * self.num_assets * len(self.quantiles),
        )

    def forward(self, past_target: Any) -> Any:
        encoded = self.input_proj(past_target) + self.positional_embedding[:, : past_target.shape[1], :]
        encoded = self.encoder(encoded)
        pooled = self.norm(encoded[:, -1, :])
        raw = self.head(pooled)
        return raw.reshape(
            past_target.shape[0],
            self.prediction_length,
            self.num_assets,
            len(self.quantiles),
        )


class QuantileFormerForecastModel(BaseProbForecastModel):
    """Paper-faithful QuantileFormer-style adapter for FinProbTS."""

    def __init__(
        self,
        quantiles: Optional[Iterable[float]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 10,
        patience: int = 3,
        grad_clip: float = 1.0,
        scaling: bool = True,
        device: Optional[str] = "auto",
        seed: Optional[int] = None,
    ) -> None:
        self.quantiles = _parse_quantiles(quantiles)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.scaling = bool(scaling)
        self.device_name = device
        self.seed = seed
        self.device = None
        self.network = None
        self.context_length = None
        self.prediction_length = None
        self.num_assets = None

    def _init_network(self, windows: RollingWindowDataset) -> None:
        require_torch()
        self.device = resolve_torch_device(self.device_name)
        self.context_length = int(windows.context_length)
        self.prediction_length = int(windows.prediction_length)
        self.num_assets = int(windows.num_assets)
        self.network = QuantileFormerNetwork(
            num_assets=self.num_assets,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            quantiles=self.quantiles,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

    def _loss(self, prediction: Any, target: Any, observed: Any) -> Any:
        qs = torch.tensor(self.quantiles, device=prediction.device, dtype=prediction.dtype)
        error = target.unsqueeze(-1) - prediction
        loss = torch.maximum(qs * error, (qs - 1.0) * error)
        weights = observed.unsqueeze(-1)
        denom = weights.sum().clamp_min(1.0) * len(self.quantiles)
        return (loss * weights).sum() / denom

    def _epoch_loss(self, loader: Any, train: bool, optimizer: Optional[Any] = None) -> float:
        assert self.network is not None
        self.network.train(train)
        total = 0.0
        count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in iter_torch_batches(loader, self.device):
                if train:
                    assert optimizer is not None
                    optimizer.zero_grad()
                prediction = self.network(batch["past_target"])
                loss = self._loss(
                    prediction,
                    batch["future_target"],
                    batch["future_observed_values"],
                )
                if train:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
        return total / max(count, 1)

    def fit(self, train_data: RollingWindowDataset, val_data: Optional[RollingWindowDataset] = None) -> None:
        require_torch()
        set_torch_seed(self.seed)
        if len(train_data) == 0:
            raise ValueError("train_data must contain at least one window.")
        self._init_network(train_data)
        assert self.network is not None
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        train_loader = make_torch_data_loader(train_data, self.batch_size, shuffle=True, include_time_features=False)
        val_loader = (
            make_torch_data_loader(val_data, self.batch_size, shuffle=False, include_time_features=False)
            if val_data is not None and len(val_data) > 0
            else None
        )

        best_state = None
        best_loss = float("inf")
        stale = 0
        for _ in range(self.max_epochs):
            self._epoch_loss(train_loader, train=True, optimizer=optimizer)
            current = self._epoch_loss(val_loader, train=False) if val_loader is not None else self._epoch_loss(train_loader, train=False)
            if current < best_loss:
                best_loss = current
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _sample_from_quantiles(self, quantile_values: np.ndarray, num_samples: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        levels = rng.uniform(0.0, 1.0, size=(quantile_values.shape[0], num_samples))
        q = np.asarray(self.quantiles, dtype=float)
        samples = np.empty((quantile_values.shape[0], num_samples, quantile_values.shape[1], quantile_values.shape[2]), dtype=float)
        for window_idx in range(quantile_values.shape[0]):
            for horizon_idx in range(quantile_values.shape[1]):
                for asset_idx in range(quantile_values.shape[2]):
                    values = quantile_values[window_idx, horizon_idx, asset_idx]
                    # Enforce monotone quantiles at inference to avoid crossing
                    # artifacts from finite training.
                    values = np.maximum.accumulate(values)
                    samples[window_idx, :, horizon_idx, asset_idx] = np.interp(
                        levels[window_idx],
                        q,
                        values,
                        left=values[0],
                        right=values[-1],
                    )
        return samples

    def predict(self, test_data: RollingWindowDataset, num_samples: int) -> ForecastResult:
        if self.network is None or self.device is None:
            raise RuntimeError("Call fit before predict.")
        loader = make_torch_data_loader(test_data, self.batch_size, shuffle=False, include_time_features=False)
        predictions = []
        self.network.eval()
        with torch.no_grad():
            for batch in iter_torch_batches(loader, self.device):
                predictions.append(self.network(batch["past_target"]).detach().cpu().numpy())
        quantile_values = np.concatenate(predictions, axis=0)
        samples = self._sample_from_quantiles(quantile_values, int(num_samples))
        return ForecastResult(
            samples=samples,
            y_true=test_data.y_target,
            start_dates=test_data.start_dates,
            item_ids=list(test_data.asset_ids),
            metadata={
                "model_name": "quantileformer",
                "implementation_status": "paper_faithful_native_no_official_code",
                "quantiles": self.quantiles,
                "seed": self.seed,
            },
        )

    def save(self, path: str) -> None:
        if self.network is None:
            raise RuntimeError("Cannot save before fit.")
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), output_dir / "model.pt")
        with open(output_dir / "model.json", "w", encoding="utf-8") as handle:
            json.dump(self._config(), handle, indent=2)

    def _config(self) -> Dict[str, Any]:
        return {
            "quantiles": self.quantiles,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "scaling": self.scaling,
            "device": self.device_name,
            "seed": self.seed,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "num_assets": self.num_assets,
        }

    @classmethod
    def load(cls, path: str) -> "QuantileFormerForecastModel":
        require_torch()
        model_dir = Path(path)
        with open(model_dir / "model.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        context_length = payload.pop("context_length")
        prediction_length = payload.pop("prediction_length")
        num_assets = payload.pop("num_assets")
        model = cls(**payload)
        model.device = resolve_torch_device(model.device_name)
        model.context_length = int(context_length)
        model.prediction_length = int(prediction_length)
        model.num_assets = int(num_assets)
        model.network = QuantileFormerNetwork(
            num_assets=model.num_assets,
            context_length=model.context_length,
            prediction_length=model.prediction_length,
            quantiles=model.quantiles,
            d_model=model.d_model,
            n_heads=model.n_heads,
            num_layers=model.num_layers,
            dim_feedforward=model.dim_feedforward,
            dropout=model.dropout,
        ).to(model.device)
        model.network.load_state_dict(
            torch.load(model_dir / "model.pt", map_location=model.device)
        )
        model.network.eval()
        return model

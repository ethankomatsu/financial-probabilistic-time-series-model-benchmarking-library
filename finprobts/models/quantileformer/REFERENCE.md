# QuantileFormer Reference Note

Reference:
- Target method: QuantileFormer, IJCAI 2025, probabilistic time-series forecasting.
- Official code: no official implementation was identified during project planning. This adapter is therefore a paper-faithful native implementation, not an official-code port.

Implementation notes:
- Uses the FinProbTS rolling-window contract as the public interface.
- Trains a Transformer encoder over context windows with learned positional embeddings.
- Predicts a fixed grid of marginal quantiles for every forecast horizon and asset.
- Optimizes multi-quantile pinball loss.
- Converts quantile forecasts into benchmark samples by drawing quantile levels and linearly interpolating the learned quantile function.

Deviations and limitations:
- The benchmark output contract requires samples, so quantile outputs are sampled through interpolation for CRPS/coverage metrics.
- Without confirmed official source code, hyperparameter defaults and implementation details should be treated as paper-faithful but independently implemented.
- If official code becomes available, this adapter should be audited against its architecture, training objective, preprocessing, and inference procedure.

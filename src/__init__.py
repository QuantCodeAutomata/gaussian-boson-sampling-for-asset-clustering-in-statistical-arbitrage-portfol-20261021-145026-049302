"""
GBS Statistical Arbitrage: source package.

Modules:
- config: hyperparameters and experiment settings
- data_loader: price and factor data retrieval
- returns: total return and beta residualisation
- correlation: correlation matrix and RMT cleaning
- gbs_sampling: fast GBS threshold sampling (numpy approximation)
- gbs_boost: GBS Boost clustering algorithm
- gbs_roots: GBS Roots clustering algorithm
- qic_gbs: QIC-GBS classical approximation
- spectral: spectral clustering with dynamic K
- sponge: symmetric SPONGE clustering
- portfolio: contrarian StatArb portfolio construction
- metrics: performance and partition quality metrics
- backtester: rolling backtest engine
"""

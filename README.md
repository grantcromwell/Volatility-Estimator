# Volatility Estimator

A comprehensive volatility estimation system implementing industry-standard models and advanced techniques.

## Features

### Industry-Standard Models
- **EWMA (Exponentially Weighted Moving Average)**: RiskMetrics-standard volatility estimation
- **GARCH Models**: GARCH(1,1), EGARCH, GJR-GARCH for volatility clustering
- **Range-Based Estimators**: Garman-Klass, Parkinson, Yang-Zhang (5-8x more efficient than close-to-close)

### Advanced Techniques
- **Prim's Minimum Spanning Tree**: Correlation analysis and network topology
- **Lorenz Attractor**: Chaotic dynamics modeling for non-linear patterns
- **Hidden Markov Model**: Regime detection and state-based volatility

### Enterprise Features
- **Validation Framework**: Accuracy metrics (MAE, RMSE, MAPE, R²)
- **Backtesting**: Walk-forward validation and model comparison
- **Sector Analysis**: Sector volatility assessment and 60-day projections
- **Ensemble Methods**: Weighted combination of multiple models
- **Production Logging**: Structured logging with performance metrics
- **Market Assessment**: Overall market volatility (0-100% scale)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd VolEstimator

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Optional: GARCH Support

For full GARCH model support, install the `arch` library:

```bash
pip install arch
```

## Quick Start

### Production Script

```bash
# Assess overall market volatility
python assess_market_volatility.py

# With favorite assets
python assess_market_volatility.py --favorites AAPL MSFT GOOGL NVDA

# Custom period
python assess_market_volatility.py --period 6mo --favorites TSLA NVDA
```

### Programmatic Usage

```python
from vol_estimator import VolatilityEstimator

# Initialize with enterprise features
estimator = VolatilityEstimator(
    mst_enabled=True,
    lorenz_enabled=True,
    hmm_enabled=True,
    ewma_enabled=True,
    garch_enabled=False  # Requires arch library
)

# Fetch data
data = estimator.fetch_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'])

# Prepare returns
returns, symbols = estimator.prepare_returns(data)

# Estimate volatility using ensemble
volatility = estimator.estimate_ensemble(returns, symbols)
print(volatility)
```

## Detailed Usage

### EWMA Volatility (Industry Standard)

```python
from vol_estimator import EWMAVolatilityEstimator
import numpy as np

returns = np.random.randn(252) * 0.02  # Daily returns

# RiskMetrics standard (decay=0.94)
ewma = EWMAVolatilityEstimator(decay_factor=0.94)
ewma.fit(returns)

current_vol = ewma.get_current_volatility()
forecast = ewma.forecast(horizon=5)  # 5-day forecast
```

### Range-Based Estimators

```python
from vol_estimator import (
    GarmanKlassVolatilityEstimator,
    YangZhangVolatilityEstimator,
    compare_estimators
)

# Garman-Klass (8x more efficient than close-to-close)
gk = GarmanKlassVolatilityEstimator()
gk.fit(high, low, close, open_)
volatility = gk.get_volatility()

# Compare all range estimators
results = compare_estimators(high, low, close, open_)
```

### GARCH Models

```python
from vol_estimator import GARCHVolatilityEstimator

# Standard GARCH(1,1)
garch = GARCHVolatilityEstimator(p=1, q=1)
garch.fit(returns, disp='off')

conditional_vol = garch.get_conditional_volatility()
forecast = garch.forecast(horizon=60)  # 60-day forecast
```

### Ensemble Estimation

```python
estimator = VolatilityEstimator(
    ewma_enabled=True,
    garch_enabled=True,
    hmm_enabled=True,
    ensemble_weights={
        'ewma': 0.35,
        'garch': 0.25,
        'hmm': 0.20,
        'mst': 0.15,
        'lorenz': 0.05
    }
)

volatility = estimator.estimate_ensemble(returns, symbols)
```

### Validation and Backtesting

```python
from vol_estimator import VolatilityBacktester, ModelValidator

# Backtesting
backtester = VolatilityBacktester(
    train_size=252,  # 1 year
    test_size=60,    # 3 months
    step_size=20     # Re-fit every 20 days
)

results = backtester.compare_models(models, returns)

# Validation
validator = ModelValidator()
validation = validator.validate_model(predictions, actuals, "ModelName")
```

### Sector Volatility Analysis

```python
# Sector analysis is built into the production script
# assess_market_volatility.py automatically includes:
# - Most volatile sector this week
# - Projected most volatile sector (next 60 days)
# - Top 5 sectors by volatility
```

## Project Structure

```
VolEstimator/
├── vol_estimator/
│   ├── __init__.py
│   ├── estimator.py              # Main estimator with ensemble
│   ├── ewma_volatility.py        # EWMA implementation
│   ├── garch_volatility.py       # GARCH models
│   ├── range_estimators.py        # Range-based estimators
│   ├── correlation_mst.py         # Prim's MST correlation
│   ├── lorenz_attractor.py        # Lorenz attractor dynamics
│   ├── hmm_regime.py              # Hidden Markov Model
│   ├── validation.py              # Validation & backtesting
│   ├── logging_config.py          # Enterprise logging
│   ├── api_clients.py             # API integrations
│   └── sector_dfs.py              # Sector analysis
├── tests/
│   ├── test_estimator.py          # Core tests
│   └── test_enterprise_models.py  # Enterprise model tests
├── assess_market_volatility.py    # Production script
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Components

### Industry-Standard Models

#### EWMA (Exponentially Weighted Moving Average)
- RiskMetrics methodology (decay factor 0.94)
- Single-asset and multi-asset support
- Online update capability
- Volatility forecasting

#### GARCH Models
- **GARCH(1,1)**: Standard GARCH for volatility clustering
- **EGARCH**: Exponential GARCH for asymmetric effects
- **GJR-GARCH**: Leverage effects modeling
- Conditional volatility estimation
- Multi-period forecasting

#### Range-Based Estimators
- **Parkinson**: Uses high/low prices (5x more efficient)
- **Garman-Klass**: Uses OHLC (8x more efficient)
- **Rogers-Satchell**: Handles drift
- **Yang-Zhang**: Most efficient, handles drift and opening jumps

### Advanced Techniques

#### Prim's Minimum Spanning Tree
- Correlation network analysis
- Core asset identification
- MST-weighted volatilities

#### Lorenz Attractor
- Chaotic volatility dynamics
- Regime transition detection
- Lyapunov exponent computation

#### Hidden Markov Model
- Volatility regime detection
- State transition probabilities
- Regime-adjusted estimates

### Enterprise Features

#### Validation Framework
- Accuracy metrics (MAE, RMSE, MAPE, R²)
- Walk-forward backtesting
- Model comparison and benchmarking
- Comprehensive validation checks

#### Sector Analysis
- Sector ETF tracking (XLK, XLF, XLV, etc.)
- Weekly volatility assessment
- 60-day volatility projections
- Top sectors ranking

## API Integration

### Supported APIs

1. **Yahoo Finance** (via yfinance)
   - No API key required
   - Stocks and cryptocurrencies
   - Primary data source

2. **Alpha Vantage**
   - Free tier: 5 calls/minute, 500 calls/day
   - Get free key: https://www.alphavantage.co/support/#api-key
   - Stocks and forex

3. **CoinGecko**
   - Free tier: 10-50 calls/minute
   - No API key required
   - Cryptocurrencies

4. **Finnhub** (optional)
   - Sector information
   - Company profiles

### API Key Setup (Optional)

```python
# With API keys
estimator = VolatilityEstimator(
    alpha_vantage_key="YOUR_AV_KEY",
    finnhub_key="YOUR_FH_KEY"
)
```

## CPU Optimization

The implementation is optimized for CPU performance:

1. **NumPy Vectorization**: All computations use vectorized operations
2. **Numba JIT**: Critical functions compiled with Numba for speed
3. **Efficient Algorithms**: Uses optimized implementations from NetworkX, SciPy
4. **Memory Efficiency**: Minimizes data copying and intermediate arrays

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=vol_estimator --cov-report=html
```

Run enterprise model tests:

```bash
pytest tests/test_enterprise_models.py -v
```

## Production Script

The `assess_market_volatility.py` script provides:

- **Overall Market Volatility**: 0-100% normalized scale
- **Market Index Analysis**: SPY, QQQ, DIA, IWM volatilities
- **Sector Volatility**: Most volatile sector this week
- **Sector Projections**: Projected most volatile sector (60 days)
- **Favorite Assets**: Custom asset volatility assessment

### Usage Examples

```bash
# Basic market assessment
python assess_market_volatility.py

# With favorites
python assess_market_volatility.py --favorites AAPL MSFT GOOGL NVDA AMZN

# 6-month period
python assess_market_volatility.py --period 6mo

# Disable ensemble (faster)
python assess_market_volatility.py --no-ensemble
```

## Dependencies

### Core
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `pandas>=2.0.0` - Data manipulation

### Algorithms
- `networkx>=3.1` - Graph algorithms (Prim's MST)
- `hmmlearn>=0.3.0` - Hidden Markov Models
- `arch>=6.2.0` - GARCH models (optional)

### Performance
- `numba>=0.57.0` - JIT compilation

### APIs
- `requests>=2.31.0` - HTTP requests
- `yfinance>=0.2.28` - Yahoo Finance client

### Testing
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting

### Optional
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical plots

## Advanced Features

### Validation & Accuracy
- ✅ Accuracy metrics (MAE, RMSE, MAPE, R²)
- ✅ Backtesting framework
- ✅ Model comparison and benchmarking
- ✅ Validation against realized volatility

### Production Readiness
- ✅ Enterprise logging with structured output
- ✅ Performance metrics tracking
- ✅ Error handling and graceful degradation
- ✅ Comprehensive test coverage (24+ tests)

### Industry Standards
- ✅ EWMA (RiskMetrics standard)
- ✅ GARCH family models
- ✅ Range-based estimators
- ✅ Ensemble methods

## Limitations

- Free API tiers have rate limits
- GARCH models require `arch` library (optional)
- HMM requires sufficient historical data (typically 100+ periods)
- MST analysis requires multiple assets (3+ recommended)

## Contributing

Feedback and contributions are welcome. Please ensure:
- Code follows PEP 8 style guidelines
- Tests pass (`pytest tests/`)
- New features include tests
- Documentation is updated
- Enterprise logging is used (no `print()` statements)

## References

- **EWMA**: RiskMetrics methodology
- **GARCH**: Bollerslev (1986), Nelson (1991) EGARCH
- **Range Estimators**: Garman-Klass (1980), Yang-Zhang (2000)
- **Prim's Algorithm**: Minimum Spanning Tree for correlation networks
- **Lorenz Attractor**: Chaotic systems in financial modeling
- **Hidden Markov Models**: Regime-switching volatility models

## License

MIT License - Open source, free to use and modify

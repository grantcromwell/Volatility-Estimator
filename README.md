# Volatility Estimator

A quantitative finance project implementing volatility estimation using:
- **Prim's Minimum Spanning Tree** for correlation analysis
- **Lorenz Attractor** for chaotic dynamics modeling
- **Hidden Markov Model** for regime detection

This project uses open-source Python libraries and free financial APIs to explore different approaches to volatility modeling.

## Features

- **Correlation Analysis**: Implements Prim's MST algorithm to identify relationships between assets
- **Chaotic Dynamics**: Applies Lorenz attractor to model non-linear volatility patterns
- **Regime Detection**: Uses HMM to identify hidden volatility states
- **API Integration**: Connects to Alpha Vantage, CoinGecko, and Yahoo Finance
- **Implementation**: Utilizes NumPy vectorization and Numba JIT compilation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd VolEstimator

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

```python
from vol_estimator import VolatilityEstimator

# Initialize estimator
estimator = VolatilityEstimator()

# Fetch data from free APIs
data = estimator.fetch_market_data(symbols=['AMD', 'SNDK', 'GOOGL'])

# Prepare returns
returns, symbols = estimator.prepare_returns(data)

# Estimate volatility
volatility = estimator.estimate_volatility(returns, symbols)
print(volatility)
```

## Detailed Usage

### Basic Volatility Estimation

```python
from vol_estimator import VolatilityEstimator

estimator = VolatilityEstimator()

# Fetch market data
symbols = ['AMD', 'SNDK', 'GOOGL', 'AMZN', 'ES']
data = estimator.fetch_market_data(symbols, period="1y")

# Prepare returns
returns, symbols = estimator.prepare_returns(data)

# Estimate volatility
volatility = estimator.estimate_volatility(returns, symbols)

for symbol, vol in volatility.items():
    print(f"{symbol}: {vol:.4f} ({vol*100:.2f}%)")
```

### With Regime Detection

```python
# Enable HMM for regime detection
estimator = VolatilityEstimator(hmm_enabled=True)

# ... fetch data and prepare returns ...

volatility = estimator.estimate_volatility(returns, symbols)

# Get regime information
regime_info = estimator.get_regime_info()
print(f"Current regime: {regime_info['current_state']}")
print(f"State volatilities: {regime_info['state_volatilities']}")
```

### With Correlation Analysis

```python
# Enable MST correlation analysis
estimator = VolatilityEstimator(mst_enabled=True)

# ... fetch data and prepare returns ...

volatility = estimator.estimate_volatility(returns, symbols)

# Get correlation structure
corr_info = estimator.get_correlation_structure()
print(f"Core assets: {corr_info['core_assets']}")
print(f"MST edges: {len(corr_info['mst_edges'])}")
```

### Portfolio Volatility

```python
estimator = VolatilityEstimator()

# ... fetch data and prepare returns ...

# Equal-weighted portfolio
portfolio_vol = estimator.estimate_portfolio_volatility(returns)
print(f"Portfolio volatility: {portfolio_vol:.4f}")

# Custom weights
weights = np.array([0.5, 0.3, 0.2])  # 50%, 30%, 20%
portfolio_vol_weighted = estimator.estimate_portfolio_volatility(
    returns, weights=weights
)
```

### Cryptocurrency Volatility

```python
# Crypto symbols (Yahoo Finance format)
crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD']

data = estimator.fetch_market_data(
    crypto_symbols, 
    period="1y", 
    source="yahoo"
)

returns, symbols = estimator.prepare_returns(data)
volatility = estimator.estimate_volatility(returns, symbols)
```

## Project Structure

```
VolEstimator/
├── vol_estimator/
│   ├── __init__.py
│   ├── correlation_mst.py      # Prim's MST correlation analysis
│   ├── lorenz_attractor.py     # Lorenz attractor dynamics
│   ├── hmm_regime.py           # Hidden Markov Model
│   ├── api_clients.py          # Free API integrations
│   └── estimator.py            # Main volatility estimator
├── tests/
│   ├── __init__.py
│   └── test_estimator.py       # Unit tests
├── example_usage.py            # Usage examples
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Components

### 1. Prim's Minimum Spanning Tree (MST)

Constructs a minimum spanning tree from the correlation matrix to analyze relationships between assets.

**Implementation:**
- Identifies core assets (central nodes in correlation network)
- Computes MST-weighted volatilities
- Filters correlation structure based on network topology

### 2. Lorenz Attractor

Applies the Lorenz system of differential equations to model volatility dynamics. Explores non-linear patterns and regime transitions.

**Implementation:**
- Simulates chaotic volatility dynamics
- Analyzes regime transitions
- Computes Lyapunov exponent as a chaos indicator

### 3. Hidden Markov Model (HMM)

Applies HMM to identify hidden volatility regimes (e.g., low, medium, high volatility) through probabilistic state modeling.

**Implementation:**
- Regime detection and classification
- State transition probability estimation
- Regime-conditional volatility calculation

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

### API Key Setup (Optional)

```python
# With Alpha Vantage API key
estimator = VolatilityEstimator(alpha_vantage_key="YOUR_API_KEY")
```

## CPU Optimization

The implementation is optimized for CPU performance through:

1. **NumPy Vectorization**: All computations use vectorized operations
2. **Numba JIT**: Critical functions compiled with Numba for speed
3. **Efficient Algorithms**: Uses optimized implementations from NetworkX, SciPy
4. **Memory Efficiency**: Minimizes data copying and intermediate arrays

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=vol_estimator --cov-report=html
```

## Examples

See `example_usage.py` for comprehensive examples:

```bash
python example_usage.py
```

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing (ODE solving)
- **pandas**: Data manipulation
- **networkx**: Graph algorithms (Prim's MST)
- **hmmlearn**: Hidden Markov Models
- **numba**: JIT compilation for performance
- **requests**: HTTP requests for APIs
- **yfinance**: Yahoo Finance API client

## Technical Notes

- **Vectorization**: Operations implemented using NumPy
- **JIT Compilation**: Correlation computation uses Numba JIT
- **Graph Operations**: NetworkX library for MST algorithms
- **Rate Limiting**: Basic rate limiting for API calls

## Limitations

- Free API tiers have rate limits
- Some APIs require API keys for higher rate limits
- HMM requires sufficient historical data (typically 100+ periods)
- MST analysis requires multiple assets (3+ recommended)
- This is a learning project and may not be suitable for production use

## Contributing

Feedback and contributions are welcome. Please ensure:
- Code follows PEP 8 style guidelines
- Tests pass (`pytest tests/`)
- New features include tests
- Documentation is updated


## References

- Prim's Algorithm: Minimum Spanning Tree for correlation networks
- Lorenz Attractor: Chaotic systems in financial modeling
- Hidden Markov Models: Regime-switching volatility models
- Free Financial APIs: Alpha Vantage, CoinGecko, Yahoo Finance


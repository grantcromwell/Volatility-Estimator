# Volatility Estimator - Project Summary

## Overview

A comprehensive volatility estimation system that integrates three advanced mathematical models:
1. **Prim's Minimum Spanning Tree** for correlation analysis
2. **Lorenz Attractor** for chaotic dynamics modeling
3. **Hidden Markov Model** for regime detection

Built entirely with open-source tools and free APIs, optimized for CPU performance.

## Architecture

### Core Components

1. **CorrelationMST** (`vol_estimator/correlation_mst.py`)
   - Implements Prim's algorithm for minimum spanning tree construction
   - Identifies core assets in correlation networks
   - Computes MST-weighted volatilities
   - CPU-optimized with NumPy vectorization and Numba JIT

2. **LorenzVolatilityModel** (`vol_estimator/lorenz_attractor.py`)
   - Simulates Lorenz attractor dynamics
   - Maps chaotic trajectories to volatility estimates
   - Detects regime transitions
   - Computes Lyapunov exponent for chaos measurement

3. **HMMRegimeDetector** (`vol_estimator/hmm_regime.py`)
   - Hidden Markov Model for volatility regime detection
   - Identifies hidden states (low/medium/high volatility)
   - Provides regime-adjusted volatility estimates
   - Tracks state transitions and probabilities

4. **FinancialDataFetcher** (`vol_estimator/api_clients.py`)
   - Unified interface for multiple free APIs
   - Yahoo Finance (via yfinance)
   - Alpha Vantage (stocks)
   - CoinGecko (cryptocurrencies)
   - Automatic rate limiting and error handling

5. **VolatilityEstimator** (`vol_estimator/estimator.py`)
   - Main integration class combining all components
   - Orchestrates data fetching, processing, and estimation
   - Provides portfolio-level volatility estimation
   - Exposes component-specific insights

## Key Features

### CPU Optimization
- **NumPy Vectorization**: All computations use vectorized operations
- **Numba JIT**: Critical functions compiled with `@jit` decorator
- **Efficient Algorithms**: Uses optimized implementations from NetworkX, SciPy
- **Memory Efficiency**: Minimizes data copying and intermediate arrays

### Free API Integration
- **Yahoo Finance**: No API key required, high reliability
- **Alpha Vantage**: Free tier (5 calls/min, 500/day)
- **CoinGecko**: Free tier (10-50 calls/min)
- **Automatic Fallback**: Tries multiple sources if one fails

### Mathematical Models

#### Prim's MST
- Converts correlation matrix to distance matrix
- Constructs minimum spanning tree
- Identifies central nodes (core assets)
- Computes weighted volatilities based on centrality

#### Lorenz Attractor
- Simulates chaotic system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
- Maps z-component to volatility
- Detects regime transitions via threshold crossing
- Computes Lyapunov exponent for chaos quantification

#### Hidden Markov Model
- Gaussian HMM with configurable states (default: 3)
- Identifies volatility regimes
- Provides state probabilities and transitions
- Regime-adjusted volatility estimation

## File Structure

```
VolEstimator/
├── vol_estimator/              # Main package
│   ├── __init__.py           # Package exports
│   ├── correlation_mst.py    # Prim's MST implementation
│   ├── lorenz_attractor.py   # Lorenz attractor model
│   ├── hmm_regime.py         # HMM regime detection
│   ├── api_clients.py        # Free API integrations
│   └── estimator.py          # Main estimator class
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_estimator.py     # Comprehensive test suite
├── example_usage.py          # Usage examples
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── README.md                 # Main documentation
├── QUICKSTART.md             # Quick start guide
└── PROJECT_SUMMARY.md        # This file
```

## Dependencies

### Core
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing (ODE solving)
- `pandas>=2.0.0` - Data manipulation

### Algorithms
- `networkx>=3.1` - Graph algorithms (Prim's MST)
- `hmmlearn>=0.3.0` - Hidden Markov Models

### Performance
- `numba>=0.57.0` - JIT compilation

### APIs
- `requests>=2.31.0` - HTTP requests
- `yfinance>=0.2.28` - Yahoo Finance client

### Optional
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical plots

## Usage Patterns

### Basic Usage
```python
from vol_estimator import VolatilityEstimator

estimator = VolatilityEstimator()
data = estimator.fetch_market_data(['AMD', 'SNDK', 'GOOGL'])
returns, symbols = estimator.prepare_returns(data)
volatility = estimator.estimate_volatility(returns, symbols)
```

### Component-Specific
```python
# MST only
estimator = VolatilityEstimator(mst_enabled=True, lorenz_enabled=False, hmm_enabled=False)

# HMM only
estimator = VolatilityEstimator(mst_enabled=False, lorenz_enabled=False, hmm_enabled=True)

# Full integration
estimator = VolatilityEstimator()  # All enabled by default
```

### Portfolio Volatility
```python
weights = np.array([0.5, 0.3, 0.2])
portfolio_vol = estimator.estimate_portfolio_volatility(returns, weights=weights)
```

## Testing

Comprehensive test suite covering:
- Correlation matrix computation
- MST construction
- Lorenz simulation
- HMM fitting and prediction
- Full integration pipeline
- Portfolio volatility estimation

Run tests:
```bash
pytest tests/ -v
```

## Performance Characteristics

### Computational Complexity
- **Correlation Matrix**: O(n²m) where n=assets, m=periods
- **MST Construction**: O(n² log n) using Prim's algorithm
- **Lorenz Simulation**: O(k) where k=simulation points
- **HMM Fitting**: O(nm²) where n=states, m=observations

### Optimization Strategies
1. Vectorized NumPy operations
2. Numba JIT for hot paths
3. Efficient data structures (NetworkX graphs)
4. Minimal data copying
5. Batch API calls

## API Rate Limits

### Yahoo Finance
- No official limits (practical: ~1000 calls/hour)
- Most reliable for general use

### Alpha Vantage
- Free tier: 5 calls/minute, 500 calls/day
- Requires API key for better limits

### CoinGecko
- Free tier: 10-50 calls/minute
- No API key required

## Limitations

1. **Data Requirements**
   - Minimum 100 periods recommended for HMM
   - At least 3 assets for meaningful MST analysis

2. **API Constraints**
   - Free tiers have rate limits
   - Some APIs may require keys for production use

3. **Computational**
   - HMM fitting can be slow for large datasets
   - Lorenz simulation adds overhead

4. **Model Assumptions**
   - Assumes stationarity for correlation
   - HMM assumes discrete states
   - Lorenz model is deterministic chaos

## Future Enhancements

Potential improvements:
1. GPU acceleration for large-scale computations
2. Additional free APIs (Twelve Data, Finnhub)
3. Real-time streaming data support
4. Advanced regime detection (multiple HMMs)
5. Ensemble methods combining multiple estimators
6. Visualization tools for correlation networks
7. Backtesting framework
8. Risk metrics beyond volatility

## License

MIT License - Open source, free to use and modify

## References

- Prim's Algorithm: Graph theory for correlation networks
- Lorenz Attractor: Chaotic systems in financial modeling
- Hidden Markov Models: Regime-switching volatility
- Free APIs: Alpha Vantage, CoinGecko, Yahoo Finance documentation

## Contact & Support

For issues, questions, or contributions:
- Check `README.md` for detailed documentation
- See `QUICKSTART.md` for quick examples
- Review `example_usage.py` for usage patterns
- Run `pytest tests/` to verify installation


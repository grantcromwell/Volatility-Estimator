#!/usr/bin/env python3
"""
Quick test script for volatility estimator
Tests basic functionality without requiring all dependencies
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Volatility Estimator - Quick Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ NumPy imported")
except ImportError as e:
    print(f"   ✗ NumPy not available: {e}")
    print("   Please install: pip install numpy")
    sys.exit(1)

try:
    from vol_estimator import VolatilityEstimator
    print("   ✓ VolatilityEstimator imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("   This is expected if dependencies are not installed")
    print("   Please install: pip install -r requirements.txt")
    sys.exit(1)

# Test initialization
print("\n2. Testing estimator initialization...")
try:
    estimator = VolatilityEstimator()
    print("   ✓ Estimator initialized successfully")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    sys.exit(1)

# Test with synthetic data
print("\n3. Testing with synthetic data...")
try:
    np.random.seed(42)
    returns = np.random.randn(5, 100) * 0.02
    symbols = ['AMD', 'SNDK', 'GOOGL', 'AMZN', 'ES']
    
    print(f"   Generated returns matrix: {returns.shape}")
    print(f"   Symbols: {symbols}")
    
    volatility = estimator.estimate_volatility(returns, symbols)
    
    print("   ✓ Volatility estimation successful!")
    print("\n   Volatility Results:")
    print("   " + "-" * 50)
    for symbol, vol in sorted(volatility.items(), key=lambda x: x[1], reverse=True):
        print(f"   {symbol:6s}: {vol:.4f} ({vol*100:6.2f}%)")
    
except Exception as e:
    print(f"   ✗ Estimation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test individual components
print("\n4. Testing individual components...")

# Test MST
try:
    from vol_estimator import CorrelationMST
    mst = CorrelationMST()
    mst.compute_correlation_matrix(returns)
    mst_graph = mst.build_mst()
    print(f"   ✓ MST: {mst_graph.number_of_nodes()} nodes, {mst_graph.number_of_edges()} edges")
except Exception as e:
    print(f"   ⚠ MST test: {e}")

# Test Lorenz
try:
    from vol_estimator import LorenzVolatilityModel
    lorenz = LorenzVolatilityModel()
    _, trajectory = lorenz.simulate(t_span=(0, 10), n_points=100)
    vol = lorenz.map_to_volatility(trajectory)
    print(f"   ✓ Lorenz: Trajectory shape {trajectory.shape}, volatility range [{vol.min():.4f}, {vol.max():.4f}]")
except Exception as e:
    print(f"   ⚠ Lorenz test: {e}")

# Test HMM
try:
    from vol_estimator import HMMRegimeDetector
    hmm = HMMRegimeDetector(n_states=3)
    avg_returns = np.mean(returns, axis=0)
    hmm.fit(avg_returns)
    states = hmm.predict_states(avg_returns)
    print(f"   ✓ HMM: Fitted with {hmm.n_states} states, predicted {len(states)} states")
except Exception as e:
    print(f"   ⚠ HMM test: {e}")

# Test portfolio volatility
print("\n5. Testing portfolio volatility...")
try:
    portfolio_vol = estimator.estimate_portfolio_volatility(returns)
    print(f"   ✓ Portfolio volatility: {portfolio_vol:.4f} ({portfolio_vol*100:.2f}%)")
except Exception as e:
    print(f"   ⚠ Portfolio test: {e}")

print("\n" + "=" * 60)
print("✓ All tests completed!")
print("=" * 60)


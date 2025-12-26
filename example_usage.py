"""
Example Usage of Volatility Estimator

Demonstrates how to use the volatility estimator with real market data
from free APIs.
"""

import numpy as np
from vol_estimator import VolatilityEstimator


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Basic Volatility Estimation Example")
    print("=" * 60)
    
    # Initialize estimator
    estimator = VolatilityEstimator()
    
    # Fetch market data (using free APIs)
    symbols = ['AMD', 'SNDK', 'GOOGL']
    print(f"\nFetching data for: {symbols}")
    
    data = estimator.fetch_market_data(symbols, period="1y")
    print(f"Fetched data for {len(data)} symbols")
    
    # Prepare returns
    returns, symbols = estimator.prepare_returns(data)
    print(f"Prepared returns matrix: {returns.shape}")
    
    # Estimate volatility
    volatility = estimator.estimate_volatility(returns, symbols)
    
    print("\nVolatility Estimates:")
    for symbol, vol in volatility.items():
        print(f"  {symbol}: {vol:.4f} ({vol*100:.2f}%)")
    
    return estimator, volatility


def example_with_regime_detection():
    """Example with HMM regime detection"""
    print("\n" + "=" * 60)
    print("Volatility Estimation with Regime Detection")
    print("=" * 60)
    
    estimator = VolatilityEstimator(hmm_enabled=True)
    
    symbols = ['AMD', 'SNDK', 'GOOGL', 'AMZN', 'ES']
    data = estimator.fetch_market_data(symbols, period="2y")
    returns, symbols = estimator.prepare_returns(data)
    
    volatility = estimator.estimate_volatility(returns, symbols)
    
    # Get regime information
    regime_info = estimator.get_regime_info()
    if regime_info:
        print("\nRegime Information:")
        print(f"  Current State: {regime_info['current_state']}")
        print(f"  State Volatilities: {regime_info['state_volatilities']}")
        print(f"  Number of Regime Changes: {len(regime_info['regime_changes'])}")
    
    return estimator, volatility, regime_info


def example_with_mst_correlation():
    """Example with MST correlation analysis"""
    print("\n" + "=" * 60)
    print("Volatility Estimation with MST Correlation Analysis")
    print("=" * 60)
    
    estimator = VolatilityEstimator(mst_enabled=True)
    
    symbols = ['AMD', 'SNDK', 'GOOGL', 'AMZN', 'ES', 'NVDA', 'META']
    data = estimator.fetch_market_data(symbols, period="1y")
    returns, symbols = estimator.prepare_returns(data)
    
    volatility = estimator.estimate_volatility(returns, symbols)
    
    # Get correlation structure
    corr_structure = estimator.get_correlation_structure()
    if corr_structure:
        print("\nCorrelation Structure (MST):")
        print(f"  Core Assets: {corr_structure['core_assets'][:5]}")
        print(f"  MST Edges: {len(corr_structure['mst_edges'])}")
        print("\n  Top MST Edges (by correlation):")
        edges_sorted = sorted(
            corr_structure['mst_edges'], 
            key=lambda x: abs(x[2]), 
            reverse=True
        )
        for asset1, asset2, corr in edges_sorted[:5]:
            print(f"    {asset1} - {asset2}: {corr:.3f}")
    
    return estimator, volatility, corr_structure


def example_portfolio_volatility():
    """Example of portfolio volatility estimation"""
    print("\n" + "=" * 60)
    print("Portfolio Volatility Estimation")
    print("=" * 60)
    
    estimator = VolatilityEstimator()
    
    symbols = ['AMD', 'SNDK', 'GOOGL']
    data = estimator.fetch_market_data(symbols, period="1y")
    returns, symbols = estimator.prepare_returns(data)
    
    # Equal-weighted portfolio
    portfolio_vol = estimator.estimate_portfolio_volatility(returns)
    print(f"\nEqual-Weighted Portfolio Volatility: {portfolio_vol:.4f} ({portfolio_vol*100:.2f}%)")
    
    # Custom weights
    weights = np.array([0.5, 0.3, 0.2])  # 50% AMD, 30% SNDK, 20% GOOGL
    portfolio_vol_weighted = estimator.estimate_portfolio_volatility(returns, weights=weights)
    print(f"Weighted Portfolio Volatility: {portfolio_vol_weighted:.4f} ({portfolio_vol_weighted*100:.2f}%)")
    
    return portfolio_vol, portfolio_vol_weighted


def example_crypto_volatility():
    """Example with cryptocurrency data"""
    print("\n" + "=" * 60)
    print("Cryptocurrency Volatility Estimation")
    print("=" * 60)
    
    estimator = VolatilityEstimator()
    
    # Crypto symbols (using Yahoo Finance format)
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD']
    print(f"\nFetching crypto data for: {crypto_symbols}")
    
    data = estimator.fetch_market_data(crypto_symbols, period="1y", source="yahoo")
    returns, symbols = estimator.prepare_returns(data)
    
    volatility = estimator.estimate_volatility(returns, symbols)
    
    print("\nCryptocurrency Volatility Estimates:")
    for symbol, vol in volatility.items():
        print(f"  {symbol}: {vol:.4f} ({vol*100:.2f}%)")
    
    return estimator, volatility


def example_full_integration():
    """Full integration example with all components"""
    print("\n" + "=" * 60)
    print("Full Integration Example (MST + Lorenz + HMM)")
    print("=" * 60)
    
    estimator = VolatilityEstimator(
        mst_enabled=True,
        lorenz_enabled=True,
        hmm_enabled=True
    )
    
    symbols = ['AMD', 'SNDK', 'GOOGL', 'AMZN', 'ES']
    data = estimator.fetch_market_data(symbols, period="2y")
    returns, symbols = estimator.prepare_returns(data)
    
    print(f"\nData shape: {returns.shape}")
    print(f"Symbols: {symbols}")
    
    # Estimate volatility
    volatility = estimator.estimate_volatility(returns, symbols)
    
    print("\nFinal Volatility Estimates:")
    for symbol, vol in sorted(volatility.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {vol:.4f} ({vol*100:.2f}%)")
    
    # Get all component information
    print("\n" + "-" * 60)
    print("Component Analysis:")
    print("-" * 60)
    
    # MST info
    corr_info = estimator.get_correlation_structure()
    if corr_info:
        print(f"\nMST Analysis:")
        print(f"  Core assets: {corr_info['core_assets'][:3]}")
    
    # HMM info
    regime_info = estimator.get_regime_info()
    if regime_info:
        print(f"\nHMM Regime Detection:")
        print(f"  Current state: {regime_info['current_state']}")
        print(f"  State volatilities: {[f'{v:.4f}' for v in regime_info['state_volatilities']]}")
    
    # Lorenz info
    lorenz_info = estimator.get_lorenz_info()
    if lorenz_info:
        print(f"\nLorenz Attractor:")
        print(f"  Lyapunov exponent: {lorenz_info['lyapunov_exponent']:.4f}")
        print(f"  Regime transitions: {lorenz_info['regime_transitions']}")
    
    return estimator, volatility


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Volatility Estimator - Example Usage")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_with_regime_detection()
        example_with_mst_correlation()
        example_portfolio_volatility()
        example_crypto_volatility()
        example_full_integration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


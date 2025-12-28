"""
Comprehensive tests for volatility models.

Tests include:
- EWMA volatility estimator
- GARCH models
- Range-based estimators
- Validation framework
- Accuracy metrics
"""

import numpy as np
import pytest
from vol_estimator import (
    EWMAVolatilityEstimator,
    MultiAssetEWMAEstimator,
    GARCHVolatilityEstimator,
    ParkinsonVolatilityEstimator,
    GarmanKlassVolatilityEstimator,
    YangZhangVolatilityEstimator,
    compare_estimators,
    ValidationMetrics,
    VolatilityBacktester,
    ModelValidator,
    calculate_accuracy_metrics,
    compute_realized_volatility,
    VolatilityEstimator,
    get_logger
)


class TestEWMAVolatility:
    """Test EWMA volatility estimator."""
    
    def test_ewma_initialization(self):
        """Test EWMA initialization."""
        ewma = EWMAVolatilityEstimator(decay_factor=0.94)
        assert ewma.decay_factor == 0.94
        assert ewma.min_periods == 10
    
    def test_ewma_fitting(self):
        """Test EWMA fitting."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        
        ewma = EWMAVolatilityEstimator(decay_factor=0.94)
        ewma.fit(returns)
        
        assert ewma.volatilities_ is not None
        assert len(ewma.volatilities_) == len(returns)
        assert np.all(ewma.volatilities_ > 0)
    
    def test_ewma_current_volatility(self):
        """Test getting current EWMA volatility."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        
        ewma = EWMAVolatilityEstimator()
        ewma.fit(returns)
        
        current_vol = ewma.get_current_volatility()
        assert isinstance(current_vol, (float, np.floating))
        assert current_vol > 0
    
    def test_ewma_forecast(self):
        """Test EWMA volatility forecasting."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        
        ewma = EWMAVolatilityEstimator()
        ewma.fit(returns)
        
        # Single period forecast
        forecast_1 = ewma.forecast(horizon=1)
        assert isinstance(forecast_1, (float, np.floating))
        assert forecast_1 > 0
        
        # Multi-period forecast
        forecast_5 = ewma.forecast(horizon=5)
        assert len(forecast_5) == 5
        assert np.all(forecast_5 > 0)
    
    def test_ewma_update(self):
        """Test EWMA online update."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        
        ewma = EWMAVolatilityEstimator()
        ewma.fit(returns)
        
        initial_vol = ewma.get_current_volatility()
        new_return = 0.03
        updated_vol = ewma.update(new_return)
        
        assert updated_vol != initial_vol
        assert updated_vol > 0
    
    def test_multi_asset_ewma(self):
        """Test multi-asset EWMA estimator."""
        np.random.seed(42)
        n_assets, n_periods = 5, 200
        returns = np.random.randn(n_assets, n_periods) * 0.02
        
        multi_ewma = MultiAssetEWMAEstimator(decay_factor=0.94)
        multi_ewma.fit(returns)
        
        current_vols = multi_ewma.get_current_volatilities()
        assert len(current_vols) == n_assets
        assert np.all(current_vols > 0)


class TestGARCHVolatility:
    """Test GARCH volatility models."""
    
    @pytest.mark.skipif(
        not hasattr(pytest, 'arch_available'),
        reason="arch library not available"
    )
    def test_garch_initialization(self):
        """Test GARCH initialization."""
        try:
            garch = GARCHVolatilityEstimator(p=1, q=1)
            assert garch.p == 1
            assert garch.q == 1
        except ImportError:
            pytest.skip("arch library not available")
    
    @pytest.mark.skipif(
        not hasattr(pytest, 'arch_available'),
        reason="arch library not available"
    )
    def test_garch_fitting(self):
        """Test GARCH fitting."""
        try:
            np.random.seed(42)
            returns = np.random.randn(252) * 0.02
            
            garch = GARCHVolatilityEstimator(p=1, q=1)
            garch.fit(returns, disp='off')
            
            assert garch.conditional_volatility_ is not None
            assert len(garch.conditional_volatility_) == len(returns)
            assert np.all(garch.conditional_volatility_ > 0)
        except ImportError:
            pytest.skip("arch library not available")


class TestRangeEstimators:
    """Test range-based volatility estimators."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        self.high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
        self.low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
        self.close = close
        self.open = np.roll(close, 1)
        self.open[0] = close[0]
    
    def test_parkinson_estimator(self):
        """Test Parkinson volatility estimator."""
        pk = ParkinsonVolatilityEstimator()
        pk.fit(self.high, self.low)
        
        vol = pk.get_volatility()
        assert isinstance(vol, (float, np.floating))
        assert vol > 0
    
    def test_garman_klass_estimator(self):
        """Test Garman-Klass volatility estimator."""
        gk = GarmanKlassVolatilityEstimator()
        gk.fit(self.high, self.low, self.close, self.open)
        
        vol = gk.get_volatility()
        assert isinstance(vol, (float, np.floating))
        assert vol > 0
    
    def test_yang_zhang_estimator(self):
        """Test Yang-Zhang volatility estimator."""
        yz = YangZhangVolatilityEstimator()
        yz.fit(self.high, self.low, self.close, self.open)
        
        vol = yz.get_volatility()
        assert isinstance(vol, (float, np.floating))
        assert vol > 0
    
    def test_rolling_parkinson(self):
        """Test rolling Parkinson estimator."""
        pk = ParkinsonVolatilityEstimator(window=20)
        pk.fit(self.high, self.low)
        
        vol_series = pk.get_volatility_series()
        assert vol_series is not None
        assert len(vol_series) == len(self.high) - 20 + 1
        assert np.all(vol_series > 0)
    
    def test_compare_estimators(self):
        """Test comparison of all range estimators."""
        results = compare_estimators(
            self.high, self.low, self.close, self.open
        )
        
        assert isinstance(results, dict)
        assert 'Parkinson' in results
        assert 'Garman-Klass' in results
        assert 'Yang-Zhang' in results
        assert all(v > 0 for v in results.values())


class TestValidationFramework:
    """Test validation and backtesting framework."""
    
    def test_calculate_accuracy_metrics(self):
        """Test accuracy metrics calculation."""
        np.random.seed(42)
        actuals = np.random.rand(100) * 0.02 + 0.01
        predictions = actuals + np.random.randn(100) * 0.002
        
        metrics = calculate_accuracy_metrics(predictions, actuals)
        
        assert isinstance(metrics, ValidationMetrics)
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape >= 0
        assert metrics.n_samples == 100
    
    def test_compute_realized_volatility(self):
        """Test realized volatility computation."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        
        realized_vol = compute_realized_volatility(returns, window=20)
        
        assert len(realized_vol) == len(returns)
        assert np.all(realized_vol >= 0)
    
    def test_backtester_initialization(self):
        """Test backtester initialization."""
        backtester = VolatilityBacktester(
            train_size=100,
            test_size=20,
            step_size=10
        )
        
        assert backtester.train_size == 100
        assert backtester.test_size == 20
        assert backtester.step_size == 10
    
    def test_model_validator(self):
        """Test model validator."""
        np.random.seed(42)
        actuals = np.random.rand(100) * 0.02 + 0.01
        predictions = actuals + np.random.randn(100) * 0.002
        
        validator = ModelValidator()
        results = validator.validate_model(predictions, actuals, "TestModel")
        
        assert 'model_name' in results
        assert 'basic_metrics' in results
        assert 'checks' in results
        assert 'overall_passed' in results


class TestEnterpriseVolatilityEstimator:
    """Test volatility estimator."""
    
    def test_enterprise_initialization(self):
        """Test estimator with enterprise features enabled."""
        estimator = VolatilityEstimator(
            mst_enabled=True,
            lorenz_enabled=True,
            hmm_enabled=True,
            ewma_enabled=True,
            garch_enabled=False,  # May not have arch library
            range_estimator_enabled=False
        )
        
        assert estimator.ewma_enabled
        assert estimator.ewma is not None
    
    def test_ewma_estimation(self):
        """Test EWMA estimation in main estimator."""
        np.random.seed(42)
        n_assets, n_periods = 3, 200
        returns = np.random.randn(n_assets, n_periods) * 0.02
        
        estimator = VolatilityEstimator(
            mst_enabled=False,
            lorenz_enabled=False,
            hmm_enabled=False,
            ewma_enabled=True
        )
        
        vols = estimator.estimate_with_ewma(returns)
        
        assert len(vols) == n_assets
        assert np.all(vols > 0)
    
    def test_ensemble_estimation(self):
        """Test ensemble volatility estimation."""
        np.random.seed(42)
        n_assets, n_periods = 3, 200
        returns = np.random.randn(n_assets, n_periods) * 0.02
        symbols = ['A', 'B', 'C']
        
        estimator = VolatilityEstimator(
            mst_enabled=True,
            lorenz_enabled=True,
            hmm_enabled=True,
            ewma_enabled=True,
            garch_enabled=False
        )
        
        result = estimator.estimate_ensemble(returns, symbols)
        
        assert len(result) == n_assets
        assert all(isinstance(v, (float, np.floating)) for v in result.values())
        assert all(v > 0 for v in result.values())
    
    def test_validation_predictions(self):
        """Test validation of predictions."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        predictions = np.full(200, 0.02)
        
        estimator = VolatilityEstimator(ewma_enabled=True)
        
        validation_result = estimator.validate_predictions(
            predictions, returns, realized_vol_window=20
        )
        
        assert 'metrics' in validation_result
        assert 'predictions' in validation_result
        assert 'realized' in validation_result
        assert isinstance(validation_result['metrics'], ValidationMetrics)
    
    def test_benchmark_comparison(self):
        """Test benchmarking against simple model."""
        np.random.seed(42)
        n_assets, n_periods = 3, 200
        returns = np.random.randn(n_assets, n_periods) * 0.02
        
        estimator = VolatilityEstimator(
            mst_enabled=True,
            ewma_enabled=True,
            hmm_enabled=True
        )
        
        comparison = estimator.compare_with_benchmark(
            returns, benchmark_model='simple_std'
        )
        
        assert 'mae_improvement_%' in comparison
        assert 'rmse_improvement_%' in comparison
        assert 'main_model_metrics' in comparison
        assert 'benchmark_metrics' in comparison


class TestIntegration:
    """Integration tests for enterprise system."""
    
    def test_full_pipeline_with_validation(self):
        """Test complete pipeline with all enterprise features."""
        np.random.seed(42)
        n_assets, n_periods = 5, 252
        returns = np.random.randn(n_assets, n_periods) * 0.02
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        # Initialize with all features
        estimator = VolatilityEstimator(
            mst_enabled=True,
            lorenz_enabled=True,
            hmm_enabled=True,
            ewma_enabled=True,
            garch_enabled=False  # Requires arch library
        )
        
        # Estimate volatility
        volatility = estimator.estimate_ensemble(returns, symbols)
        
        # Validate
        assert len(volatility) == n_assets
        assert all(vol > 0 for vol in volatility.values())
        
        # Test component access
        assert estimator.ewma is not None
        assert estimator.mst is not None
        assert estimator.hmm is not None
    
    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        logger = get_logger("vol_estimator")
        
        assert logger is not None
        assert "vol_estimator" in logger.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


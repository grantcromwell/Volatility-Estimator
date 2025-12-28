"""
Validation and Backtesting Framework for Volatility Models.

Provides validation with:
- Accuracy metrics (MAE, RMSE, MAPE)
- Backtesting against realized volatility
- Model comparison and benchmarking
- Performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from .logging_config import get_logger, MetricsLogger

logger = get_logger(__name__)
metrics_logger = MetricsLogger()


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r_squared: float  # R-squared
    n_samples: int  # Number of samples
    
    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.6f}, RMSE={self.rmse:.6f}, "
            f"MAPE={self.mape:.2f}%, R²={self.r_squared:.4f}, n={self.n_samples}"
        )


@dataclass
class BacktestResult:
    """Container for backtest results."""
    model_name: str
    metrics: ValidationMetrics
    predictions: np.ndarray
    actuals: np.ndarray
    timestamps: Optional[List] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            'model': self.model_name,
            'mae': self.metrics.mae,
            'rmse': self.metrics.rmse,
            'mape': self.metrics.mape,
            'r_squared': self.metrics.r_squared,
            'n_samples': self.metrics.n_samples
        }


def compute_realized_volatility(
    returns: np.ndarray,
    window: int = 20,
    method: str = 'std'
) -> np.ndarray:
    """
    Compute realized volatility from returns.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        method: Method for volatility ('std', 'squared')
    
    Returns:
        Array of realized volatility
    """
    if method == 'std':
        # Standard deviation method
        realized_vol = np.array([
            np.std(returns[max(0, i-window):i+1])
            for i in range(len(returns))
        ])
    elif method == 'squared':
        # Sum of squared returns
        realized_vol = np.array([
            np.sqrt(np.sum(returns[max(0, i-window):i+1]**2))
            for i in range(len(returns))
        ])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return realized_vol


def calculate_accuracy_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    epsilon: float = 1e-10
) -> ValidationMetrics:
    """
    Calculate accuracy metrics for volatility predictions.
    
    Args:
        predictions: Predicted volatility values
        actuals: Actual/realized volatility values
        epsilon: Small value to avoid division by zero
    
    Returns:
        ValidationMetrics object with calculated metrics
    """
    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[-min_len:]
    actuals = actuals[-min_len:]
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - actuals))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
    
    # R-squared
    ss_res = np.sum((actuals - predictions)**2)
    ss_tot = np.sum((actuals - np.mean(actuals))**2)
    r_squared = 1 - (ss_res / (ss_tot + epsilon))
    
    metrics = ValidationMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        r_squared=r_squared,
        n_samples=len(predictions)
    )
    
    logger.debug(f"Accuracy metrics calculated | {metrics}")
    
    return metrics


class VolatilityBacktester:
    """
    Backtesting framework for volatility models.
    
    Validates model predictions against realized volatility using
    walk-forward analysis and generates comprehensive performance metrics.
    """
    
    def __init__(
        self,
        train_size: int = 252,  # ~1 year of daily data
        test_size: int = 60,    # ~3 months
        step_size: int = 20     # Re-fit every 20 days
    ):
        """
        Initialize backtester.
        
        Args:
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for walk-forward (re-fit frequency)
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        
        logger.info(
            f"Backtester initialized | train={train_size} | "
            f"test={test_size} | step={step_size}"
        )
    
    def backtest_model(
        self,
        model_func: Callable,
        returns: np.ndarray,
        model_name: str = "Model",
        realized_vol_window: int = 20
    ) -> BacktestResult:
        """
        Backtest a volatility model.
        
        Args:
            model_func: Function that takes returns and returns volatility estimate.
                       Signature: func(returns: np.ndarray) -> float
            returns: Historical returns
            model_name: Name of the model for logging
            realized_vol_window: Window for realized volatility calculation
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest | model={model_name} | n_periods={len(returns)}")
        
        n = len(returns)
        predictions = []
        actuals = []
        
        # Walk-forward validation
        for i in range(self.train_size, n - self.test_size, self.step_size):
            # Training data
            train_returns = returns[i-self.train_size:i]
            
            # Generate prediction
            try:
                pred_vol = model_func(train_returns)
                
                # Test period (compute realized volatility)
                test_returns = returns[i:i+self.test_size]
                realized_vol = np.std(test_returns)
                
                predictions.append(pred_vol)
                actuals.append(realized_vol)
            
            except Exception as e:
                logger.warning(
                    f"Backtest step failed | model={model_name} | "
                    f"step={i} | error={e}"
                )
                continue
        
        if not predictions:
            raise ValueError(f"Backtest failed for {model_name}: no valid predictions")
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        metrics = calculate_accuracy_metrics(predictions, actuals)
        
        # Log results
        metrics_logger.log_accuracy(
            model_name=model_name,
            mae=metrics.mae,
            rmse=metrics.rmse,
            mape=metrics.mape,
            n_samples=metrics.n_samples
        )
        
        result = BacktestResult(
            model_name=model_name,
            metrics=metrics,
            predictions=predictions,
            actuals=actuals
        )
        
        logger.info(f"Backtest complete | model={model_name} | {metrics}")
        
        return result
    
    def compare_models(
        self,
        models: Dict[str, Callable],
        returns: np.ndarray
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple volatility models.
        
        Args:
            models: Dictionary of {model_name: model_func}
            returns: Historical returns
        
        Returns:
            Dictionary of {model_name: BacktestResult}
        """
        results = {}
        
        for model_name, model_func in models.items():
            try:
                result = self.backtest_model(
                    model_func=model_func,
                    returns=returns,
                    model_name=model_name
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"Model comparison failed | model={model_name} | error={e}")
        
        # Log comparison
        if results:
            best_model = min(results.items(), key=lambda x: x[1].metrics.rmse)
            logger.info(
                f"Model comparison complete | "
                f"best_model={best_model[0]} | "
                f"best_rmse={best_model[1].metrics.rmse:.6f}"
            )
        
        return results


class ModelValidator:
    """
    Comprehensive model validation framework.
    
    Validates volatility models against multiple criteria:
    - Accuracy against realized volatility
    - Stability across time periods
    - Performance in different market regimes
    - Robustness to outliers
    """
    
    def __init__(self):
        self.validation_history = []
    
    def validate_model(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Validate a volatility model.
        
        Args:
            predictions: Model predictions
            actuals: Actual/realized volatility
            model_name: Name of the model
        
        Returns:
            Dictionary with validation results
        """
        # Basic accuracy metrics
        metrics = calculate_accuracy_metrics(predictions, actuals)
        
        # Additional validation checks
        validation_results = {
            'model_name': model_name,
            'basic_metrics': metrics,
            'checks': {}
        }
        
        # Check 1: Bias test (mean error)
        bias = np.mean(predictions - actuals)
        validation_results['checks']['bias'] = {
            'value': bias,
            'passed': abs(bias) < 0.01  # Less than 1% bias
        }
        
        # Check 2: Stability test (variance of errors)
        error_std = np.std(predictions - actuals)
        validation_results['checks']['stability'] = {
            'error_std': error_std,
            'passed': error_std < np.std(actuals)  # Lower variance than actuals
        }
        
        # Check 3: Outlier resistance (robust to extreme values)
        errors = np.abs(predictions - actuals)
        extreme_errors = np.sum(errors > 3 * np.std(errors))
        validation_results['checks']['outlier_resistance'] = {
            'extreme_errors': int(extreme_errors),
            'percentage': float(extreme_errors / len(errors) * 100),
            'passed': extreme_errors / len(errors) < 0.05  # Less than 5%
        }
        
        # Check 4: Directional accuracy (captures increases/decreases)
        if len(actuals) > 1:
            actual_changes = np.diff(actuals) > 0
            pred_changes = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_changes == pred_changes)
            validation_results['checks']['directional_accuracy'] = {
                'value': float(directional_accuracy),
                'passed': directional_accuracy > 0.55  # Better than random
            }
        
        # Overall validation status
        all_checks = [
            check['passed'] for check in validation_results['checks'].values()
            if 'passed' in check
        ]
        validation_results['overall_passed'] = all(all_checks) if all_checks else False
        
        # Log validation
        metrics_logger.log_validation(
            model_name=model_name,
            validation_type="comprehensive",
            passed=validation_results['overall_passed'],
            details=f"Checks: {sum(all_checks)}/{len(all_checks)} passed"
        )
        
        self.validation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': model_name,
            'results': validation_results
        })
        
        logger.info(
            f"Validation complete | model={model_name} | "
            f"passed={validation_results['overall_passed']} | {metrics}"
        )
        
        return validation_results
    
    def get_validation_report(self) -> pd.DataFrame:
        """
        Generate validation report as DataFrame.
        
        Returns:
            DataFrame with validation history
        """
        if not self.validation_history:
            return pd.DataFrame()
        
        records = []
        for entry in self.validation_history:
            record = {
                'timestamp': entry['timestamp'],
                'model': entry['model_name'],
                'passed': entry['results']['overall_passed'],
                'mae': entry['results']['basic_metrics'].mae,
                'rmse': entry['results']['basic_metrics'].rmse,
                'mape': entry['results']['basic_metrics'].mape,
                'r_squared': entry['results']['basic_metrics'].r_squared
            }
            
            # Add check results
            for check_name, check_data in entry['results']['checks'].items():
                if 'passed' in check_data:
                    record[f'{check_name}_passed'] = check_data['passed']
            
            records.append(record)
        
        return pd.DataFrame(records)


def rolling_validation(
    predictions: np.ndarray,
    actuals: np.ndarray,
    window: int = 60
) -> List[ValidationMetrics]:
    """
    Perform rolling validation to assess model stability over time.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        window: Rolling window size
    
    Returns:
        List of ValidationMetrics for each window
    """
    results = []
    
    for i in range(window, len(predictions)):
        pred_window = predictions[i-window:i]
        actual_window = actuals[i-window:i]
        
        metrics = calculate_accuracy_metrics(pred_window, actual_window)
        results.append(metrics)
    
    logger.debug(f"Rolling validation complete | n_windows={len(results)}")
    
    return results


def benchmark_vs_naive(
    model_predictions: np.ndarray,
    actuals: np.ndarray,
    naive_predictions: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Benchmark model against naive forecast (e.g., last observed value).
    
    Args:
        model_predictions: Model predictions
        actuals: Actual values
        naive_predictions: Optional naive predictions. If None, uses last observed value.
    
    Returns:
        Dictionary with improvement metrics
    """
    if naive_predictions is None:
        # Naive forecast: use last observed value
        naive_predictions = np.concatenate([[actuals[0]], actuals[:-1]])
    
    model_metrics = calculate_accuracy_metrics(model_predictions, actuals)
    naive_metrics = calculate_accuracy_metrics(naive_predictions, actuals)
    
    improvement = {
        'mae_improvement': (naive_metrics.mae - model_metrics.mae) / naive_metrics.mae * 100,
        'rmse_improvement': (naive_metrics.rmse - model_metrics.rmse) / naive_metrics.rmse * 100,
        'model_mae': model_metrics.mae,
        'naive_mae': naive_metrics.mae,
        'model_rmse': model_metrics.rmse,
        'naive_rmse': naive_metrics.rmse
    }
    
    logger.info(
        f"Benchmark vs naive | MAE improvement: {improvement['mae_improvement']:.2f}% | "
        f"RMSE improvement: {improvement['rmse_improvement']:.2f}%"
    )
    
    return improvement


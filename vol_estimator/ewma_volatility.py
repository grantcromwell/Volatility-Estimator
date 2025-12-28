"""
Exponentially Weighted Moving Average (EWMA) Volatility Estimator.

Industry-standard volatility model that assigns exponentially decreasing weights
to historical returns, making it responsive to recent market changes while
capturing volatility clustering effects.
"""

import numpy as np
from typing import Optional, Union
from numba import jit
from .logging_config import get_logger

logger = get_logger(__name__)


@jit(nopython=True, cache=True)
def _ewma_volatility_numba(returns: np.ndarray, decay_factor: float, initial_vol: float) -> np.ndarray:
    """
    Compute EWMA volatility using Numba for speed optimization.
    
    Args:
        returns: 1D array of returns
        decay_factor: Decay factor (lambda), typically 0.94-0.97
        initial_vol: Initial volatility estimate
    
    Returns:
        Array of EWMA volatility estimates
    """
    n = len(returns)
    volatilities = np.zeros(n)
    volatilities[0] = initial_vol
    
    for i in range(1, n):
        # EWMA formula: σ²(t) = λ * σ²(t-1) + (1 - λ) * r²(t)
        volatilities[i] = np.sqrt(
            decay_factor * volatilities[i-1]**2 + 
            (1 - decay_factor) * returns[i]**2
        )
    
    return volatilities


class EWMAVolatilityEstimator:
    """
    Exponentially Weighted Moving Average (EWMA) volatility estimator.
    
    This is an industry-standard model widely used by financial institutions
    and is the basis for RiskMetrics methodology. EWMA assigns exponentially
    decreasing weights to historical returns, making it responsive to recent
    market changes.
    
    Key properties:
    - Captures volatility clustering
    - No lookback window required
    - Simple and computationally efficient
    - Used by major financial institutions (RiskMetrics)
    """
    
    def __init__(
        self,
        decay_factor: float = 0.94,
        initial_vol_method: str = 'simple_std',
        min_periods: int = 10
    ):
        """
        Initialize EWMA volatility estimator.
        
        Args:
            decay_factor: Decay factor (lambda), typically 0.94-0.97.
                        Higher values give more weight to historical data.
                        RiskMetrics recommends 0.94 for daily data.
            initial_vol_method: Method for initial volatility estimate:
                              - 'simple_std': Standard deviation of first min_periods
                              - 'first_return': Absolute value of first return
            min_periods: Minimum number of periods for initial volatility
        """
        if not 0 < decay_factor < 1:
            raise ValueError("decay_factor must be between 0 and 1")
        
        self.decay_factor = decay_factor
        self.initial_vol_method = initial_vol_method
        self.min_periods = min_periods
        
        self.volatilities_ = None
        self.current_volatility_ = None
        
        logger.info(
            f"EWMA initialized | decay_factor={decay_factor} | "
            f"initial_method={initial_vol_method} | min_periods={min_periods}"
        )
    
    def _compute_initial_volatility(self, returns: np.ndarray) -> float:
        """
        Compute initial volatility estimate.
        
        Args:
            returns: Array of returns
        
        Returns:
            Initial volatility estimate
        """
        if self.initial_vol_method == 'simple_std':
            if len(returns) >= self.min_periods:
                return np.std(returns[:self.min_periods])
            else:
                return np.std(returns)
        elif self.initial_vol_method == 'first_return':
            return abs(returns[0]) if len(returns) > 0 else 0.01
        else:
            raise ValueError(f"Unknown initial_vol_method: {self.initial_vol_method}")
    
    def fit(self, returns: Union[np.ndarray, list]) -> 'EWMAVolatilityEstimator':
        """
        Fit EWMA volatility model to returns.
        
        Args:
            returns: Array of returns (1D)
        
        Returns:
            Self for method chaining
        """
        returns = np.asarray(returns)
        
        if returns.ndim != 1:
            raise ValueError("Returns must be 1D array")
        
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns to estimate volatility")
        
        # Compute initial volatility
        initial_vol = self._compute_initial_volatility(returns)
        
        # Compute EWMA volatilities using optimized Numba function
        self.volatilities_ = _ewma_volatility_numba(
            returns, 
            self.decay_factor, 
            initial_vol
        )
        
        self.current_volatility_ = self.volatilities_[-1]
        
        logger.debug(
            f"EWMA fitted | n_periods={len(returns)} | "
            f"current_vol={self.current_volatility_:.6f}"
        )
        
        return self
    
    def forecast(self, horizon: int = 1) -> Union[float, np.ndarray]:
        """
        Forecast volatility for future periods.
        
        Under EWMA, the volatility forecast for any horizon h > 0 is simply
        the current volatility estimate (constant forecast).
        
        Args:
            horizon: Number of periods ahead to forecast
        
        Returns:
            Volatility forecast (scalar if horizon=1, array otherwise)
        """
        if self.current_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if horizon == 1:
            return self.current_volatility_
        else:
            # EWMA produces constant forecasts
            return np.full(horizon, self.current_volatility_)
    
    def update(self, new_return: float) -> float:
        """
        Update volatility estimate with new return (online update).
        
        Args:
            new_return: New return observation
        
        Returns:
            Updated volatility estimate
        """
        if self.current_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Update using EWMA formula
        self.current_volatility_ = np.sqrt(
            self.decay_factor * self.current_volatility_**2 + 
            (1 - self.decay_factor) * new_return**2
        )
        
        return self.current_volatility_
    
    def get_volatility_series(self) -> np.ndarray:
        """
        Get the full time series of volatility estimates.
        
        Returns:
            Array of volatility estimates for each time period
        """
        if self.volatilities_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.volatilities_.copy()
    
    def get_current_volatility(self) -> float:
        """
        Get the most recent volatility estimate.
        
        Returns:
            Current volatility estimate
        """
        if self.current_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.current_volatility_
    
    @staticmethod
    def fit_optimal_decay(
        returns: np.ndarray,
        realized_vol: np.ndarray,
        decay_range: tuple = (0.85, 0.99),
        n_trials: int = 50
    ) -> float:
        """
        Find optimal decay factor by minimizing error against realized volatility.
        
        Args:
            returns: Historical returns
            realized_vol: Realized volatility to compare against
            decay_range: Range of decay factors to search (min, max)
            n_trials: Number of decay factors to try
        
        Returns:
            Optimal decay factor
        """
        best_decay = decay_range[0]
        best_error = float('inf')
        
        decay_factors = np.linspace(decay_range[0], decay_range[1], n_trials)
        
        for decay in decay_factors:
            estimator = EWMAVolatilityEstimator(decay_factor=decay)
            estimator.fit(returns)
            predicted_vol = estimator.get_volatility_series()
            
            # Align lengths
            min_len = min(len(predicted_vol), len(realized_vol))
            error = np.mean((predicted_vol[-min_len:] - realized_vol[-min_len:])**2)
            
            if error < best_error:
                best_error = error
                best_decay = decay
        
        logger.info(
            f"Optimal decay found | decay={best_decay:.4f} | "
            f"error={best_error:.6f}"
        )
        
        return best_decay


class MultiAssetEWMAEstimator:
    """
    EWMA volatility estimator for multiple assets simultaneously.
    
    Efficiently estimates EWMA volatility for a portfolio of assets,
    useful for large-scale portfolio risk management.
    """
    
    def __init__(self, decay_factor: float = 0.94, min_periods: int = 10):
        """
        Initialize multi-asset EWMA estimator.
        
        Args:
            decay_factor: Decay factor for EWMA
            min_periods: Minimum periods for initial volatility
        """
        self.decay_factor = decay_factor
        self.min_periods = min_periods
        self.estimators = []
        self.volatilities_ = None
    
    def fit(self, returns: np.ndarray) -> 'MultiAssetEWMAEstimator':
        """
        Fit EWMA models to multiple asset returns.
        
        Args:
            returns: Returns matrix (n_assets, n_periods)
        
        Returns:
            Self for method chaining
        """
        if returns.ndim != 2:
            raise ValueError("Returns must be 2D array (n_assets, n_periods)")
        
        n_assets, n_periods = returns.shape
        self.volatilities_ = np.zeros((n_assets, n_periods))
        self.estimators = []
        
        for i in range(n_assets):
            estimator = EWMAVolatilityEstimator(
                decay_factor=self.decay_factor,
                min_periods=self.min_periods
            )
            estimator.fit(returns[i, :])
            self.estimators.append(estimator)
            self.volatilities_[i, :] = estimator.get_volatility_series()
        
        logger.info(f"MultiAsset EWMA fitted | n_assets={n_assets} | n_periods={n_periods}")
        
        return self
    
    def get_current_volatilities(self) -> np.ndarray:
        """
        Get current volatility estimates for all assets.
        
        Returns:
            Array of current volatilities for each asset
        """
        if not self.estimators:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return np.array([est.get_current_volatility() for est in self.estimators])
    
    def get_volatility_matrix(self) -> np.ndarray:
        """
        Get full volatility time series matrix.
        
        Returns:
            Volatility matrix (n_assets, n_periods)
        """
        if self.volatilities_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.volatilities_.copy()


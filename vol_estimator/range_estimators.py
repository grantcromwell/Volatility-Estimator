"""
Range-Based Volatility Estimators.

Implements Garman-Klass, Parkinson, Rogers-Satchell, and Yang-Zhang estimators
which use high, low, open, and close prices for more accurate volatility estimation.
These estimators are significantly more efficient than close-to-close methods.
"""

import numpy as np
from typing import Optional, Dict, Union
from numba import jit
from .logging_config import get_logger

logger = get_logger(__name__)


@jit(nopython=True, cache=True)
def _parkinson_volatility_numba(high: np.ndarray, low: np.ndarray) -> float:
    """
    Compute Parkinson volatility (daily).
    
    Formula: σ_P² = (1/4ln(2)) * E[(ln(H/L))²]
    
    Args:
        high: High prices
        low: Low prices
    
    Returns:
        Parkinson volatility estimate
    """
    n = len(high)
    sum_sq = 0.0
    
    for i in range(n):
        if high[i] > 0 and low[i] > 0:
            log_hl = np.log(high[i] / low[i])
            sum_sq += log_hl * log_hl
    
    return np.sqrt(sum_sq / (4 * np.log(2) * n))


@jit(nopython=True, cache=True)
def _garman_klass_volatility_numba(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    open_: np.ndarray
) -> float:
    """
    Compute Garman-Klass volatility.
    
    Formula: σ_GK² = 0.5 * (ln(H/L))² - (2ln(2) - 1) * (ln(C/O))²
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices
    
    Returns:
        Garman-Klass volatility estimate
    """
    n = len(high)
    sum_sq = 0.0
    factor = 2 * np.log(2) - 1
    
    for i in range(n):
        if high[i] > 0 and low[i] > 0 and close[i] > 0 and open_[i] > 0:
            log_hl = np.log(high[i] / low[i])
            log_co = np.log(close[i] / open_[i])
            sum_sq += 0.5 * log_hl * log_hl - factor * log_co * log_co
    
    return np.sqrt(sum_sq / n)


@jit(nopython=True, cache=True)
def _rogers_satchell_volatility_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray
) -> float:
    """
    Compute Rogers-Satchell volatility.
    
    Handles drift and is unbiased for zero drift.
    
    Formula: σ_RS² = E[ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)]
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices
    
    Returns:
        Rogers-Satchell volatility estimate
    """
    n = len(high)
    sum_val = 0.0
    
    for i in range(n):
        if high[i] > 0 and low[i] > 0 and close[i] > 0 and open_[i] > 0:
            log_hc = np.log(high[i] / close[i])
            log_ho = np.log(high[i] / open_[i])
            log_lc = np.log(low[i] / close[i])
            log_lo = np.log(low[i] / open_[i])
            sum_val += log_hc * log_ho + log_lc * log_lo
    
    return np.sqrt(sum_val / n)


class ParkinsonVolatilityEstimator:
    """
    Parkinson (1980) volatility estimator.
    
    Uses only high and low prices. About 5 times more efficient than
    close-to-close estimator. Best for assets without opening gaps.
    
    Reference:
    Parkinson, M. (1980). The Extreme Value Method for Estimating
    the Variance of the Rate of Return.
    """
    
    def __init__(self, window: Optional[int] = None):
        """
        Initialize Parkinson estimator.
        
        Args:
            window: Rolling window size. If None, uses all data.
        """
        self.window = window
        self.volatility_ = None
        self.volatility_series_ = None
    
    def fit(
        self, 
        high: Union[np.ndarray, list], 
        low: Union[np.ndarray, list]
    ) -> 'ParkinsonVolatilityEstimator':
        """
        Fit Parkinson volatility estimator.
        
        Args:
            high: High prices
            low: Low prices
        
        Returns:
            Self for method chaining
        """
        high = np.asarray(high)
        low = np.asarray(low)
        
        if len(high) != len(low):
            raise ValueError("high and low must have same length")
        
        if self.window is None:
            # Single volatility estimate
            self.volatility_ = _parkinson_volatility_numba(high, low)
            logger.debug(f"Parkinson fitted | n_periods={len(high)} | vol={self.volatility_:.6f}")
        else:
            # Rolling volatility
            n = len(high)
            self.volatility_series_ = np.zeros(n - self.window + 1)
            
            for i in range(len(self.volatility_series_)):
                high_window = high[i:i+self.window]
                low_window = low[i:i+self.window]
                self.volatility_series_[i] = _parkinson_volatility_numba(high_window, low_window)
            
            self.volatility_ = self.volatility_series_[-1]
            logger.debug(
                f"Parkinson fitted (rolling) | n_periods={len(high)} | "
                f"window={self.window} | current_vol={self.volatility_:.6f}"
            )
        
        return self
    
    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        if self.volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.volatility_
    
    def get_volatility_series(self) -> Optional[np.ndarray]:
        """Get rolling volatility series (if window was used)."""
        return self.volatility_series_


class GarmanKlassVolatilityEstimator:
    """
    Garman-Klass (1980) volatility estimator.
    
    Uses high, low, open, and close prices. About 8 times more efficient
    than close-to-close estimator. Assumes zero drift (no strong trend).
    
    Reference:
    Garman, M. B., & Klass, M. J. (1980). On the estimation of security
    price volatilities from historical data.
    """
    
    def __init__(self, window: Optional[int] = None):
        """
        Initialize Garman-Klass estimator.
        
        Args:
            window: Rolling window size. If None, uses all data.
        """
        self.window = window
        self.volatility_ = None
        self.volatility_series_ = None
    
    def fit(
        self,
        high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        open_: Union[np.ndarray, list]
    ) -> 'GarmanKlassVolatilityEstimator':
        """
        Fit Garman-Klass volatility estimator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_: Open prices
        
        Returns:
            Self for method chaining
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        open_ = np.asarray(open_)
        
        if not (len(high) == len(low) == len(close) == len(open_)):
            raise ValueError("All price arrays must have same length")
        
        if self.window is None:
            # Single volatility estimate
            self.volatility_ = _garman_klass_volatility_numba(high, low, close, open_)
            logger.debug(f"Garman-Klass fitted | n_periods={len(high)} | vol={self.volatility_:.6f}")
        else:
            # Rolling volatility
            n = len(high)
            self.volatility_series_ = np.zeros(n - self.window + 1)
            
            for i in range(len(self.volatility_series_)):
                h = high[i:i+self.window]
                l = low[i:i+self.window]
                c = close[i:i+self.window]
                o = open_[i:i+self.window]
                self.volatility_series_[i] = _garman_klass_volatility_numba(h, l, c, o)
            
            self.volatility_ = self.volatility_series_[-1]
            logger.debug(
                f"Garman-Klass fitted (rolling) | n_periods={len(high)} | "
                f"window={self.window} | current_vol={self.volatility_:.6f}"
            )
        
        return self
    
    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        if self.volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.volatility_
    
    def get_volatility_series(self) -> Optional[np.ndarray]:
        """Get rolling volatility series (if window was used)."""
        return self.volatility_series_


class RogersSatchellVolatilityEstimator:
    """
    Rogers-Satchell (1991) volatility estimator.
    
    Allows for non-zero drift and is unbiased under drift.
    Uses high, low, open, and close prices.
    
    Reference:
    Rogers, L. C. G., & Satchell, S. E. (1991). Estimating variance
    from high, low and closing prices.
    """
    
    def __init__(self, window: Optional[int] = None):
        """
        Initialize Rogers-Satchell estimator.
        
        Args:
            window: Rolling window size. If None, uses all data.
        """
        self.window = window
        self.volatility_ = None
        self.volatility_series_ = None
    
    def fit(
        self,
        high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        open_: Union[np.ndarray, list]
    ) -> 'RogersSatchellVolatilityEstimator':
        """
        Fit Rogers-Satchell volatility estimator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_: Open prices
        
        Returns:
            Self for method chaining
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        open_ = np.asarray(open_)
        
        if not (len(high) == len(low) == len(close) == len(open_)):
            raise ValueError("All price arrays must have same length")
        
        if self.window is None:
            # Single volatility estimate
            self.volatility_ = _rogers_satchell_volatility_numba(high, low, close, open_)
            logger.debug(f"Rogers-Satchell fitted | n_periods={len(high)} | vol={self.volatility_:.6f}")
        else:
            # Rolling volatility
            n = len(high)
            self.volatility_series_ = np.zeros(n - self.window + 1)
            
            for i in range(len(self.volatility_series_)):
                h = high[i:i+self.window]
                l = low[i:i+self.window]
                c = close[i:i+self.window]
                o = open_[i:i+self.window]
                self.volatility_series_[i] = _rogers_satchell_volatility_numba(h, l, c, o)
            
            self.volatility_ = self.volatility_series_[-1]
            logger.debug(
                f"Rogers-Satchell fitted (rolling) | n_periods={len(high)} | "
                f"window={self.window} | current_vol={self.volatility_:.6f}"
            )
        
        return self
    
    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        if self.volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.volatility_
    
    def get_volatility_series(self) -> Optional[np.ndarray]:
        """Get rolling volatility series (if window was used)."""
        return self.volatility_series_


class YangZhangVolatilityEstimator:
    """
    Yang-Zhang (2000) volatility estimator.
    
    Combines Rogers-Satchell with open jump and close-to-open volatility.
    Most efficient estimator, handles both drift and opening jumps.
    
    Reference:
    Yang, D., & Zhang, Q. (2000). Drift-independent volatility
    estimation based on high, low, open, and close prices.
    """
    
    def __init__(self, window: Optional[int] = None, k: float = 0.34):
        """
        Initialize Yang-Zhang estimator.
        
        Args:
            window: Rolling window size. If None, uses all data.
            k: Tuning parameter (default 0.34 is optimal for daily data)
        """
        self.window = window
        self.k = k
        self.volatility_ = None
        self.volatility_series_ = None
    
    def fit(
        self,
        high: Union[np.ndarray, list],
        low: Union[np.ndarray, list],
        close: Union[np.ndarray, list],
        open_: Union[np.ndarray, list]
    ) -> 'YangZhangVolatilityEstimator':
        """
        Fit Yang-Zhang volatility estimator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_: Open prices
        
        Returns:
            Self for method chaining
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        open_ = np.asarray(open_)
        
        if not (len(high) == len(low) == len(close) == len(open_)):
            raise ValueError("All price arrays must have same length")
        
        if self.window is None:
            self.volatility_ = self._compute_yang_zhang(high, low, close, open_)
            logger.debug(f"Yang-Zhang fitted | n_periods={len(high)} | vol={self.volatility_:.6f}")
        else:
            # Rolling volatility
            n = len(high)
            self.volatility_series_ = np.zeros(n - self.window + 1)
            
            for i in range(len(self.volatility_series_)):
                h = high[i:i+self.window]
                l = low[i:i+self.window]
                c = close[i:i+self.window]
                o = open_[i:i+self.window]
                self.volatility_series_[i] = self._compute_yang_zhang(h, l, c, o)
            
            self.volatility_ = self.volatility_series_[-1]
            logger.debug(
                f"Yang-Zhang fitted (rolling) | n_periods={len(high)} | "
                f"window={self.window} | current_vol={self.volatility_:.6f}"
            )
        
        return self
    
    def _compute_yang_zhang(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_: np.ndarray
    ) -> float:
        """Compute Yang-Zhang volatility."""
        n = len(high)
        
        # Overnight volatility (close-to-open)
        if n > 1:
            co = np.log(open_[1:] / close[:-1])
            var_o = np.var(co, ddof=1) if len(co) > 1 else 0
        else:
            var_o = 0
        
        # Open-to-close volatility
        oc = np.log(close / open_)
        var_c = np.var(oc, ddof=1) if len(oc) > 1 else 0
        
        # Rogers-Satchell volatility
        var_rs = _rogers_satchell_volatility_numba(high, low, close, open_)**2
        
        # Yang-Zhang combines these with optimal weights
        var_yz = var_o + self.k * var_c + (1 - self.k) * var_rs
        
        return np.sqrt(var_yz)
    
    def get_volatility(self) -> float:
        """Get current volatility estimate."""
        if self.volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.volatility_
    
    def get_volatility_series(self) -> Optional[np.ndarray]:
        """Get rolling volatility series (if window was used)."""
        return self.volatility_series_


def compare_estimators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    window: Optional[int] = None
) -> Dict[str, float]:
    """
    Compare all range-based volatility estimators.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices
        window: Optional rolling window
    
    Returns:
        Dictionary of volatility estimates from each method
    """
    results = {}
    
    # Parkinson (only needs high/low)
    pk = ParkinsonVolatilityEstimator(window=window)
    pk.fit(high, low)
    results['Parkinson'] = pk.get_volatility()
    
    # Garman-Klass
    gk = GarmanKlassVolatilityEstimator(window=window)
    gk.fit(high, low, close, open_)
    results['Garman-Klass'] = gk.get_volatility()
    
    # Rogers-Satchell
    rs = RogersSatchellVolatilityEstimator(window=window)
    rs.fit(high, low, close, open_)
    results['Rogers-Satchell'] = rs.get_volatility()
    
    # Yang-Zhang
    yz = YangZhangVolatilityEstimator(window=window)
    yz.fit(high, low, close, open_)
    results['Yang-Zhang'] = yz.get_volatility()
    
    # Simple close-to-close for comparison
    if len(close) > 1:
        returns = np.diff(np.log(close))
        results['Close-to-Close'] = np.std(returns)
    
    logger.info(f"Range estimators compared | {' | '.join(f'{k}={v:.6f}' for k, v in results.items())}")
    
    return results


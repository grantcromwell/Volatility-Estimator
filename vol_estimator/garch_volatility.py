"""
GARCH Volatility Models.

Industry-standard GARCH family models for volatility estimation:
- GARCH(1,1): Standard GARCH model
- EGARCH: Exponential GARCH (captures asymmetry)
- GJR-GARCH: Glosten-Jagannathan-Runkle GARCH (leverage effects)

These models capture volatility clustering and are widely used in finance.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from .logging_config import get_logger

logger = get_logger(__name__)

# Check if arch is available
try:
    from arch import arch_model
    from arch.univariate import ConstantMean, GARCH, EGARCH, VolatilityProcess
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning(
        "arch library not available. Install with: pip install arch. "
        "GARCH models will not be functional."
    )


class GARCHVolatilityEstimator:
    """
    GARCH(p,q) volatility estimator.
    
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is
    the industry-standard model for volatility forecasting. It captures
    volatility clustering - periods of high/low volatility tend to cluster.
    
    GARCH(1,1) formula:
    σ²(t) = ω + α * ε²(t-1) + β * σ²(t-1)
    
    Reference:
    Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. Journal of Econometrics.
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean_model: str = 'Constant',
        vol_model: str = 'GARCH',
        dist: str = 'normal'
    ):
        """
        Initialize GARCH volatility estimator.
        
        Args:
            p: Order of GARCH term (lag of variance)
            q: Order of ARCH term (lag of squared residuals)
            mean_model: Mean model ('Constant', 'Zero', 'AR', 'ARX')
            vol_model: Volatility model ('GARCH', 'EGARCH', 'GJR')
            dist: Distribution ('normal', 't', 'skewt')
        """
        if not ARCH_AVAILABLE:
            raise ImportError(
                "arch library is required for GARCH models. "
                "Install with: pip install arch"
            )
        
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist
        
        self.model_ = None
        self.result_ = None
        self.conditional_volatility_ = None
        
        logger.info(
            f"GARCH initialized | model={vol_model}({p},{q}) | "
            f"mean={mean_model} | dist={dist}"
        )
    
    def fit(
        self,
        returns: Union[np.ndarray, list],
        update_freq: int = 0,
        disp: str = 'off'
    ) -> 'GARCHVolatilityEstimator':
        """
        Fit GARCH model to returns.
        
        Args:
            returns: Array of returns (can be percentage or log returns)
            update_freq: Frequency of iteration updates (0 = no updates)
            disp: Display option ('off', 'final', 'iter')
        
        Returns:
            Self for method chaining
        """
        returns = np.asarray(returns) * 100  # Scale to percentage for numerical stability
        
        if len(returns) < 100:
            logger.warning(
                f"GARCH with limited data | n_periods={len(returns)} | "
                "Recommend at least 100 observations"
            )
        
        # Create model
        self.model_ = arch_model(
            returns,
            mean=self.mean_model,
            vol=self.vol_model,
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        
        # Fit model
        try:
            self.result_ = self.model_.fit(update_freq=update_freq, disp=disp)
            
            # Extract conditional volatility (convert back from percentage)
            self.conditional_volatility_ = self.result_.conditional_volatility.values / 100
            
            logger.info(
                f"GARCH fitted | n_periods={len(returns)} | "
                f"current_vol={self.conditional_volatility_[-1]:.6f}"
            )
            
        except Exception as e:
            logger.error(f"GARCH fitting failed | error={e}")
            raise
        
        return self
    
    def forecast(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> Union[float, np.ndarray]:
        """
        Forecast volatility for future periods.
        
        Args:
            horizon: Number of periods ahead to forecast
            method: Forecasting method ('analytic', 'simulation', 'bootstrap')
        
        Returns:
            Volatility forecast (scalar if horizon=1, array otherwise)
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate forecast
        forecast = self.result_.forecast(horizon=horizon, method=method)
        variance_forecast = forecast.variance.values[-1, :] / 10000  # Scale back
        volatility_forecast = np.sqrt(variance_forecast)
        
        if horizon == 1:
            return volatility_forecast[0]
        else:
            return volatility_forecast
    
    def get_conditional_volatility(self) -> np.ndarray:
        """
        Get the conditional volatility time series.
        
        Returns:
            Array of conditional volatility estimates
        """
        if self.conditional_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.conditional_volatility_.copy()
    
    def get_current_volatility(self) -> float:
        """
        Get the most recent volatility estimate.
        
        Returns:
            Current volatility estimate
        """
        if self.conditional_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.conditional_volatility_[-1]
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get fitted model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return dict(self.result_.params)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model statistics and diagnostics
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return {
            'aic': self.result_.aic,
            'bic': self.result_.bic,
            'loglikelihood': self.result_.loglikelihood,
            'parameters': self.get_parameters(),
            'num_obs': self.result_.nobs,
            'model': f"{self.vol_model}({self.p},{self.q})"
        }


class EGARCHVolatilityEstimator(GARCHVolatilityEstimator):
    """
    EGARCH (Exponential GARCH) volatility estimator.
    
    EGARCH captures asymmetric effects where negative returns increase
    volatility more than positive returns of the same magnitude.
    
    Reference:
    Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns:
    A new approach. Econometrica.
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean_model: str = 'Constant',
        dist: str = 'normal'
    ):
        """
        Initialize EGARCH volatility estimator.
        
        Args:
            p: Order of GARCH term
            q: Order of ARCH term
            mean_model: Mean model type
            dist: Distribution assumption
        """
        super().__init__(
            p=p,
            q=q,
            mean_model=mean_model,
            vol_model='EGARCH',
            dist=dist
        )


class GJRGARCHVolatilityEstimator(GARCHVolatilityEstimator):
    """
    GJR-GARCH volatility estimator.
    
    GJR-GARCH (Glosten-Jagannathan-Runkle) captures leverage effects
    where negative shocks have larger impact on volatility than positive shocks.
    
    Reference:
    Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993).
    On the relation between the expected value and the volatility.
    """
    
    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean_model: str = 'Constant',
        dist: str = 'normal'
    ):
        """
        Initialize GJR-GARCH volatility estimator.
        
        Args:
            p: Order of symmetric GARCH term
            o: Order of asymmetric (leverage) term
            q: Order of ARCH term
            mean_model: Mean model type
            dist: Distribution assumption
        """
        # Note: For arch library, we use vol='GARCH' with additional parameters
        super().__init__(
            p=p,
            q=q,
            mean_model=mean_model,
            vol_model='GARCH',  # Will be configured as GJR in fit
            dist=dist
        )
        self.o = o
        self.vol_model = 'GJR'  # Override for proper identification


def compare_garch_models(
    returns: np.ndarray,
    models: Optional[list] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different GARCH model specifications.
    
    Args:
        returns: Historical returns
        models: List of model configurations. If None, uses default set.
    
    Returns:
        Dictionary of model results with AIC/BIC for comparison
    """
    if not ARCH_AVAILABLE:
        logger.error("arch library not available for GARCH comparison")
        return {}
    
    if models is None:
        models = [
            ('GARCH(1,1)', {'p': 1, 'q': 1, 'vol_model': 'GARCH'}),
            ('EGARCH(1,1)', {'p': 1, 'q': 1, 'vol_model': 'EGARCH'}),
            ('GARCH(2,2)', {'p': 2, 'q': 2, 'vol_model': 'GARCH'})
        ]
    
    results = {}
    
    for model_name, config in models:
        try:
            estimator = GARCHVolatilityEstimator(**config)
            estimator.fit(returns, disp='off')
            info = estimator.get_model_info()
            
            results[model_name] = {
                'aic': info['aic'],
                'bic': info['bic'],
                'loglik': info['loglikelihood'],
                'current_vol': estimator.get_current_volatility(),
                'parameters': info['parameters']
            }
            
            logger.info(
                f"GARCH comparison | model={model_name} | "
                f"AIC={info['aic']:.2f} | BIC={info['bic']:.2f}"
            )
            
        except Exception as e:
            logger.warning(f"GARCH comparison failed | model={model_name} | error={e}")
    
    # Find best model by BIC (lower is better)
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['bic'])
        logger.info(f"Best GARCH model | model={best_model[0]} | BIC={best_model[1]['bic']:.2f}")
    
    return results


# Simple GARCH implementation fallback (if arch library not available)
class SimpleGARCH11:
    """
    Simple GARCH(1,1) implementation without external dependencies.
    
    This is a fallback implementation with basic functionality.
    For production use, install the arch library for full GARCH support.
    """
    
    def __init__(self, omega: float = 0.0001, alpha: float = 0.1, beta: float = 0.85):
        """
        Initialize simple GARCH(1,1).
        
        Args:
            omega: Constant term
            alpha: ARCH coefficient
            beta: GARCH coefficient
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conditional_volatility_ = None
        
        logger.warning("Using simple GARCH fallback. Install arch library for full functionality.")
    
    def fit(self, returns: np.ndarray) -> 'SimpleGARCH11':
        """Fit simple GARCH(1,1) using recursive formula."""
        returns = np.asarray(returns)
        n = len(returns)
        
        # Initialize
        variance = np.zeros(n)
        variance[0] = np.var(returns)
        
        # Recursive calculation
        for t in range(1, n):
            variance[t] = (
                self.omega + 
                self.alpha * returns[t-1]**2 + 
                self.beta * variance[t-1]
            )
        
        self.conditional_volatility_ = np.sqrt(variance)
        
        logger.info(f"Simple GARCH(1,1) fitted | n_periods={n}")
        
        return self
    
    def get_current_volatility(self) -> float:
        """Get current volatility estimate."""
        if self.conditional_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.conditional_volatility_[-1]
    
    def get_conditional_volatility(self) -> np.ndarray:
        """Get conditional volatility series."""
        if self.conditional_volatility_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.conditional_volatility_.copy()


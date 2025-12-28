"""Main volatility estimator with validation"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Optional, Tuple
from .correlation_mst import CorrelationMST
from .lorenz_attractor import LorenzVolatilityModel
from .hmm_regime import HMMRegimeDetector
from .api_clients import FinancialDataFetcher
from .sector_dfs import SectorDFSVolatility, SectorVolatility
from .ewma_volatility import EWMAVolatilityEstimator, MultiAssetEWMAEstimator
from .range_estimators import (
    ParkinsonVolatilityEstimator,
    GarmanKlassVolatilityEstimator,
    YangZhangVolatilityEstimator
)
from .garch_volatility import GARCHVolatilityEstimator, ARCH_AVAILABLE
from .logging_config import get_logger, MetricsLogger
from .validation import calculate_accuracy_metrics, compute_realized_volatility

logger = get_logger(__name__)
metrics_logger = MetricsLogger()


class VolatilityEstimator:
    
    def __init__(
        self,
        mst_enabled: bool = True,
        lorenz_enabled: bool = True,
        hmm_enabled: bool = True,
        sector_dfs_enabled: bool = False,
        ewma_enabled: bool = True,
        garch_enabled: bool = False,
        range_estimator_enabled: bool = False,
        alpha_vantage_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize volatility estimator.
        
        Args:
            mst_enabled: Enable MST correlation analysis
            lorenz_enabled: Enable Lorenz attractor modeling
            hmm_enabled: Enable HMM regime detection
            sector_dfs_enabled: Enable sector-based analysis
            ewma_enabled: Enable EWMA volatility (industry-standard)
            garch_enabled: Enable GARCH models (requires arch library)
            range_estimator_enabled: Enable range-based estimators (Garman-Klass, etc.)
            alpha_vantage_key: API key for Alpha Vantage
            finnhub_key: API key for Finnhub
            ensemble_weights: Custom weights for ensemble models
        """
        self.mst_enabled = mst_enabled
        self.lorenz_enabled = lorenz_enabled
        self.hmm_enabled = hmm_enabled
        self.ewma_enabled = ewma_enabled
        self.garch_enabled = garch_enabled
        self.range_estimator_enabled = range_estimator_enabled
        
        # Initialize components
        if mst_enabled:
            self.mst = CorrelationMST()
        else:
            self.mst = None
        
        if lorenz_enabled:
            self.lorenz = LorenzVolatilityModel()
        else:
            self.lorenz = None
        
        if hmm_enabled:
            self.hmm = HMMRegimeDetector(n_states=3)
        else:
            self.hmm = None
        
        if sector_dfs_enabled:
            self.sector_dfs = SectorDFSVolatility()
        else:
            self.sector_dfs = None
        
        if ewma_enabled:
            self.ewma = MultiAssetEWMAEstimator(decay_factor=0.94)
        else:
            self.ewma = None
        
        if garch_enabled:
            if not ARCH_AVAILABLE:
                logger.warning("GARCH enabled but arch library not available. Disabling GARCH.")
                self.garch_enabled = False
                self.garch = None
            else:
                self.garch = None  # Will be initialized per-asset
        else:
            self.garch = None
        
        self.data_fetcher = FinancialDataFetcher(
            alpha_vantage_key=alpha_vantage_key,
            finnhub_key=finnhub_key
        )
        
        # Ensemble weights (default or custom)
        self.ensemble_weights = ensemble_weights or {
            'simple_std': 0.1,
            'ewma': 0.3,
            'garch': 0.2,
            'hmm': 0.15,
            'mst': 0.15,
            'lorenz': 0.1
        }
        
        # Storage for intermediate results
        self.returns_data = None
        self.symbols = None
        self.individual_volatilities = None
        self.final_volatility = None
        self.model_components = {}  # Store individual model results
        
        logger.info(
            f"VolatilityEstimator initialized | "
            f"mst={mst_enabled} | lorenz={lorenz_enabled} | hmm={hmm_enabled} | "
            f"ewma={ewma_enabled} | garch={garch_enabled} | range={range_estimator_enabled}"
        )
    
    def fetch_market_data(self, symbols: List[str], period: str = "1y", source: str = "auto") -> Dict[str, pd.DataFrame]:
        data = self.data_fetcher.fetch_data(symbols, period=period, source=source)
        return data
    
    def prepare_returns(self, data: Dict[str, pd.DataFrame], method: str = "log", min_length: int = 100) -> Tuple[np.ndarray, List[str]]:
        returns_dict = self.data_fetcher.compute_returns(data, method=method)
        returns_matrix, symbols = self.data_fetcher.align_returns(
            returns_dict, min_length=min_length
        )
        
        self.returns_data = returns_matrix
        self.symbols = symbols
        
        return returns_matrix, symbols
    
    def compute_individual_volatilities(self, returns: np.ndarray, window: Optional[int] = None) -> np.ndarray:
        if window is None:
            volatilities = np.std(returns, axis=1)
        else:
            volatilities = np.std(returns[:, -window:], axis=1)
        
        self.individual_volatilities = volatilities
        
        return volatilities
    
    def estimate_volatility(self, returns: Optional[np.ndarray] = None, symbols: Optional[List[str]] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if returns is None:
            if self.returns_data is None:
                raise ValueError("No returns data available. "
                               "Call prepare_returns or fetch_market_data first.")
            returns = self.returns_data
        else:
            self.returns_data = returns
        
        if symbols is None:
            if self.symbols is None:
                symbols = [f"Asset_{i}" for i in range(returns.shape[0])]
            else:
                symbols = self.symbols
        else:
            self.symbols = symbols
        
        n_assets, n_periods = returns.shape
        
        individual_vols = self.compute_individual_volatilities(returns)
        
        regime_adjusted_vols = individual_vols.copy()
        if self.hmm_enabled and self.hmm is not None:
            try:
                avg_returns = np.mean(returns, axis=0)
                self.hmm.fit(avg_returns)
                
                regime_vol = self.hmm.estimate_regime_volatility(avg_returns)
                
                base_vol = np.std(avg_returns)
                if base_vol > 0:
                    regime_factor = regime_vol[-1] / base_vol
                    regime_adjusted_vols = individual_vols * regime_factor
                    logger.debug(f"HMM applied | regime_factor={regime_factor:.4f}")
            except Exception as e:
                logger.warning(f"HMM estimation failed | error={e}")
        
        if self.mst_enabled:
            try:
                self.mst.compute_correlation_matrix(returns, asset_names=symbols)
                self.mst.build_mst()
                
                mst_weighted_vols = self.mst.compute_mst_volatility_weights(
                    regime_adjusted_vols
                )
                
                final_vols = mst_weighted_vols
                logger.debug("MST weighting applied")
            except Exception as e:
                logger.warning(f"MST estimation failed | error={e}")
                final_vols = regime_adjusted_vols
        else:
            final_vols = regime_adjusted_vols
        
        if self.lorenz_enabled:
            try:
                avg_returns = np.mean(returns, axis=0)
                lorenz_vol = self.lorenz.fit_to_returns(avg_returns)
                
                base_vol = np.std(avg_returns)
                if base_vol > 0 and len(lorenz_vol) > 0:
                    lorenz_factor = lorenz_vol[-1] / base_vol
                    final_vols = 0.7 * final_vols + 0.3 * (final_vols * lorenz_factor)
                    logger.debug(f"Lorenz applied | lorenz_factor={lorenz_factor:.4f}")
            except Exception as e:
                logger.warning(f"Lorenz estimation failed | error={e}")
        
        if weights:
            weight_array = np.array([weights.get(sym, 1.0) for sym in symbols])
            weight_array = weight_array / weight_array.sum()
            final_vols = final_vols * weight_array
        
        self.final_volatility = dict(zip(symbols, final_vols))
        self.final_volatility = dict(zip(symbols, final_vols))
        
        return self.final_volatility
    
    def estimate_portfolio_volatility(
        self,
        returns: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate portfolio-level volatility.
        
        Args:
            returns: Returns matrix (n_assets, n_periods)
            weights: Portfolio weights. If None, uses equal weights
        
        Returns:
            Portfolio volatility estimate
        """
        if returns is None:
            if self.returns_data is None:
                raise ValueError("No returns data available.")
            returns = self.returns_data
        
        n_assets = returns.shape[0]
        
        if weights is None:
            weights = np.ones(n_assets) / n_assets
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Compute portfolio returns
        portfolio_returns = np.dot(weights, returns)
        
        # Estimate volatility using integrated approach
        # Create temporary estimator for portfolio
        portfolio_vol_estimator = VolatilityEstimator(
            mst_enabled=False,  # Single asset, no correlation needed
            lorenz_enabled=self.lorenz_enabled,
            hmm_enabled=self.hmm_enabled
        )
        
        portfolio_returns_2d = portfolio_returns.reshape(1, -1)
        vol_dict = portfolio_vol_estimator.estimate_volatility(
            returns=portfolio_returns_2d,
            symbols=["Portfolio"]
        )
        
        return vol_dict["Portfolio"]
    
    def get_correlation_structure(self) -> Optional[Dict]:
        """
        Get correlation structure from MST analysis.
        
        Returns:
            Dictionary with MST information, or None if MST not computed
        """
        if self.mst is None or self.mst.mst_graph is None:
            return None
        
        return {
            "mst_edges": self.mst.get_mst_edges(),
            "core_assets": self.mst.get_core_assets(),
            "correlation_matrix": self.mst.correlation_matrix.tolist()
        }
    
    def get_regime_info(self) -> Optional[Dict]:
        """
        Get HMM regime information.
        
        Returns:
            Dictionary with regime information, or None if HMM not fitted
        """
        if self.hmm is None:
            return None
        
        if self.returns_data is None:
            return None
        
        # Check if HMM has been fitted (has model attribute)
        if not hasattr(self.hmm, 'model') or self.hmm.model is None:
            return None
        
        avg_returns = np.mean(self.returns_data, axis=0)
        states = self.hmm.predict_states(avg_returns)
        state_vols = self.hmm.get_state_volatilities(avg_returns)
        regimes = self.hmm.detect_regime_changes(avg_returns)
        
        return {
            "states": states.tolist(),
            "state_volatilities": state_vols.tolist(),
            "regime_changes": regimes,
            "transition_matrix": self.hmm.get_transition_matrix().tolist(),
            "current_state": int(states[-1]) if len(states) > 0 else None
        }
    
    def get_lorenz_info(self) -> Optional[Dict]:
        """
        Get Lorenz attractor information.
        
        Returns:
            Dictionary with Lorenz model info, or None if not computed
        """
        if self.lorenz is None or self.lorenz.trajectory is None:
            return None
        
        lyapunov = self.lorenz.compute_lyapunov_exponent()
        transitions = self.lorenz.detect_regime_transitions()
        
        return {
            "lyapunov_exponent": float(lyapunov),
            "regime_transitions": int(transitions.sum()),
            "trajectory_shape": self.lorenz.trajectory.shape
        }
    
    def estimate_sector_relative_volatility(self, returns: Optional[np.ndarray] = None, symbols: Optional[List[str]] = None) -> Dict[str, SectorVolatility]:
        if self.sector_dfs is None:
            raise ValueError("Sector DFS not enabled. Initialize with sector_dfs_enabled=True")
        
        if returns is None:
            if self.returns_data is None:
                raise ValueError("No returns data available.")
            returns = self.returns_data
        
        if symbols is None:
            if self.symbols is None:
                raise ValueError("No symbols available.")
            symbols = self.symbols
        
        sectors = self.data_fetcher.get_sectors(symbols)
        
        individual_vols = self.compute_individual_volatilities(returns)
        vol_dict = dict(zip(symbols, individual_vols))
        
        correlation_matrix = None
        if self.mst_enabled and self.mst is not None:
            self.mst.compute_correlation_matrix(returns, asset_names=symbols)
            correlation_matrix = self.mst.correlation_matrix
        
        self.sector_dfs.build_sector_hierarchy(
            symbols=symbols,
            sectors=sectors,
            correlations=correlation_matrix
        )
        
        sector_volatilities = self.sector_dfs.dfs_sector_volatility(vol_dict)
        sector_volatilities = self.sector_dfs.dfs_sector_volatility(vol_dict)
        
        return sector_volatilities

    def get_sector_summary(self) -> Optional[Dict]:
        if self.sector_dfs is None:
            return None
        return self.sector_dfs.get_sector_summary()
    
    def estimate_with_ewma(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate volatility using EWMA (industry-standard).
        
        Args:
            returns: Returns matrix (n_assets, n_periods)
        
        Returns:
            Current volatility estimates for each asset
        """
        if not self.ewma_enabled or self.ewma is None:
            logger.warning("EWMA not enabled")
            return np.std(returns, axis=1)
        
        start_time = time.time()
        
        try:
            self.ewma.fit(returns)
            vols = self.ewma.get_current_volatilities()
            
            execution_time = (time.time() - start_time) * 1000
            metrics_logger.log_estimation(
                model_name="EWMA",
                n_assets=returns.shape[0],
                n_periods=returns.shape[1],
                execution_time_ms=execution_time,
                success=True
            )
            
            self.model_components['ewma'] = vols
            return vols
            
        except Exception as e:
            logger.error(f"EWMA estimation failed | error={e}")
            return np.std(returns, axis=1)
    
    def estimate_with_garch(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate volatility using GARCH models.
        
        Args:
            returns: Returns matrix (n_assets, n_periods)
        
        Returns:
            Current volatility estimates for each asset
        """
        if not self.garch_enabled or not ARCH_AVAILABLE:
            logger.warning("GARCH not enabled or arch library not available")
            return np.std(returns, axis=1)
        
        n_assets = returns.shape[0]
        vols = np.zeros(n_assets)
        
        start_time = time.time()
        
        for i in range(n_assets):
            try:
                garch = GARCHVolatilityEstimator(p=1, q=1)
                garch.fit(returns[i, :], disp='off')
                vols[i] = garch.get_current_volatility()
            except Exception as e:
                logger.warning(f"GARCH failed for asset {i} | error={e}")
                vols[i] = np.std(returns[i, :])
        
        execution_time = (time.time() - start_time) * 1000
        metrics_logger.log_estimation(
            model_name="GARCH",
            n_assets=n_assets,
            n_periods=returns.shape[1],
            execution_time_ms=execution_time,
            success=True
        )
        
        self.model_components['garch'] = vols
        return vols
    
    def estimate_ensemble(
        self,
        returns: Optional[np.ndarray] = None,
        symbols: Optional[List[str]] = None,
        use_validation: bool = False
    ) -> Dict[str, float]:
        """
        Estimate volatility using ensemble of all available models.
        
        Combines multiple models with weighted averaging for robust estimation.
        
        Args:
            returns: Returns matrix (n_assets, n_periods)
            symbols: Asset symbols
            use_validation: Whether to validate against realized volatility
        
        Returns:
            Dictionary of volatility estimates by symbol
        """
        if returns is None:
            if self.returns_data is None:
                raise ValueError("No returns data available")
            returns = self.returns_data
        
        if symbols is None:
            if self.symbols is None:
                symbols = [f"Asset_{i}" for i in range(returns.shape[0])]
            else:
                symbols = self.symbols
        
        n_assets, n_periods = returns.shape
        logger.info(f"Ensemble estimation started | assets={n_assets} | periods={n_periods}")
        
        start_time = time.time()
        
        # Get estimates from all enabled models
        model_estimates = {}
        
        # Simple std (baseline)
        model_estimates['simple_std'] = np.std(returns, axis=1)
        
        # EWMA
        if self.ewma_enabled:
            model_estimates['ewma'] = self.estimate_with_ewma(returns)
        
        # GARCH
        if self.garch_enabled and ARCH_AVAILABLE:
            model_estimates['garch'] = self.estimate_with_garch(returns)
        
        # HMM
        if self.hmm_enabled and self.hmm is not None:
            try:
                avg_returns = np.mean(returns, axis=0)
                self.hmm.fit(avg_returns)
                regime_vol = self.hmm.estimate_regime_volatility(avg_returns)
                base_vol = np.std(avg_returns)
                if base_vol > 0:
                    regime_factor = regime_vol[-1] / base_vol
                    model_estimates['hmm'] = model_estimates['simple_std'] * regime_factor
            except Exception as e:
                logger.warning(f"HMM in ensemble failed | error={e}")
        
        # MST
        if self.mst_enabled:
            try:
                self.mst.compute_correlation_matrix(returns, asset_names=symbols)
                self.mst.build_mst()
                model_estimates['mst'] = self.mst.compute_mst_volatility_weights(
                    model_estimates['simple_std']
                )
            except Exception as e:
                logger.warning(f"MST in ensemble failed | error={e}")
        
        # Lorenz
        if self.lorenz_enabled:
            try:
                avg_returns = np.mean(returns, axis=0)
                lorenz_vol = self.lorenz.fit_to_returns(avg_returns)
                base_vol = np.std(avg_returns)
                if base_vol > 0 and len(lorenz_vol) > 0:
                    lorenz_factor = lorenz_vol[-1] / base_vol
                    model_estimates['lorenz'] = model_estimates['simple_std'] * lorenz_factor
            except Exception as e:
                logger.warning(f"Lorenz in ensemble failed | error={e}")
        
        # Weighted ensemble
        ensemble_vols = np.zeros(n_assets)
        total_weight = 0.0
        
        for model_name, estimate in model_estimates.items():
            weight = self.ensemble_weights.get(model_name, 0.0)
            if weight > 0:
                ensemble_vols += weight * estimate
                total_weight += weight
        
        if total_weight > 0:
            ensemble_vols /= total_weight
        
        execution_time = (time.time() - start_time) * 1000
        
        # Log performance
        metrics_logger.log_estimation(
            model_name="Ensemble",
            n_assets=n_assets,
            n_periods=n_periods,
            execution_time_ms=execution_time,
            success=True
        )
        
        result = dict(zip(symbols, ensemble_vols))
        
        logger.info(
            f"Ensemble estimation complete | models={len(model_estimates)} | "
            f"time={execution_time:.2f}ms"
        )
        
        return result
    
    def validate_predictions(
        self,
        predictions: np.ndarray,
        returns: np.ndarray,
        realized_vol_window: int = 20
    ) -> Dict:
        """
        Validate volatility predictions against realized volatility.
        
        Args:
            predictions: Predicted volatilities
            returns: Historical returns for computing realized volatility
            realized_vol_window: Window for realized volatility calculation
        
        Returns:
            Dictionary with validation metrics
        """
        # Compute realized volatility
        realized_vol = compute_realized_volatility(
            returns.flatten() if returns.ndim > 1 else returns,
            window=realized_vol_window
        )
        
        # Align lengths
        min_len = min(len(predictions), len(realized_vol))
        predictions_aligned = predictions[-min_len:]
        realized_aligned = realized_vol[-min_len:]
        
        # Calculate metrics
        metrics = calculate_accuracy_metrics(predictions_aligned, realized_aligned)
        
        # Log metrics
        metrics_logger.log_accuracy(
            model_name="VolatilityEstimator",
            mae=metrics.mae,
            rmse=metrics.rmse,
            mape=metrics.mape,
            n_samples=metrics.n_samples
        )
        
        return {
            'metrics': metrics,
            'predictions': predictions_aligned,
            'realized': realized_aligned
        }
    
    def compare_with_benchmark(
        self,
        returns: np.ndarray,
        benchmark_model: str = 'simple_std'
    ) -> Dict:
        """
        Compare estimator with benchmark model.
        
        Args:
            returns: Returns data
            benchmark_model: Benchmark model ('simple_std', 'ewma', 'garch')
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Benchmarking against {benchmark_model}")
        
        # Get predictions from main estimator
        main_vol = self.estimate_volatility(returns)
        main_vols_array = np.array(list(main_vol.values()))
        
        # Get benchmark predictions
        if benchmark_model == 'simple_std':
            benchmark_vols = np.std(returns, axis=1)
        elif benchmark_model == 'ewma':
            benchmark_vols = self.estimate_with_ewma(returns)
        elif benchmark_model == 'garch' and self.garch_enabled:
            benchmark_vols = self.estimate_with_garch(returns)
        else:
            logger.error(f"Unknown benchmark model: {benchmark_model}")
            return {}
        
        # Compute realized volatility for comparison
        avg_returns = np.mean(returns, axis=0)
        realized_vol = compute_realized_volatility(avg_returns, window=20)
        
        # Calculate metrics for both
        main_metrics = calculate_accuracy_metrics(
            np.full(len(realized_vol), np.mean(main_vols_array)),
            realized_vol
        )
        
        benchmark_metrics = calculate_accuracy_metrics(
            np.full(len(realized_vol), np.mean(benchmark_vols)),
            realized_vol
        )
        
        improvement = {
            'mae_improvement_%': (
                (benchmark_metrics.mae - main_metrics.mae) / benchmark_metrics.mae * 100
            ),
            'rmse_improvement_%': (
                (benchmark_metrics.rmse - main_metrics.rmse) / benchmark_metrics.rmse * 100
            ),
            'main_model_metrics': main_metrics,
            'benchmark_metrics': benchmark_metrics
        }
        
        logger.info(
            f"Benchmark comparison | model={benchmark_model} | "
            f"MAE_improvement={improvement['mae_improvement_%']:.2f}% | "
            f"RMSE_improvement={improvement['rmse_improvement_%']:.2f}%"
        )
        
        return improvement


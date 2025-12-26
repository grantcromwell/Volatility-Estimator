"""Main volatility estimator"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from .correlation_mst import CorrelationMST
from .lorenz_attractor import LorenzVolatilityModel
from .hmm_regime import HMMRegimeDetector
from .api_clients import FinancialDataFetcher
from .sector_dfs import SectorDFSVolatility, SectorVolatility


class VolatilityEstimator:
    
    def __init__(
        self,
        mst_enabled: bool = True,
        lorenz_enabled: bool = True,
        hmm_enabled: bool = True,
        sector_dfs_enabled: bool = False,
        alpha_vantage_key: Optional[str] = None,
        finnhub_key: Optional[str] = None
    ):
        self.mst_enabled = mst_enabled
        self.lorenz_enabled = lorenz_enabled
        self.hmm_enabled = hmm_enabled
        
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
        
        self.data_fetcher = FinancialDataFetcher(
            alpha_vantage_key=alpha_vantage_key,
            finnhub_key=finnhub_key
        )
        
        # Storage for intermediate results
        self.returns_data = None
        self.symbols = None
        self.individual_volatilities = None
        self.final_volatility = None
    
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
            except Exception as e:
                print(f"Warning: HMM estimation failed: {e}")
        
        if self.mst_enabled:
            try:
                self.mst.compute_correlation_matrix(returns, asset_names=symbols)
                self.mst.build_mst()
                
                mst_weighted_vols = self.mst.compute_mst_volatility_weights(
                    regime_adjusted_vols
                )
                
                final_vols = mst_weighted_vols
            except Exception as e:
                print(f"Warning: MST estimation failed: {e}")
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
            except Exception as e:
                print(f"Warning: Lorenz estimation failed: {e}")
        
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


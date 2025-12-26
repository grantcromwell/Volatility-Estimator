"""Tests for volatility estimator"""

import numpy as np
import pytest
from vol_estimator import (
    VolatilityEstimator,
    CorrelationMST,
    LorenzVolatilityModel,
    HMMRegimeDetector,
    SectorDFSVolatility,
    SectorVolatility
)


class TestCorrelationMST:
    
    def test_correlation_matrix_computation(self):
        mst = CorrelationMST()
        
        np.random.seed(42)
        returns = np.random.randn(5, 100)
        
        corr_matrix = mst.compute_correlation_matrix(returns)
        
        assert corr_matrix.shape == (5, 5)
        assert np.allclose(np.diag(corr_matrix), 1.0)
        assert np.all(corr_matrix >= -1) and np.all(corr_matrix <= 1)
    
    def test_mst_construction(self):
        """Test MST construction"""
        mst = CorrelationMST()
        
        np.random.seed(42)
        returns = np.random.randn(5, 100)
        asset_names = ['A', 'B', 'C', 'D', 'E']
        
        mst.compute_correlation_matrix(returns, asset_names=asset_names)
        mst_graph = mst.build_mst()
        
        assert mst_graph.number_of_nodes() == 5
        assert mst_graph.number_of_edges() == 4
    
    def test_core_assets(self):
        """Test core asset identification"""
        mst = CorrelationMST()
        
        np.random.seed(42)
        returns = np.random.randn(5, 100)
        asset_names = ['A', 'B', 'C', 'D', 'E']
        
        mst.compute_correlation_matrix(returns, asset_names=asset_names)
        mst.build_mst()
        
        core_assets = mst.get_core_assets(top_n=3)
        assert len(core_assets) == 3
        assert all(asset in asset_names for asset in core_assets)


class TestLorenzAttractor:
    
    def test_lorenz_simulation(self):
        """Test Lorenz system simulation"""
        lorenz = LorenzVolatilityModel()
        
        time_points, trajectory = lorenz.simulate(
            initial_state=np.array([0.0, 1.0, 1.05]),
            t_span=(0, 10),
            n_points=1000
        )
        
        assert len(time_points) == 1000
        assert trajectory.shape == (1000, 3)
        assert np.all(np.isfinite(trajectory))
    
    def test_volatility_mapping(self):
        """Test mapping Lorenz trajectory to volatility"""
        lorenz = LorenzVolatilityModel()
        
        _, trajectory = lorenz.simulate(t_span=(0, 10), n_points=100)
        volatility = lorenz.map_to_volatility(trajectory)
        
        assert len(volatility) == 100
        assert np.all(volatility > 0)
    
    def test_fit_to_returns(self):
        """Test fitting Lorenz model to returns"""
        lorenz = LorenzVolatilityModel()
        
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        
        volatility = lorenz.fit_to_returns(returns)
        
        assert len(volatility) == len(returns)
        assert np.all(volatility > 0)


class TestHMMRegimeDetector:
    
    def test_hmm_fitting(self):
        """Test HMM fitting"""
        hmm = HMMRegimeDetector(n_states=3)
        
        np.random.seed(42)
        returns = np.random.randn(200)
        
        hmm.fit(returns)
        
        assert hmm.model is not None
        assert hmm.model.n_components == 3
    
    def test_state_prediction(self):
        """Test state prediction"""
        hmm = HMMRegimeDetector(n_states=2)
        
        np.random.seed(42)
        returns = np.random.randn(200)
        
        hmm.fit(returns)
        states = hmm.predict_states(returns)
        
        assert len(states) == len(returns)
        assert np.all(states >= 0) and np.all(states < 2)
    
    def test_regime_volatility(self):
        """Test regime-adjusted volatility estimation"""
        hmm = HMMRegimeDetector(n_states=2)
        
        np.random.seed(42)
        returns = np.random.randn(200)
        
        hmm.fit(returns)
        regime_vol = hmm.estimate_regime_volatility(returns)
        
        assert len(regime_vol) == len(returns)
        assert np.all(regime_vol > 0)


class TestVolatilityEstimator:
    
    def test_initialization(self):
        """Test estimator initialization"""
        estimator = VolatilityEstimator()
        
        assert estimator.mst is not None
        assert estimator.lorenz is not None
        assert estimator.hmm is not None
    
    def test_volatility_estimation_synthetic(self):
        """Test volatility estimation with synthetic data"""
        estimator = VolatilityEstimator()
        
        np.random.seed(42)
        n_assets, n_periods = 5, 200
        returns = np.random.randn(n_assets, n_periods) * 0.02
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        volatility = estimator.estimate_volatility(returns, symbols)
        
        assert len(volatility) == len(symbols)
        assert all(vol > 0 for vol in volatility.values())
    
    def test_individual_volatilities(self):
        """Test individual volatility computation"""
        estimator = VolatilityEstimator()
        
        np.random.seed(42)
        returns = np.random.randn(5, 100) * 0.02
        
        individual_vols = estimator.compute_individual_volatilities(returns)
        
        assert len(individual_vols) == 5
        assert np.all(individual_vols > 0)
    
    def test_portfolio_volatility(self):
        """Test portfolio volatility estimation"""
        estimator = VolatilityEstimator()
        
        np.random.seed(42)
        returns = np.random.randn(3, 100) * 0.02
        
        portfolio_vol = estimator.estimate_portfolio_volatility(returns)
        
        assert portfolio_vol > 0
        
        weights = np.array([0.5, 0.3, 0.2])
        weights = np.array([0.5, 0.3, 0.2])
        portfolio_vol_weighted = estimator.estimate_portfolio_volatility(
            returns, weights=weights
        )
        
        assert portfolio_vol_weighted > 0
    
    def test_component_info(self):
        """Test getting component information"""
        estimator = VolatilityEstimator()
        
        np.random.seed(42)
        returns = np.random.randn(5, 200) * 0.02
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        estimator.estimate_volatility(returns, symbols)
        
        # Test correlation structure
        corr_info = estimator.get_correlation_structure()
        assert corr_info is not None
        assert 'mst_edges' in corr_info
        
        # Test regime info
        regime_info = estimator.get_regime_info()
        assert regime_info is not None
        assert 'current_state' in regime_info
        
        # Test Lorenz info
        lorenz_info = estimator.get_lorenz_info()
        assert lorenz_info is not None


class TestSectorDFSVolatility:
    
    def test_build_sector_hierarchy(self):
        """Test building sector hierarchy graph"""
        sector_dfs = SectorDFSVolatility()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC']
        sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'JPM': 'Financial Services',
            'BAC': 'Financial Services'
        }
        
        G = sector_dfs.build_sector_hierarchy(symbols, sectors)
        
        assert G.has_node("Market")
        assert G.has_node("Technology")
        assert G.has_node("Financial Services")
        assert G.number_of_nodes() == 8  # Market + 2 sectors + 5 stocks
    
    def test_dfs_sector_volatility(self):
        """Test DFS sector-relative volatility computation"""
        sector_dfs = SectorDFSVolatility()
        
        symbols = ['A', 'B', 'C', 'D']
        sectors = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Finance'}
        volatilities = {'A': 0.02, 'B': 0.04, 'C': 0.01, 'D': 0.03}
        
        sector_dfs.build_sector_hierarchy(symbols, sectors)
        results = sector_dfs.dfs_sector_volatility(volatilities)
        
        assert len(results) == 4
        assert results['A'].sector == 'Tech'
        assert results['A'].relative_volatility < 1.0
        assert results['B'].relative_volatility > 1.0
    
    def test_balanced_cross_sector(self):
        """Test balanced cross-sector comparison"""
        sector_dfs = SectorDFSVolatility()
        
        symbols = ['A', 'B', 'C', 'D']
        sectors = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Finance'}
        volatilities = {'A': 0.02, 'B': 0.04, 'C': 0.01, 'D': 0.03}
        
        sector_dfs.build_sector_hierarchy(symbols, sectors)
        results = sector_dfs.balanced_cross_sector_comparison(volatilities)
        
        assert len(results) == 4
        sectors_order = [r[1] for r in results]
        assert sectors_order[0] != sectors_order[1] or len(set(sectors_order)) == 1


class TestFinnhubClient:
    
    def test_client_initialization(self):
        """Test Finnhub client initialization"""
        from vol_estimator.api_clients import FinnhubClient
        
        client = FinnhubClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.min_call_interval == 1.0
    
    def test_sector_caching(self):
        """Test that sector data is cached"""
        from vol_estimator.api_clients import FinnhubClient
        
        client = FinnhubClient(api_key="test_key")
        client._sector_cache['AAPL'] = 'Technology'
        
        assert client.get_sector('AAPL') == 'Technology'


class TestCorrelationMSTDFS:
    
    def test_dfs_volatility_order(self):
        """Test DFS traversal order from MST"""
        mst = CorrelationMST()
        
        np.random.seed(42)
        returns = np.random.randn(5, 100)
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        mst.compute_correlation_matrix(returns, asset_names=symbols)
        mst.build_mst()
        
        dfs_order = mst.dfs_volatility_order()
        
        assert len(dfs_order) == 5
        assert set(dfs_order) == set(symbols)
    
    def test_dfs_correlation_path(self):
        """Test DFS correlation path"""
        mst = CorrelationMST()
        
        np.random.seed(42)
        returns = np.random.randn(5, 100)
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        mst.compute_correlation_matrix(returns, asset_names=symbols)
        mst.build_mst()
        
        dfs_path = mst.dfs_correlation_path()
        
        assert len(dfs_path) == 4
        assert all(len(edge) == 3 for edge in dfs_path)


class TestSectorDFSIntegration:
    
    def test_sector_relative_volatility(self):
        """Test sector-relative volatility estimation"""
        estimator = VolatilityEstimator(sector_dfs_enabled=True)
        
        # Mock sector data
        symbols = ['AAPL', 'MSFT', 'JPM', 'BAC']
        sectors = {'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Finance', 'BAC': 'Finance'}
        
        # Mock the get_sectors method to return our test data
        estimator.data_fetcher.get_sectors = lambda syms: sectors
        
        np.random.seed(42)
        returns = np.random.randn(4, 200) * 0.02
        
        sector_vols = estimator.estimate_sector_relative_volatility(returns, symbols)
        
        assert len(sector_vols) == 4
        assert all(isinstance(sv, SectorVolatility) for sv in sector_vols.values())
        assert all(sv.sector in ['Technology', 'Finance'] for sv in sector_vols.values())
    
    def test_get_sector_summary(self):
        """Test sector summary functionality"""
        estimator = VolatilityEstimator(sector_dfs_enabled=True)
        
        # Mock sector data
        symbols = ['A', 'B', 'C']
        sectors = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance'}
        estimator.data_fetcher.get_sectors = lambda syms: sectors
        
        np.random.seed(42)
        returns = np.random.randn(3, 100) * 0.02
        
        # First estimate sector-relative volatility
        estimator.estimate_sector_relative_volatility(returns, symbols)
        
        # Then get summary
        summary = estimator.get_sector_summary()
        
        assert summary is not None
        assert 'Tech' in summary
        assert 'Finance' in summary


class TestIntegration:
    
    def test_full_pipeline_synthetic(self):
        """Test full estimation pipeline with synthetic data"""
        estimator = VolatilityEstimator(
            mst_enabled=True,
            lorenz_enabled=True,
            hmm_enabled=True
        )
        
        np.random.seed(42)
        returns = np.random.randn(5, 200) * 0.02
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        volatility = estimator.estimate_volatility(returns, symbols)
        
        assert len(volatility) == 5
        assert all(vol > 0 for vol in volatility.values())
        
        # Verify all components were used
        assert estimator.get_correlation_structure() is not None
        assert estimator.get_regime_info() is not None
        assert estimator.get_lorenz_info() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


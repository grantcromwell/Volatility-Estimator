"""MST correlation analysis"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from numba import jit


def _compute_correlation_matrix_numba(returns: np.ndarray) -> np.ndarray:
    n_assets, n_periods = returns.shape
    correlation = np.zeros((n_assets, n_assets))
    
    # Compute correlation for each pair manually (Numba-compatible)
    for i in range(n_assets):
        for j in range(i, n_assets):
            # Compute means
            mean_i = 0.0
            mean_j = 0.0
            for k in range(n_periods):
                mean_i += returns[i, k]
                mean_j += returns[j, k]
            mean_i /= n_periods
            mean_j /= n_periods
            
            # Compute correlation
            cov = 0.0
            var_i = 0.0
            var_j = 0.0
            for k in range(n_periods):
                diff_i = returns[i, k] - mean_i
                diff_j = returns[j, k] - mean_j
                cov += diff_i * diff_j
                var_i += diff_i * diff_i
                var_j += diff_j * diff_j
            
            if var_i > 0 and var_j > 0:
                corr_val = cov / (np.sqrt(var_i) * np.sqrt(var_j))
            else:
                corr_val = 1.0 if i == j else 0.0
            
            correlation[i, j] = corr_val
            correlation[j, i] = corr_val
    
    return correlation


def _compute_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(returns)
    
    return correlation


class CorrelationMST:
    
    def __init__(self):
        self.mst_graph = None
        self.correlation_matrix = None
        self.asset_names = None
    
    def compute_correlation_matrix(self, returns: np.ndarray, asset_names: List[str] = None) -> np.ndarray:
        self.correlation_matrix = _compute_correlation_matrix(returns)
        
        if asset_names is None:
            self.asset_names = [f"Asset_{i}" for i in range(returns.shape[0])]
        else:
            self.asset_names = asset_names
            
        return self.correlation_matrix
    
    def build_mst(self, correlation_matrix: np.ndarray = None) -> nx.Graph:
        if correlation_matrix is None:
            if self.correlation_matrix is None:
                raise ValueError("No correlation matrix available. "
                               "Call compute_correlation_matrix first.")
            correlation_matrix = self.correlation_matrix
        
        n_assets = correlation_matrix.shape[0]
        
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        G = nx.Graph()
        
        for i in range(n_assets):
            node_name = self.asset_names[i] if self.asset_names else f"Asset_{i}"
            G.add_node(node_name)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                node_i = self.asset_names[i] if self.asset_names else f"Asset_{i}"
                node_j = self.asset_names[j] if self.asset_names else f"Asset_{j}"
                weight = distance_matrix[i, j]
                G.add_edge(node_i, node_j, weight=weight, correlation=correlation_matrix[i, j])
        
        self.mst_graph = nx.minimum_spanning_tree(G, algorithm='prim')
        self.mst_graph = nx.minimum_spanning_tree(G, algorithm='prim')
        
        return self.mst_graph
    
    def get_core_assets(self, top_n: int = None) -> List[str]:
        if self.mst_graph is None:
            raise ValueError("MST not built. Call build_mst first.")
        
        degrees = dict(self.mst_graph.degree())
        
        sorted_assets = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        if top_n is None:
            return [asset for asset, _ in sorted_assets]
        else:
            return [asset for asset, _ in sorted_assets[:top_n]]
    
    def get_mst_edges(self) -> List[Tuple[str, str, float]]:
        if self.mst_graph is None:
            raise ValueError("MST not built. Call build_mst first.")
        
        edges = []
        for u, v, data in self.mst_graph.edges(data=True):
            edges.append((u, v, data.get('correlation', 0.0)))
        
        return edges
    
    def compute_mst_volatility_weights(self, individual_volatilities: np.ndarray) -> np.ndarray:
        if self.mst_graph is None:
            raise ValueError("MST not built. Call build_mst first.")
        
        degrees = dict(self.mst_graph.degree())
        
        degree_values = np.array([degrees.get(name, 0) for name in self.asset_names])
        weights = degree_values / (degree_values.sum() + 1e-10)
        
        weighted_vol = individual_volatilities * weights
        
        return weighted_vol
    
    def dfs_volatility_order(self, start_asset: str = None) -> List[str]:
        if self.mst_graph is None:
            raise ValueError("MST not built. Call build_mst first.")
        
        if start_asset is None:
            degrees = dict(self.mst_graph.degree())
            start_asset = max(degrees, key=degrees.get)
        
        return list(nx.dfs_preorder_nodes(self.mst_graph, source=start_asset))

    def dfs_correlation_path(self, start_asset: str = None) -> List[Tuple[str, str, float]]:
        if self.mst_graph is None:
            raise ValueError("MST not built. Call build_mst first.")
        
        if start_asset is None:
            degrees = dict(self.mst_graph.degree())
            start_asset = max(degrees, key=degrees.get)
        
        edges = []
        for u, v in nx.dfs_edges(self.mst_graph, source=start_asset):
            corr = self.mst_graph[u][v].get('correlation', 0.0)
            edges.append((u, v, corr))
        
        return edges


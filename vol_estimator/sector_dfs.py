"""DFS sector volatility analysis"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SectorVolatility:
    symbol: str
    sector: str
    absolute_volatility: float
    sector_mean_volatility: float
    relative_volatility: float
    sector_rank: int


class SectorDFSVolatility:
    
    def __init__(self):
        self.sector_graph: Optional[nx.Graph] = None
        self.sectors: Dict[str, str] = {}
        self.sector_stocks: Dict[str, List[str]] = {}
        self.sector_volatilities: Dict[str, SectorVolatility] = {}
    
    def build_sector_hierarchy(self, symbols: List[str], sectors: Dict[str, str], correlations: Optional[np.ndarray] = None) -> nx.Graph:
        G = nx.Graph()
        
        G.add_node("Market", node_type="root")
        
        self.sector_stocks = {}
        for symbol in symbols:
            sector = sectors.get(symbol, "Unknown")
            if sector not in self.sector_stocks:
                self.sector_stocks[sector] = []
            self.sector_stocks[sector].append(symbol)
        
        for sector, stocks in self.sector_stocks.items():
            G.add_node(sector, node_type="sector")
            G.add_edge("Market", sector, weight=0)
            
            for i, stock in enumerate(stocks):
                G.add_node(stock, node_type="stock", sector=sector)
                G.add_edge(sector, stock, weight=0)
                
                if correlations is not None:
                    for j, other in enumerate(stocks[i+1:], i+1):
                        try:
                            sym_i = symbols.index(stock)
                            sym_j = symbols.index(other)
                            corr = correlations[sym_i, sym_j]
                            G.add_edge(stock, other, weight=1-abs(corr), correlation=corr)
                        except (ValueError, IndexError):
                            pass
        
        self.sector_graph = G
        self.sectors = sectors
        return G
    
    def dfs_sector_volatility(self, volatilities: Dict[str, float]) -> Dict[str, SectorVolatility]:
        if self.sector_graph is None:
            raise ValueError("Sector hierarchy not built. Call build_sector_hierarchy first.")
        
        results = {}
        
        for node in nx.dfs_preorder_nodes(self.sector_graph, source="Market"):
            if self.sector_graph.nodes[node].get('node_type') != 'sector':
                continue
            
            sector = node
            
            sector_stock_list = self.sector_stocks.get(sector, [])
            
            sector_vols = [volatilities.get(s, 0) for s in sector_stock_list if s in volatilities]
            sector_mean = np.mean(sector_vols) if sector_vols else 0
            
            sorted_stocks = sorted(sector_stock_list, key=lambda s: volatilities.get(s, 0), reverse=True)
            
            for rank, stock in enumerate(sorted_stocks, 1):
                if stock in volatilities:
                    vol = volatilities[stock]
                    results[stock] = SectorVolatility(
                        symbol=stock,
                        sector=sector,
                        absolute_volatility=vol,
                        sector_mean_volatility=sector_mean,
                        relative_volatility=vol / sector_mean if sector_mean > 0 else 1.0,
                        sector_rank=rank
                    )
        
        self.sector_volatilities = results
        return results
    
    def balanced_cross_sector_comparison(self, volatilities: Dict[str, float]) -> List[Tuple[str, str, float, float]]:
        if self.sector_graph is None:
            raise ValueError("Sector hierarchy not built.")
        
        sectors = [n for n in self.sector_graph.nodes 
                   if self.sector_graph.nodes[n].get('node_type') == 'sector']
        
        sector_means = {}
        for sector in sectors:
            stocks = self.sector_stocks.get(sector, [])
            vols = [volatilities.get(s, 0) for s in stocks if s in volatilities]
            sector_means[sector] = np.mean(vols) if vols else 1.0
        
        sector_iters = {
            sector: iter(self.sector_stocks.get(sector, []))
            for sector in sectors
        }
        
        results = []
        active_sectors = set(sectors)
        
        while active_sectors:
            for sector in list(active_sectors):
                try:
                    stock = next(sector_iters[sector])
                    if stock in volatilities:
                        vol = volatilities[stock]
                        rel_vol = vol / sector_means[sector]
                        results.append((stock, sector, vol, rel_vol))
                except StopIteration:
                    active_sectors.discard(sector)
        
        return results
    
    def get_sector_summary(self) -> Dict[str, Dict]:
        if not self.sector_volatilities:
            return {}
        
        summary = {}
        for sector, stocks in self.sector_stocks.items():
            sector_results = [self.sector_volatilities.get(s) for s in stocks 
                            if s in self.sector_volatilities]
            
            if not sector_results:
                continue
            
            vols = [r.absolute_volatility for r in sector_results]
            sorted_by_vol = sorted(sector_results, key=lambda x: x.absolute_volatility, reverse=True)
            
            summary[sector] = {
                'mean_volatility': np.mean(vols),
                'std_volatility': np.std(vols),
                'count': len(vols),
                'top_volatile': sorted_by_vol[0].symbol if sorted_by_vol else None,
                'lowest_volatile': sorted_by_vol[-1].symbol if sorted_by_vol else None
            }
        
        return summary

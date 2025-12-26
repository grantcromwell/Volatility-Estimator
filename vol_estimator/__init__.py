"""
Volatility Estimator Package

A comprehensive volatility estimation system combining:
- Prim's Minimum Spanning Tree for correlation analysis
- Lorenz Attractor for chaotic dynamics
- Hidden Markov Model for regime detection
"""

from .estimator import VolatilityEstimator
from .correlation_mst import CorrelationMST
from .lorenz_attractor import LorenzVolatilityModel
from .hmm_regime import HMMRegimeDetector
from .api_clients import FinancialDataFetcher
from .sector_dfs import SectorDFSVolatility, SectorVolatility

__version__ = "0.1.0"
__all__ = [
    "VolatilityEstimator",
    "CorrelationMST",
    "LorenzVolatilityModel",
    "HMMRegimeDetector",
    "FinancialDataFetcher",
    "SectorDFSVolatility",
    "SectorVolatility",
]


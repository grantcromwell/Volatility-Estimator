"""
Volatility Estimator Package

A comprehensive volatility estimation system combining:
- Industry-standard models (GARCH, EWMA)
- Advanced techniques (HMM, MST, Lorenz Attractor)
- Range-based estimators (Garman-Klass, Parkinson, Yang-Zhang)
- Validation and backtesting framework
- Structured logging and monitoring
"""

from .estimator import VolatilityEstimator
from .correlation_mst import CorrelationMST
from .lorenz_attractor import LorenzVolatilityModel
from .hmm_regime import HMMRegimeDetector
from .api_clients import FinancialDataFetcher
from .sector_dfs import SectorDFSVolatility, SectorVolatility
from .ewma_volatility import EWMAVolatilityEstimator, MultiAssetEWMAEstimator
from .range_estimators import (
    ParkinsonVolatilityEstimator,
    GarmanKlassVolatilityEstimator,
    RogersSatchellVolatilityEstimator,
    YangZhangVolatilityEstimator,
    compare_estimators
)
from .garch_volatility import (
    GARCHVolatilityEstimator,
    EGARCHVolatilityEstimator,
    GJRGARCHVolatilityEstimator,
    SimpleGARCH11
)
from .validation import (
    ValidationMetrics,
    BacktestResult,
    VolatilityBacktester,
    ModelValidator,
    calculate_accuracy_metrics,
    compute_realized_volatility
)
from .logging_config import get_logger, MetricsLogger

__version__ = "1.0.0"
__all__ = [
    # Core estimator
    "VolatilityEstimator",
    
    # Advanced models
    "CorrelationMST",
    "LorenzVolatilityModel",
    "HMMRegimeDetector",
    
    # Industry-standard models
    "EWMAVolatilityEstimator",
    "MultiAssetEWMAEstimator",
    "GARCHVolatilityEstimator",
    "EGARCHVolatilityEstimator",
    "GJRGARCHVolatilityEstimator",
    "SimpleGARCH11",
    
    # Range-based estimators
    "ParkinsonVolatilityEstimator",
    "GarmanKlassVolatilityEstimator",
    "RogersSatchellVolatilityEstimator",
    "YangZhangVolatilityEstimator",
    "compare_estimators",
    
    # Validation framework
    "ValidationMetrics",
    "BacktestResult",
    "VolatilityBacktester",
    "ModelValidator",
    "calculate_accuracy_metrics",
    "compute_realized_volatility",
    
    # Logging
    "get_logger",
    "MetricsLogger",
    
    # Data fetching and sector analysis
    "FinancialDataFetcher",
    "SectorDFSVolatility",
    "SectorVolatility",
]


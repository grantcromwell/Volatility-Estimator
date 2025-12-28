"""
Logging configuration for Volatility Estimator.

Provides structured logging with configurable levels, handlers, and formatters
for monitoring and debugging.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path


class VolatilityLogger:
    """
    Centralized logging configuration for the volatility estimator package.
    
    Provides:
    - Structured log formatting
    - Configurable log levels
    - File and console handlers
    - Performance metrics logging
    - Error tracking with context
    """
    
    _instance: Optional['VolatilityLogger'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = "vol_estimator",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        enable_console: bool = True
    ):
        """
        Initialize the volatility logger.
        
        Args:
            name: Logger name (default: vol_estimator)
            level: Logging level (default: INFO)
            log_file: Optional path to log file
            enable_console: Whether to log to console (default: True)
        """
        if VolatilityLogger._initialized:
            return
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create formatter
        self.formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        
        VolatilityLogger._initialized = True
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger
    
    @classmethod
    def reset(cls):
        """Reset the singleton for testing purposes."""
        cls._instance = None
        cls._initialized = False


def get_logger(name: str = "vol_estimator") -> logging.Logger:
    """
    Get a logger instance for the volatility estimator.
    
    Args:
        name: Logger name (can be module-specific)
    
    Returns:
        Configured logger instance
    """
    vol_logger = VolatilityLogger(name=name)
    return vol_logger.get_logger()


class MetricsLogger:
    """
    Specialized logger for tracking volatility estimation metrics.
    
    Tracks:
    - Model performance metrics (MAE, RMSE, MAPE)
    - Execution times
    - Model parameters and configurations
    - Validation results
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("vol_estimator.metrics")
        self._metrics_history = []
    
    def log_estimation(
        self,
        model_name: str,
        n_assets: int,
        n_periods: int,
        execution_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log a volatility estimation event."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "n_assets": n_assets,
            "n_periods": n_periods,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "error": error_message
        }
        self._metrics_history.append(metric)
        
        if success:
            self.logger.info(
                f"Estimation complete | model={model_name} | "
                f"assets={n_assets} | periods={n_periods} | "
                f"time={execution_time_ms:.2f}ms"
            )
        else:
            self.logger.error(
                f"Estimation failed | model={model_name} | "
                f"assets={n_assets} | periods={n_periods} | "
                f"error={error_message}"
            )
    
    def log_accuracy(
        self,
        model_name: str,
        mae: float,
        rmse: float,
        mape: float,
        n_samples: int
    ):
        """Log accuracy metrics for a volatility model."""
        self.logger.info(
            f"Accuracy metrics | model={model_name} | "
            f"MAE={mae:.6f} | RMSE={rmse:.6f} | MAPE={mape:.2f}% | "
            f"samples={n_samples}"
        )
    
    def log_model_config(self, model_name: str, config: dict):
        """Log model configuration."""
        config_str = " | ".join(f"{k}={v}" for k, v in config.items())
        self.logger.info(f"Model config | {model_name} | {config_str}")
    
    def log_validation(
        self,
        model_name: str,
        validation_type: str,
        passed: bool,
        details: Optional[str] = None
    ):
        """Log validation results."""
        status = "PASSED" if passed else "FAILED"
        msg = f"Validation {status} | model={model_name} | type={validation_type}"
        if details:
            msg += f" | {details}"
        
        if passed:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)
    
    def get_metrics_history(self) -> list:
        """Get all logged metrics."""
        return self._metrics_history.copy()


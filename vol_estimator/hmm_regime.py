"""
Hidden Markov Model for Volatility Regime Detection

Uses HMM to identify hidden volatility states (e.g., low, medium, high volatility),
capturing regime-switching behavior in financial time series.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class HMMRegimeDetector:
    """
    Hidden Markov Model for volatility regime detection.
    
    Identifies hidden states corresponding to different volatility regimes,
    enabling regime-aware volatility estimation.
    """
    
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = 'diag',
        n_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize HMM regime detector.
        
        Args:
            n_states: Number of hidden states (typically 2-4 for volatility)
            covariance_type: Type of covariance matrix ('diag', 'full', 'spherical')
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.hidden_states = None
        self.state_probabilities = None
    
    def fit(self, returns: np.ndarray) -> 'HMMRegimeDetector':
        """
        Fit HMM to return series.
        
        Args:
            returns: Array of returns (1D or 2D with shape (n_samples, 1))
        
        Returns:
            Self for method chaining
        """
        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(returns)
        
        return self
    
    def predict_states(self, returns: np.ndarray) -> np.ndarray:
        """
        Predict hidden states for given returns.
        
        Args:
            returns: Array of returns (1D or 2D)
        
        Returns:
            Array of predicted state labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        self.hidden_states = self.model.predict(returns)
        
        return self.hidden_states
    
    def predict_state_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities for each observation.
        
        Args:
            returns: Array of returns (1D or 2D)
        
        Returns:
            Array of shape (n_samples, n_states) with state probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        self.state_probabilities = self.model.predict_proba(returns)
        
        return self.state_probabilities
    
    def get_state_volatilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute volatility for each state.
        
        Args:
            returns: Array of returns used for fitting
        
        Returns:
            Array of volatilities for each state
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.hidden_states is None:
            self.predict_states(returns)
        
        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        state_volatilities = []
        for state in range(self.n_states):
            state_mask = self.hidden_states == state
            if np.any(state_mask):
                # Compute volatility (std) for this state
                state_returns = returns[state_mask, 0]
                state_vol = np.std(state_returns)
                state_volatilities.append(state_vol)
            else:
                # If state not observed, use model's variance
                state_vol = np.sqrt(self.model.covars_[state, 0, 0])
                state_volatilities.append(state_vol)
        
        return np.array(state_volatilities)
    
    def estimate_regime_volatility(
        self, 
        returns: np.ndarray,
        use_probabilities: bool = True
    ) -> np.ndarray:
        """
        Estimate volatility using regime information.
        
        Args:
            returns: Array of returns
            use_probabilities: If True, use state probabilities for weighted average.
                             If False, use most likely state.
        
        Returns:
            Array of regime-adjusted volatility estimates
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure 2D array
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Get state volatilities
        state_vols = self.get_state_volatilities(returns)
        
        if use_probabilities:
            # Use probability-weighted volatility
            if self.state_probabilities is None:
                self.predict_state_probabilities(returns)
            
            # Weighted average: sum(state_prob * state_vol)
            regime_vol = np.dot(self.state_probabilities, state_vols)
        else:
            # Use most likely state
            if self.hidden_states is None:
                self.predict_states(returns)
            
            regime_vol = state_vols[self.hidden_states]
        
        return regime_vol
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get state transition probability matrix.
        
        Returns:
            Transition matrix of shape (n_states, n_states)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.transmat_
    
    def get_initial_state_probabilities(self) -> np.ndarray:
        """
        Get initial state probabilities.
        
        Returns:
            Initial state probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.startprob_
    
    def get_state_means(self) -> np.ndarray:
        """
        Get mean return for each state.
        
        Returns:
            Array of means for each state
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.means_.flatten()
    
    def detect_regime_changes(
        self, 
        returns: np.ndarray,
        min_duration: int = 5
    ) -> List[Tuple[int, int, int]]:
        """
        Detect regime change points.
        
        Args:
            returns: Array of returns
            min_duration: Minimum duration (in periods) for a regime
        
        Returns:
            List of tuples (start_idx, end_idx, state) for each regime
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.hidden_states is None:
            self.predict_states(returns)
        
        regimes = []
        current_state = self.hidden_states[0]
        start_idx = 0
        
        for i in range(1, len(self.hidden_states)):
            if self.hidden_states[i] != current_state:
                # Regime change detected
                if i - start_idx >= min_duration:
                    regimes.append((start_idx, i - 1, current_state))
                start_idx = i
                current_state = self.hidden_states[i]
        
        # Add final regime
        if len(self.hidden_states) - start_idx >= min_duration:
            regimes.append((start_idx, len(self.hidden_states) - 1, current_state))
        
        return regimes


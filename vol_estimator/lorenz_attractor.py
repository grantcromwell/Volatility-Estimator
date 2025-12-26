"""
Lorenz Attractor for Volatility Dynamics Modeling

Models volatility as a chaotic system using the Lorenz attractor,
capturing non-linear dynamics and regime transitions.
CPU-optimized with NumPy and SciPy.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
from numba import jit


@jit(nopython=True, cache=True)
def lorenz_system(t: float, state: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    """
    Lorenz system of differential equations.
    
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    
    Args:
        t: Time (not used but required by ODE solver interface)
        state: Current state [x, y, z]
        sigma: Prandtl number (default 10)
        beta: Geometric factor (default 8/3)
        rho: Rayleigh number (default 28)
    
    Returns:
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state[0], state[1], state[2]
    
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    return np.array([dxdt, dydt, dzdt])


class LorenzVolatilityModel:
    """
    Lorenz Attractor model for volatility dynamics.
    
    Uses the Lorenz system to model volatility as a chaotic system,
    capturing regime transitions and non-linear dynamics.
    """
    
    def __init__(
        self, 
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
        rho: float = 28.0
    ):
        """
        Initialize Lorenz volatility model.
        
        Args:
            sigma: Prandtl number (controls rate of convergence)
            beta: Geometric factor (controls dissipation)
            rho: Rayleigh number (controls system behavior, chaos threshold ~24.74)
        """
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.trajectory = None
        self.time_points = None
    
    def simulate(
        self, 
        initial_state: np.ndarray = None,
        t_span: Tuple[float, float] = (0, 50),
        n_points: int = 10000,
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Lorenz attractor trajectory.
        
        Args:
            initial_state: Initial [x, y, z]. If None, uses [0, 1, 1.05]
            t_span: Time span (t_start, t_end)
            n_points: Number of time points
            t_eval: Optional specific time points to evaluate
        
        Returns:
            Tuple of (time_points, trajectory) where trajectory is (n_points, 3)
        """
        if initial_state is None:
            initial_state = np.array([0.0, 1.0, 1.05])
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Solve ODE system
        solution = solve_ivp(
            lambda t, y: lorenz_system(t, y, self.sigma, self.beta, self.rho),
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        self.time_points = solution.t
        self.trajectory = solution.y.T  # Shape: (n_points, 3)
        
        return self.time_points, self.trajectory
    
    def map_to_volatility(
        self, 
        trajectory: np.ndarray = None,
        volatility_scale: float = 0.02,
        volatility_base: float = 0.15
    ) -> np.ndarray:
        """
        Map Lorenz trajectory to volatility values.
        
        Uses the z-component (vertical) as primary volatility indicator,
        with normalization and scaling.
        
        Args:
            trajectory: Lorenz trajectory (n_points, 3). If None, uses stored trajectory
            volatility_scale: Scaling factor for volatility range
            volatility_base: Base volatility level
        
        Returns:
            Array of volatility values
        """
        if trajectory is None:
            if self.trajectory is None:
                raise ValueError("No trajectory available. Call simulate first.")
            trajectory = self.trajectory
        
        # Use z-component (vertical) as volatility proxy
        z_values = trajectory[:, 2]
        
        # Normalize z to [0, 1] range
        z_min, z_max = z_values.min(), z_values.max()
        if z_max > z_min:
            z_normalized = (z_values - z_min) / (z_max - z_min)
        else:
            z_normalized = np.ones_like(z_values) * 0.5
        
        # Map to volatility: base + scaled variation
        volatility = volatility_base + volatility_scale * (z_normalized - 0.5) * 2
        
        # Ensure positive volatility
        volatility = np.maximum(volatility, 0.01)
        
        return volatility
    
    def fit_to_returns(
        self, 
        returns: np.ndarray,
        t_span: Tuple[float, float] = (0, 50),
        n_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit Lorenz model to return series and extract volatility.
        
        Args:
            returns: Array of returns
            t_span: Time span for simulation
            n_points: Number of simulation points. If None, matches returns length
        
        Returns:
            Estimated volatility from Lorenz model
        """
        if n_points is None:
            n_points = len(returns)
        
        # Simulate Lorenz system
        _, trajectory = self.simulate(t_span=t_span, n_points=n_points)
        
        # Map to volatility
        volatility = self.map_to_volatility(trajectory)
        
        # Resample to match returns length if needed
        if len(volatility) != len(returns):
            indices = np.linspace(0, len(volatility) - 1, len(returns), dtype=int)
            volatility = volatility[indices]
        
        return volatility
    
    def detect_regime_transitions(
        self, 
        trajectory: np.ndarray = None,
        threshold: float = 20.0
    ) -> np.ndarray:
        """
        Detect regime transitions based on z-component threshold.
        
        Lorenz attractor has two "wings" - transitions occur when
        z crosses certain thresholds.
        
        Args:
            trajectory: Lorenz trajectory. If None, uses stored trajectory
            threshold: Z-value threshold for regime detection
        
        Returns:
            Boolean array indicating regime transitions
        """
        if trajectory is None:
            if self.trajectory is None:
                raise ValueError("No trajectory available. Call simulate first.")
            trajectory = self.trajectory
        
        z_values = trajectory[:, 2]
        
        # Detect transitions: when z crosses threshold
        above_threshold = z_values > threshold
        transitions = np.diff(above_threshold.astype(int)) != 0
        
        # Pad to match original length
        transitions = np.concatenate([[False], transitions])
        
        return transitions
    
    def compute_lyapunov_exponent(
        self, 
        trajectory: np.ndarray = None,
        window: int = 100
    ) -> float:
        """
        Estimate Lyapunov exponent from trajectory.
        
        Lyapunov exponent measures sensitivity to initial conditions,
        indicating chaotic behavior (positive exponent).
        
        Args:
            trajectory: Lorenz trajectory. If None, uses stored trajectory
            window: Window size for local estimation
        
        Returns:
            Estimated Lyapunov exponent
        """
        if trajectory is None:
            if self.trajectory is None:
                raise ValueError("No trajectory available. Call simulate first.")
            trajectory = self.trajectory
        
        # Compute local divergence rates
        n_points = len(trajectory)
        if n_points < window:
            return 0.0
        
        divergences = []
        for i in range(n_points - window):
            segment = trajectory[i:i+window]
            # Compute average distance growth
            distances = np.linalg.norm(np.diff(segment, axis=0), axis=1)
            if len(distances) > 1 and distances[0] > 0:
                growth_rate = np.mean(np.log(distances[1:] / distances[:-1] + 1e-10))
                divergences.append(growth_rate)
        
        if len(divergences) > 0:
            return np.mean(divergences)
        else:
            return 0.0


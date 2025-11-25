"""
Temporal Feedback Buffer
========================

Exponentially-weighted history buffer for adaptive hyperparameter adjustment.

This module maintains a rolling history of training metrics and computes
a weighted contribution signal that can be used to adjust hyperparameters
(particularly the amplitude α of the periodic LR scheduler).

Mathematical Formulation:
-------------------------
Given histories of β (loss variance), ω (frequency), and φ (phase) values,
the temporal feedback gain at time t is computed as:

    gain(t) = (1/W) · Σᵢ βᵢ · sin(ωᵢ·t + φᵢ) · exp(-λ·(t-i))

where:
    - W is the window size (number of recent steps to consider)
    - βᵢ, ωᵢ, φᵢ are the values at step i
    - λ is the exponential decay rate
    - The sum is over i ∈ [max(0, t-W+1), t]

The gain is then used to adjust the LR scheduler's amplitude:

    α' = clamp(α + push · max(0, gain), α_min, α_max)

Interpretation:
---------------
- Positive gain: Training dynamics are stable and "in resonance"
  → Increase amplitude for faster progress
  
- Zero/negative gain: Training dynamics are unstable or oscillating
  → Keep amplitude unchanged

The exponential decay (λ) ensures that recent observations have more
influence than older ones, allowing the system to adapt to changing
training dynamics.

Typical Parameters:
-------------------
- window: 12 (covers roughly one full period at ω≈6.0)
- decay (λ): 0.05 (half-life ≈ 14 steps)
- push: 0.01 (small adjustments)
- α bounds: [0.03, 0.12]

Integration Example:
--------------------
    from atf.modules.temporal_buffer import TemporalFeedbackBuffer
    
    buffer = TemporalFeedbackBuffer(window=12, decay=0.05)
    
    for epoch in range(epochs):
        # ... training ...
        
        # Record current state
        buffer.record(beta, omega, phi)
        
        # Compute gain and adjust amplitude
        gain = buffer.compute_gain()
        if gain > 0 and scheduler is not None:
            new_alpha = buffer.suggest_alpha(scheduler.alpha, gain)
            scheduler.set_params(alpha=new_alpha)

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import math
from typing import List, Sequence, Optional


def compute_temporal_gain(
    beta_history: Sequence[float],
    omega_history: Sequence[float],
    phi_history: Sequence[float],
    window: int = 12,
    decay: float = 0.05,
) -> float:
    """
    Compute temporal feedback gain from metric histories.
    
    This function computes a weighted sum of periodic contributions
    from the recent history of training metrics. The gain indicates
    whether the training dynamics are stable and can benefit from
    increased exploration.
    
    Args:
        beta_history: Sequence of loss variance estimates
        omega_history: Sequence of frequency values
        phi_history: Sequence of phase values
        window: Number of recent steps to consider (default: 12)
        decay: Exponential decay rate λ (default: 0.05)
    
    Returns:
        Temporal gain value (can be positive or negative)
        Returns 0.0 if history is empty
    
    Example:
        >>> betas = [0.1, 0.08, 0.06, 0.05, 0.04]
        >>> omegas = [6.0] * 5
        >>> phis = [1.05] * 5
        >>> gain = compute_temporal_gain(betas, omegas, phis)
        >>> print(f"Gain: {gain:.4f}")
    """
    n = min(len(beta_history), len(omega_history), len(phi_history))
    if n == 0:
        return 0.0
    
    t = n - 1  # Current time index
    w = max(1, int(window))
    start = max(0, t - w + 1)
    
    accumulator = 0.0
    count = 0
    
    for i in range(start, t + 1):
        beta = float(beta_history[i])
        omega = float(omega_history[i])
        phi = float(phi_history[i])
        
        # Periodic contribution at current time t
        periodic = math.sin(omega * t + phi)
        
        # Exponential decay for older observations
        time_decay = math.exp(-decay * (t - i))
        
        accumulator += beta * periodic * time_decay
        count += 1
    
    if count == 0:
        return 0.0
    
    return accumulator / float(count)


def suggest_amplitude_adjustment(
    current_alpha: float,
    gain: float,
    push: float = 0.01,
    alpha_min: float = 0.03,
    alpha_max: float = 0.12
) -> float:
    """
    Suggest new amplitude value based on temporal gain.
    
    Only increases amplitude when gain is positive. The adjustment
    is bounded to prevent extreme values.
    
    Args:
        current_alpha: Current amplitude value
        gain: Temporal feedback gain
        push: Adjustment strength (default: 0.01)
        alpha_min: Minimum allowed amplitude (default: 0.03)
        alpha_max: Maximum allowed amplitude (default: 0.12)
    
    Returns:
        Suggested new amplitude value in [alpha_min, alpha_max]
    """
    # Ensure valid input
    if not math.isfinite(current_alpha):
        current_alpha = (alpha_min + alpha_max) * 0.5
    
    # Only push when gain is positive (stable dynamics)
    if gain > 0.0:
        new_alpha = current_alpha + push * gain
    else:
        new_alpha = current_alpha
    
    # Clamp to valid range
    return max(alpha_min, min(alpha_max, new_alpha))


class TemporalFeedbackBuffer:
    """
    Stateful buffer for temporal feedback computation.
    
    This class maintains histories of training metrics and provides
    methods for computing the temporal gain and suggesting parameter
    adjustments.
    
    Parameters:
        window: History window size (default: 12)
        decay: Exponential decay rate (default: 0.05)
        push: Amplitude adjustment strength (default: 0.01)
        alpha_min: Minimum amplitude (default: 0.03)
        alpha_max: Maximum amplitude (default: 0.12)
    
    Example:
        >>> buffer = TemporalFeedbackBuffer()
        >>> for epoch in range(100):
        ...     beta = compute_beta(...)
        ...     omega, phi = scheduler.omega, scheduler.phi
        ...     buffer.record(beta, omega, phi)
        ...     gain = buffer.compute_gain()
        ...     if gain > 0:
        ...         new_alpha = buffer.suggest_alpha(scheduler.alpha, gain)
        ...         scheduler.set_params(alpha=new_alpha)
    """
    
    def __init__(
        self,
        window: int = 12,
        decay: float = 0.05,
        push: float = 0.01,
        alpha_min: float = 0.03,
        alpha_max: float = 0.12,
        max_history: int = 1000
    ):
        self.window = int(window)
        self.decay = float(decay)
        self.push = float(push)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.max_history = int(max_history)
        
        # Histories
        self._beta_history: List[float] = []
        self._omega_history: List[float] = []
        self._phi_history: List[float] = []
    
    def record(self, beta: float, omega: float, phi: float) -> None:
        """
        Record current metric values.
        
        Args:
            beta: Current loss variance estimate
            omega: Current frequency value
            phi: Current phase value
        """
        self._beta_history.append(float(beta))
        self._omega_history.append(float(omega))
        self._phi_history.append(float(phi))
        
        # Prevent unbounded growth
        if len(self._beta_history) > self.max_history:
            trim = len(self._beta_history) - self.max_history
            self._beta_history = self._beta_history[trim:]
            self._omega_history = self._omega_history[trim:]
            self._phi_history = self._phi_history[trim:]
    
    def compute_gain(self) -> float:
        """
        Compute temporal feedback gain from current history.
        
        Returns:
            Gain value (positive = stable dynamics)
        """
        return compute_temporal_gain(
            self._beta_history,
            self._omega_history,
            self._phi_history,
            window=self.window,
            decay=self.decay
        )
    
    def suggest_alpha(
        self,
        current_alpha: float,
        gain: Optional[float] = None
    ) -> float:
        """
        Suggest new amplitude value.
        
        Args:
            current_alpha: Current amplitude
            gain: Pre-computed gain (computes if None)
            
        Returns:
            Suggested new amplitude
        """
        if gain is None:
            gain = self.compute_gain()
        
        return suggest_amplitude_adjustment(
            current_alpha,
            gain,
            push=self.push,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max
        )
    
    @property
    def beta_history(self) -> List[float]:
        """Copy of beta history."""
        return self._beta_history.copy()
    
    @property
    def omega_history(self) -> List[float]:
        """Copy of omega history."""
        return self._omega_history.copy()
    
    @property
    def phi_history(self) -> List[float]:
        """Copy of phi history."""
        return self._phi_history.copy()
    
    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self._beta_history)
    
    def reset(self) -> None:
        """Clear all histories."""
        self._beta_history.clear()
        self._omega_history.clear()
        self._phi_history.clear()
    
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'window': self.window,
            'decay': self.decay,
            'push': self.push,
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'beta_history': self._beta_history.copy(),
            'omega_history': self._omega_history.copy(),
            'phi_history': self._phi_history.copy(),
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.window = state.get('window', self.window)
        self.decay = state.get('decay', self.decay)
        self.push = state.get('push', self.push)
        self.alpha_min = state.get('alpha_min', self.alpha_min)
        self.alpha_max = state.get('alpha_max', self.alpha_max)
        self._beta_history = state.get('beta_history', [])
        self._omega_history = state.get('omega_history', [])
        self._phi_history = state.get('phi_history', [])


# Backward compatibility aliases
trp_gain = compute_temporal_gain
apply_alpha_trp_push = suggest_amplitude_adjustment

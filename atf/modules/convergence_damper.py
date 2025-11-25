"""
Convergence Damper
==================

Learning rate damping mechanism for stable convergence.

This module implements adaptive learning rate reduction when the model
approaches convergence. The core idea is that when the loss variance
becomes very small (indicating stability near a minimum), the learning
rate should be reduced to prevent oscillation around the optimum.

Mathematical Formulation:
-------------------------
The damper uses a Gaussian profile based on the loss variance (β):

    damping_factor = 1 - α · exp(-β² / ε)

where:
    - β is the estimated loss variance (standard deviation)
    - α is the maximum damping strength (default: 0.40)
    - ε is a small constant for numerical stability (default: 1e-4)

The damped learning rate is:

    lr' = lr · damping_factor

Behavior:
---------
- When β ≈ 0 (near convergence): damping_factor ≈ 1 - α ≈ 0.60
  → Learning rate reduced by ~40%
  
- When β >> ε (high variance): damping_factor ≈ 1.0
  → Learning rate unchanged

This creates a smooth transition that progressively reduces the
learning rate as the model stabilizes, without abrupt changes.

Use Case:
---------
This mechanism is particularly useful in the late stages of training
when:
1. The model is oscillating around a local minimum
2. Fine-tuning requires smaller steps
3. Validation loss shows high-frequency noise

The damper should be activated only when β falls below a threshold
(typically 0.008-0.01), allowing normal training dynamics otherwise.

Integration Example:
--------------------
    from atf.modules.convergence_damper import apply_convergence_damping
    
    # In training loop, after computing β (loss variance estimate)
    if beta < beta_threshold:
        for param_group in optimizer.param_groups:
            param_group['lr'] = apply_convergence_damping(
                param_group['lr'], 
                beta, 
                alpha=0.40
            )

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import math
from typing import Optional


def apply_convergence_damping(
    lr: float,
    beta: float,
    alpha: float = 0.40,
    epsilon: float = 1e-4
) -> float:
    """
    Apply convergence-based learning rate damping.
    
    This function reduces the learning rate when the loss variance (β)
    is small, indicating that the model is near convergence. The
    damping follows a Gaussian profile centered at β=0.
    
    Mathematical formulation:
        damping = 1 - α · exp(-β² / ε)
        lr' = lr · damping
    
    Args:
        lr: Current learning rate
        beta: Estimated loss variance (standard deviation)
              Should be non-negative
        alpha: Maximum damping factor (default: 0.40)
               When β=0, lr is reduced by this fraction
               Recommended range: [0.20, 0.50]
        epsilon: Numerical stability constant (default: 1e-4)
                 Controls the width of the Gaussian profile
    
    Returns:
        Damped learning rate (always positive, minimum 1e-8)
    
    Examples:
        >>> apply_convergence_damping(0.001, beta=0.0, alpha=0.40)
        0.0006  # 40% reduction when β=0
        
        >>> apply_convergence_damping(0.001, beta=0.1, alpha=0.40)
        0.001   # No reduction when β is large
    
    Note:
        This function should only be called when β is below a threshold
        (e.g., 0.008). The caller is responsible for this gating logic.
    """
    # Handle invalid inputs
    if not math.isfinite(lr) or lr <= 0.0:
        return 1e-8
    
    beta_squared = beta * beta
    
    # Gaussian profile: peaks at β=0, decays for larger β
    # s ≈ 1.0 when β ≈ 0
    # s → 0 when β >> sqrt(ε)
    gaussian_factor = math.exp(-beta_squared / (epsilon + 1e-12))
    
    # Damping factor: 
    # = 1 - α when β ≈ 0 (maximum damping)
    # → 1 when β is large (no damping)
    damping = 1.0 - alpha * gaussian_factor
    
    # Safety checks
    if not math.isfinite(damping):
        damping = 1.0
    
    # Ensure damping is in valid range [0, 1]
    damping = max(0.0, min(1.0, damping))
    
    # Apply damping
    new_lr = lr * damping
    
    # Ensure minimum learning rate
    return max(new_lr, 1e-8)


class ConvergenceDamper:
    """
    Stateful convergence damper with automatic threshold detection.
    
    This class wraps the damping function and maintains state for
    tracking when damping should be active. It provides additional
    features like automatic beta estimation from loss history.
    
    Parameters:
        alpha: Maximum damping factor (default: 0.40)
        beta_threshold: Activate damping when β < threshold (default: 0.008)
        epsilon: Numerical stability constant (default: 1e-4)
        ema_alpha: Smoothing factor for beta estimation (default: 0.3)
    
    Example:
        >>> damper = ConvergenceDamper(alpha=0.40, beta_threshold=0.008)
        >>> for epoch in range(epochs):
        ...     train_loss = train_epoch(...)
        ...     val_loss = validate(...)
        ...     
        ...     # Update damper and optionally adjust LR
        ...     if damper.should_damp(val_loss):
        ...         for pg in optimizer.param_groups:
        ...             pg['lr'] = damper.apply(pg['lr'])
    """
    
    def __init__(
        self,
        alpha: float = 0.40,
        beta_threshold: float = 0.008,
        epsilon: float = 1e-4,
        ema_alpha: float = 0.3
    ):
        self.alpha = float(alpha)
        self.beta_threshold = float(beta_threshold)
        self.epsilon = float(epsilon)
        self.ema_alpha = float(ema_alpha)
        
        # State for beta estimation
        self._loss_ema: Optional[float] = None
        self._loss_sq_ema: Optional[float] = None
        self._beta: float = 1.0  # Start high (no damping)
        self._is_active: bool = False
    
    def update(self, loss: float) -> float:
        """
        Update beta estimate from new loss value.
        
        Args:
            loss: Current loss value
            
        Returns:
            Updated beta estimate
        """
        if self._loss_ema is None:
            self._loss_ema = loss
            self._loss_sq_ema = loss * loss
            self._beta = 1.0
            return self._beta
        
        # Update EMAs
        a = self.ema_alpha
        self._loss_ema = (1.0 - a) * self._loss_ema + a * loss
        self._loss_sq_ema = (1.0 - a) * self._loss_sq_ema + a * (loss * loss)
        
        # Estimate variance: Var(L) = E[L²] - E[L]²
        variance = max(self._loss_sq_ema - self._loss_ema ** 2, 1e-12)
        self._beta = math.sqrt(variance)
        
        # Update active state
        self._is_active = self._beta < self.beta_threshold
        
        return self._beta
    
    def should_damp(self, loss: Optional[float] = None) -> bool:
        """
        Check if damping should be applied.
        
        Args:
            loss: If provided, updates beta estimate first
            
        Returns:
            True if damping should be applied
        """
        if loss is not None:
            self.update(loss)
        return self._is_active
    
    def apply(self, lr: float, beta: Optional[float] = None) -> float:
        """
        Apply damping to learning rate.
        
        Args:
            lr: Current learning rate
            beta: Override beta value (uses internal estimate if None)
            
        Returns:
            Damped learning rate
        """
        b = beta if beta is not None else self._beta
        return apply_convergence_damping(lr, b, self.alpha, self.epsilon)
    
    @property
    def beta(self) -> float:
        """Current beta estimate."""
        return self._beta
    
    @property
    def is_active(self) -> bool:
        """Whether damping is currently active."""
        return self._is_active
    
    def reset(self) -> None:
        """Reset internal state."""
        self._loss_ema = None
        self._loss_sq_ema = None
        self._beta = 1.0
        self._is_active = False
    
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'alpha': self.alpha,
            'beta_threshold': self.beta_threshold,
            'epsilon': self.epsilon,
            'ema_alpha': self.ema_alpha,
            'loss_ema': self._loss_ema,
            'loss_sq_ema': self._loss_sq_ema,
            'beta': self._beta,
            'is_active': self._is_active,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.alpha = state.get('alpha', self.alpha)
        self.beta_threshold = state.get('beta_threshold', self.beta_threshold)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.ema_alpha = state.get('ema_alpha', self.ema_alpha)
        self._loss_ema = state.get('loss_ema')
        self._loss_sq_ema = state.get('loss_sq_ema')
        self._beta = state.get('beta', 1.0)
        self._is_active = state.get('is_active', False)


# Backward compatibility alias
stillness_amplifier = apply_convergence_damping

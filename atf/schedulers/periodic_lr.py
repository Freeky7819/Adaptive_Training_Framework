"""
Periodic Learning Rate Scheduler
================================

Exponentially-decaying learning rate with periodic oscillation.

This scheduler implements a learning rate that combines exponential
decay with a sinusoidal oscillation in log-time. This combination
has been observed to improve convergence on various deep learning
tasks by preventing premature settling into suboptimal minima.

Mathematical Formulation:
-------------------------
The learning rate at time t is:

    η(t) = η₀ · exp(-k·t) · (1 + α · sin(ω · log(1+t) + φ))

where:
    - η₀ is the base learning rate
    - k is the exponential decay coefficient (≥0)
    - α is the oscillation amplitude (typically 0.05-0.15)
    - ω is the angular frequency (typically 5.0-7.0)
    - φ is the phase offset

Components:
-----------
1. Base decay: exp(-k·t)
   - Provides the overall downward trend
   - k=0 gives constant LR (no decay)
   - k>0 provides exponential reduction

2. Periodic modulation: (1 + α · sin(ω · log(1+t) + φ))
   - Adds oscillation around the decay curve
   - Log-time frequency prevents rapid oscillation at late stages
   - Amplitude α controls the magnitude of variation

Behavior:
---------
- Early training (small t): Oscillations are rapid, large LR
- Late training (large t): Oscillations slow down, LR decays
- The log-periodic structure ensures scale-invariance

Typical Parameters:
-------------------
- η₀: 1e-3 (standard for Adam)
- k: 0.003 (gentle decay)
- α: 0.08 (8% amplitude)
- ω: 6.0 (≈1 period per e-fold in log-time)
- φ: π/3 ≈ 1.05 (shifts initial phase)

Integration Example:
--------------------
    from atf.schedulers import PeriodicLRScheduler
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = PeriodicLRScheduler(
        optimizer,
        base_lr=1e-3,
        decay=0.003,
        amplitude=0.08,
        omega=6.0
    )
    
    for epoch in range(epochs):
        for batch in dataloader:
            loss = compute_loss(...)
            loss.backward()
            optimizer.step()
            scheduler.step_iter()  # Update LR per iteration
        
        # Or: scheduler.step()  # Update LR per epoch

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


def _safe_log1p(x: float) -> float:
    """Numerically safe log(1+x) for x >= 0."""
    if x <= 0:
        return 0.0
    return math.log1p(x)


class PeriodicLRScheduler(_LRScheduler):
    """
    Learning rate scheduler with exponential decay and periodic oscillation.
    
    Implements: η(t) = η₀ · exp(-k·t) · (1 + α · sin(ω · log(1+t) + φ))
    
    Parameters:
        optimizer: PyTorch optimizer
        base_lr: Base learning rate η₀ (default: 1e-3)
        decay: Exponential decay coefficient k (default: 0.0)
               0 = no decay, >0 = exponential reduction
        amplitude: Oscillation amplitude α (default: 0.05)
                   Recommended range: [0.03, 0.15]
        omega: Angular frequency ω (default: 6.0)
               Recommended range: [4.0, 8.0]
        phi: Phase offset φ (default: π/3 ≈ 1.05)
        step_mode: When to update LR (default: "epoch")
                   - "epoch": Update per epoch (call step())
                   - "iter": Update per iteration (call step_iter())
        last_epoch: Initial epoch count (default: -1)
    
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = PeriodicLRScheduler(
        ...     optimizer,
        ...     base_lr=1e-3,
        ...     decay=0.003,
        ...     amplitude=0.08,
        ...     omega=6.0
        ... )
        >>> 
        >>> for epoch in range(100):
        ...     for inputs, targets in dataloader:
        ...         loss = criterion(model(inputs), targets)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step_iter()  # Per-iteration update
        ...     
        ...     # Or: scheduler.step()  # Per-epoch update
    
    Note:
        The scheduler modifies optimizer.param_groups[i]['lr'] directly.
        Initial LRs are set to base_lr if not already configured.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-3,
        decay: float = 0.0,
        amplitude: float = 0.05,
        omega: float = 6.0,
        phi: float = math.pi / 3,
        step_mode: str = "epoch",
        last_epoch: int = -1
    ):
        self.base_lr = float(base_lr)
        self.decay = float(decay)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phi = float(phi)
        self.step_mode = step_mode
        
        # Internal counters
        self._iter_count = 0
        self._epoch_count = 0
        
        # Set initial LRs to base_lr if not present
        for param_group in optimizer.param_groups:
            if 'lr' not in param_group:
                param_group['lr'] = self.base_lr
        
        super().__init__(optimizer, last_epoch=last_epoch)
        
        # Ensure base_lrs are consistent
        self.base_lrs = [self.base_lr for _ in self.base_lrs]
        self._last_lr = self.base_lrs.copy()
    
    def _compute_lr(self, t: int) -> float:
        """
        Compute learning rate for time step t.
        
        Args:
            t: Time step (iteration or epoch depending on mode)
        
        Returns:
            Learning rate value
        """
        t = max(int(t), 0)
        
        # Periodic component: 1 + α · sin(ω · log(1+t) + φ)
        oscillation = 1.0 + self.amplitude * math.sin(
            self.omega * _safe_log1p(t) + self.phi
        )
        
        # Safety: ensure positive
        if oscillation <= 0:
            oscillation = 1e-8
        
        # Full formula: η₀ · exp(-k·t) · oscillation
        lr = self.base_lr * math.exp(-self.decay * t) * oscillation
        
        return lr
    
    def get_lr(self) -> List[float]:
        """
        Compute current learning rates for all parameter groups.
        
        Returns:
            List of learning rates (one per parameter group)
        """
        # Choose time based on step mode
        if self.step_mode == "iter":
            t = self._iter_count
        else:
            t = self._epoch_count
        
        lr_t = self._compute_lr(t)
        
        # Scale each group proportionally
        scale = lr_t / self.base_lr if self.base_lr > 0 else 1.0
        return [base_lr * scale for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Perform epoch-based learning rate update.
        
        Args:
            epoch: Epoch number (increments internal counter if None)
        """
        if epoch is None:
            self._epoch_count += 1
            self.last_epoch += 1
        else:
            self._epoch_count = int(epoch)
            self.last_epoch = int(epoch)
        
        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        self._last_lr = new_lrs
    
    def step_iter(self) -> None:
        """
        Perform iteration-based learning rate update.
        
        Call this method after each training iteration (batch).
        """
        self._iter_count += 1
        self.last_epoch += 1
        
        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        self._last_lr = new_lrs
    
    def set_params(
        self,
        base_lr: Optional[float] = None,
        decay: Optional[float] = None,
        amplitude: Optional[float] = None,
        omega: Optional[float] = None,
        phi: Optional[float] = None,
        step_mode: Optional[str] = None
    ) -> None:
        """
        Update scheduler parameters dynamically.
        
        This allows the Meta-Learning Controller to adjust parameters
        during training based on observed dynamics.
        
        Args:
            base_lr: New base learning rate (also updates base_lrs)
            decay: New decay coefficient
            amplitude: New oscillation amplitude
            omega: New angular frequency
            phi: New phase offset
            step_mode: New step mode ("epoch" or "iter")
        """
        if base_lr is not None:
            self.base_lr = float(base_lr)
            self.base_lrs = [self.base_lr for _ in self.base_lrs]
        if decay is not None:
            self.decay = float(decay)
        if amplitude is not None:
            self.amplitude = float(amplitude)
        if omega is not None:
            self.omega = float(omega)
        if phi is not None:
            self.phi = float(phi)
        if step_mode is not None:
            self.step_mode = str(step_mode)
    
    @property
    def alpha(self) -> float:
        """Alias for amplitude (backward compatibility)."""
        return self.amplitude
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        self.amplitude = float(value)
    
    @property
    def eta0(self) -> float:
        """Alias for base_lr (backward compatibility)."""
        return self.base_lr
    
    @eta0.setter
    def eta0(self, value: float) -> None:
        self.base_lr = float(value)
        self.base_lrs = [self.base_lr for _ in self.base_lrs]
    
    @property
    def k(self) -> float:
        """Alias for decay (backward compatibility)."""
        return self.decay
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        state = super().state_dict()
        state.update({
            'base_lr': self.base_lr,
            'decay': self.decay,
            'amplitude': self.amplitude,
            'omega': self.omega,
            'phi': self.phi,
            'step_mode': self.step_mode,
            '_iter_count': self._iter_count,
            '_epoch_count': self._epoch_count,
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        super().load_state_dict(state_dict)
        
        for key in ['base_lr', 'decay', 'amplitude', 'omega', 'phi',
                    'step_mode', '_iter_count', '_epoch_count']:
            if key in state_dict:
                setattr(self, key, state_dict[key])
        
        # Update base_lrs if base_lr was restored
        if 'base_lr' in state_dict:
            self.base_lrs = [float(self.base_lr) for _ in self.base_lrs]
        
        self._last_lr = [pg['lr'] for pg in self.optimizer.param_groups]


# Backward compatibility alias
ResonantLRScheduler = PeriodicLRScheduler

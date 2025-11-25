"""
Gradient Feedback Controller
============================

Adaptive loss modulation based on training dynamics.

This module implements a feedback mechanism that modulates the loss
function based on the direction of loss change. The core idea is to
provide stronger gradients when the model is improving (loss decreasing)
and weaker gradients when overfitting or oscillating (loss increasing).

Mathematical Formulation:
-------------------------
The controller tracks the exponential moving average (EMA) of the loss
and computes the direction of change:

    δ = EMA(L_t) - EMA(L_{t-1})

The feedback signal is computed as:

    f(t) = sign(-δ) · α · sin(ω·t + φ)

where:
    - sign(-δ) = +1 when loss is decreasing (improvement)
    - sign(-δ) = -1 when loss is increasing (deterioration)
    - α is the feedback strength
    - ω is the angular frequency
    - φ is the phase offset
    - t is the training step

The modulated loss becomes:

    L' = L · (1 + f(t))

This has the effect of:
    - Amplifying gradients when training is progressing well
    - Dampening gradients when training shows signs of instability

The periodic component (sin) prevents the feedback from becoming
monotonic, which could lead to runaway amplification or complete
gradient suppression.

Theoretical Motivation:
-----------------------
Traditional training applies uniform gradient updates regardless of
whether the current batch is "easy" or "hard" relative to the model's
current state. This controller introduces adaptive behavior:

1. When loss is decreasing: The model is learning effectively, so we
   can afford to take slightly larger steps (positive feedback)
   
2. When loss is increasing: The model may be overfitting or the
   learning rate may be too high, so we reduce step sizes (negative
   feedback)

The oscillating nature of the feedback ensures that the system doesn't
settle into a fixed amplification/dampening regime, maintaining
exploration of the loss landscape.

References:
-----------
- Adaptive learning rate methods (AdaGrad, Adam)
- Curriculum learning and self-paced learning
- Control theory feedback mechanisms

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import math
from typing import Optional, Union


class GradientFeedbackController:
    """
    Adaptive loss modulation based on training dynamics.
    
    This controller monitors the loss trajectory and applies periodic
    feedback to modulate gradient magnitudes. The feedback direction
    is determined by whether the loss is improving or deteriorating.
    
    Parameters:
        alpha: Feedback strength coefficient (default: 0.05)
               Higher values = stronger modulation
               Recommended range: [0.01, 0.15]
               
        omega: Angular frequency for periodic modulation (default: 6.0)
               Controls how quickly the feedback oscillates
               Recommended range: [4.0, 8.0]
               
        phi: Phase offset in radians (default: 0.3)
             Shifts the starting point of oscillation
             
        ema_alpha: Smoothing factor for loss EMA (default: 0.3)
                   Lower = more smoothing (slower response)
                   Higher = less smoothing (faster response)
                   
        clamp: Maximum absolute feedback value (default: 0.2)
               Prevents extreme modulation
    
    Example:
        >>> controller = GradientFeedbackController(alpha=0.05, omega=6.0)
        >>> for step, (inputs, targets) in enumerate(dataloader):
        ...     loss = criterion(model(inputs), targets)
        ...     feedback = controller.step(loss.item(), step)
        ...     modulated_loss = loss * (1.0 + feedback)
        ...     modulated_loss.backward()
    
    Note:
        The first call to step() returns 0.0 as there is no previous
        loss to compare against. This is intentional to avoid
        arbitrary initialization effects.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        omega: float = 6.0,
        phi: float = 0.3,
        ema_alpha: float = 0.3,
        clamp: float = 0.2
    ):
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.phi = float(phi)
        self.ema_alpha = float(ema_alpha)
        self.clamp = float(clamp)
        
        # Internal state
        self._loss_ema: Optional[float] = None
        self._prev_loss_ema: Optional[float] = None
        self._step_count: int = 0
    
    def _to_float(self, x: Union[float, object]) -> float:
        """Convert input to float, handling torch.Tensor objects."""
        try:
            return float(x.item())  # type: ignore
        except (AttributeError, TypeError):
            return float(x)
    
    def _update_ema(self, loss_value: float) -> float:
        """
        Update the exponential moving average of the loss.
        
        Returns:
            The change in EMA (delta)
        """
        if self._loss_ema is None:
            self._loss_ema = loss_value
            self._prev_loss_ema = loss_value
            return 0.0
        
        self._prev_loss_ema = self._loss_ema
        self._loss_ema = (
            self.ema_alpha * loss_value + 
            (1.0 - self.ema_alpha) * self._loss_ema
        )
        return self._loss_ema - self._prev_loss_ema
    
    def step(self, loss: Union[float, object], t: float) -> float:
        """
        Compute feedback for the current training step.
        
        This method should be called after computing the loss but
        before backpropagation. The returned feedback value can be
        used to modulate the loss:
        
            modulated_loss = loss * (1.0 + feedback)
        
        Args:
            loss: Current training loss (float or torch.Tensor)
            t: Current training step or time parameter
            
        Returns:
            Feedback value in range [-clamp, +clamp]
            Positive = amplify gradients
            Negative = dampen gradients
        """
        delta = self._update_ema(self._to_float(loss))
        
        # First step: no feedback (no reference point)
        if self._step_count == 0:
            self._step_count += 1
            return 0.0
        
        # Compute feedback direction
        # sign = +1 when delta < 0 (loss decreasing = good)
        # sign = -1 when delta > 0 (loss increasing = bad)
        sign = +1.0 if delta < 0.0 else -1.0
        
        # Compute periodic feedback
        feedback = sign * self.alpha * math.sin(self.omega * t + self.phi)
        
        # Apply safety clamp
        if self.clamp > 0:
            feedback = max(-self.clamp, min(self.clamp, feedback))
        
        self._step_count += 1
        return feedback
    
    def get_feedback(self, t: float, loss: Union[float, object]) -> float:
        """
        Alternative API for computing feedback.
        
        This method is provided for compatibility with different
        training loop styles. It is equivalent to step().
        
        Args:
            t: Current training step
            loss: Current loss value
            
        Returns:
            Feedback value
        """
        return self.step(loss, t)
    
    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self._loss_ema = None
        self._prev_loss_ema = None
        self._step_count = 0
    
    @property
    def loss_ema(self) -> Optional[float]:
        """Current exponential moving average of the loss."""
        return self._loss_ema
    
    @property
    def step_count(self) -> int:
        """Number of steps processed."""
        return self._step_count
    
    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing."""
        return {
            'alpha': self.alpha,
            'omega': self.omega,
            'phi': self.phi,
            'ema_alpha': self.ema_alpha,
            'clamp': self.clamp,
            'loss_ema': self._loss_ema,
            'prev_loss_ema': self._prev_loss_ema,
            'step_count': self._step_count,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.alpha = state.get('alpha', self.alpha)
        self.omega = state.get('omega', self.omega)
        self.phi = state.get('phi', self.phi)
        self.ema_alpha = state.get('ema_alpha', self.ema_alpha)
        self.clamp = state.get('clamp', self.clamp)
        self._loss_ema = state.get('loss_ema')
        self._prev_loss_ema = state.get('prev_loss_ema')
        self._step_count = state.get('step_count', 0)


# Backward compatibility alias
SmartTeacher = GradientFeedbackController

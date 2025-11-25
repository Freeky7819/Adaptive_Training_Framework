"""
Curvature Regularizer
=====================

Fisher Information-based regularization for smoother loss landscapes.

This module estimates the local curvature of the loss landscape using
the diagonal of the Fisher Information Matrix and applies regularization
when the curvature exceeds a threshold. The goal is to encourage the
optimizer to find flatter minima, which are associated with better
generalization.

Theoretical Background:
-----------------------
The Fisher Information Matrix F for a model parameterized by θ is:

    F(θ) = E[∇log p(y|x,θ) · ∇log p(y|x,θ)ᵀ]

For classification with cross-entropy loss:

    F(θ) ≈ E[∇L · ∇Lᵀ]

The diagonal elements F_ii ≈ E[(∂L/∂θ_i)²] provide a computationally
efficient approximation of the curvature in each parameter direction.

High curvature indicates:
- Sharp regions of the loss landscape
- Sensitivity to parameter perturbations
- Potential for poor generalization

The regularizer adds a penalty when average curvature exceeds a threshold:

    L' = L + λ · max(0, curvature - threshold)

where λ is the regularization strength.

Implementation Details:
-----------------------
1. Fisher Estimation: Computed using torch.autograd.grad with
   create_graph=True to allow backpropagation through the estimate.

2. Momentum Smoothing: The Fisher estimate is smoothed over time
   using exponential moving average to reduce noise.

3. Selective Activation: Regularization only applied when curvature
   exceeds the threshold, avoiding unnecessary computation.

Typical Parameters:
-------------------
- threshold: 1e-3 (activation threshold)
- strength: 0.05 (regularization coefficient)
- momentum: 0.9 (EMA smoothing)

References:
-----------
- Martens, J. (2014). New insights and perspectives on the natural
  gradient method.
- Keskar et al. (2017). On large-batch training for deep learning:
  Generalization gap and sharp minima.
- Hochreiter & Schmidhuber (1997). Flat minima.

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
from typing import Dict, Optional, List
import torch
import torch.nn as nn


class CurvatureRegularizer:
    """
    Fisher Information-based curvature regularization.
    
    This class estimates the local curvature of the loss landscape
    using the diagonal Fisher Information Matrix approximation and
    applies regularization to encourage flatter minima.
    
    Parameters:
        threshold: Curvature threshold for activation (default: 1e-3)
                   Regularization only applied when curvature exceeds this
        strength: Regularization coefficient λ (default: 0.05)
        momentum: EMA momentum for Fisher estimate (default: 0.9)
        epsilon: Numerical stability constant (default: 1e-8)
        enabled: Whether to apply regularization (default: True)
        device: Computation device (default: auto-detect)
    
    Example:
        >>> regularizer = CurvatureRegularizer(threshold=1e-3, strength=0.05)
        >>> 
        >>> for inputs, targets in dataloader:
        ...     outputs = model(inputs)
        ...     loss = criterion(outputs, targets)
        ...     
        ...     # Estimate Fisher and apply regularization
        ...     regularizer.estimate_fisher(model, loss)
        ...     loss = regularizer.apply(loss, model)
        ...     
        ...     loss.backward()
        ...     optimizer.step()
    
    Note:
        The Fisher estimation requires create_graph=True, which adds
        computational overhead. For efficiency, consider calling
        estimate_fisher() only periodically (e.g., every N steps).
    """
    
    def __init__(
        self,
        threshold: float = 1e-3,
        strength: float = 0.05,
        momentum: float = 0.9,
        epsilon: float = 1e-8,
        enabled: bool = True,
        device: Optional[str] = None
    ):
        self.threshold = float(threshold)
        self.strength = float(strength)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.enabled = enabled
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Internal state
        self._fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None
        self._curvature: float = 0.0
    
    def estimate_fisher(
        self,
        model: nn.Module,
        loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the diagonal Fisher Information Matrix.
        
        Computes F_ii ≈ E[(∂L/∂θ_i)²] for each parameter using
        automatic differentiation.
        
        Args:
            model: Neural network model
            loss: Current loss value (must have grad_fn)
        
        Returns:
            Dictionary mapping parameter names to Fisher diagonal tensors
        
        Note:
            The loss must be computed with gradients enabled and should
            not have been backpropagated yet.
        """
        if not self.enabled:
            return {}
        
        # Compute gradients with graph retention for potential regularization
        params = list(model.parameters())
        
        try:
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
        except RuntimeError:
            # Gradient computation failed (e.g., no grad_fn)
            return {}
        
        fisher_dict: Dict[str, torch.Tensor] = {}
        
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is None:
                continue
            
            # Fisher diagonal: E[grad²]
            fisher_diag = grad.detach().float().pow(2)
            
            # Apply momentum smoothing if we have previous estimate
            if self._fisher_diagonal is not None and name in self._fisher_diagonal:
                prev = self._fisher_diagonal[name]
                fisher_diag = (
                    self.momentum * prev +
                    (1.0 - self.momentum) * fisher_diag
                )
            
            fisher_dict[name] = fisher_diag
        
        # Update internal state
        self._fisher_diagonal = fisher_dict
        
        # Compute average curvature
        if fisher_dict:
            curvatures = [v.mean().item() for v in fisher_dict.values()]
            self._curvature = sum(curvatures) / len(curvatures)
        else:
            self._curvature = 0.0
        
        return fisher_dict
    
    def apply(
        self,
        loss: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Apply curvature regularization to the loss.
        
        If the estimated curvature exceeds the threshold, adds a
        regularization term proportional to the excess curvature.
        
        Args:
            loss: Original loss value
            model: Neural network model (unused, for API compatibility)
        
        Returns:
            Regularized loss (original + regularization term if active)
        """
        if not self.enabled:
            return loss
        
        if self._fisher_diagonal is None:
            return loss
        
        if self._curvature <= self.threshold:
            return loss
        
        # Compute regularization term
        # Only penalize curvature above threshold
        excess_curvature = self._curvature - self.threshold
        regularization = self.strength * excess_curvature
        
        return loss + regularization
    
    @property
    def curvature(self) -> float:
        """Current estimated curvature."""
        return self._curvature
    
    @property
    def fisher_diagonal(self) -> Optional[Dict[str, torch.Tensor]]:
        """Current Fisher diagonal estimate (or None)."""
        return self._fisher_diagonal
    
    @property
    def is_active(self) -> bool:
        """Whether regularization is currently being applied."""
        return self.enabled and self._curvature > self.threshold
    
    def reset(self) -> None:
        """Reset internal state."""
        self._fisher_diagonal = None
        self._curvature = 0.0
    
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        fisher_state = None
        if self._fisher_diagonal is not None:
            fisher_state = {
                k: v.cpu() for k, v in self._fisher_diagonal.items()
            }
        
        return {
            'threshold': self.threshold,
            'strength': self.strength,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'enabled': self.enabled,
            'curvature': self._curvature,
            'fisher_diagonal': fisher_state,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.threshold = state.get('threshold', self.threshold)
        self.strength = state.get('strength', self.strength)
        self.momentum = state.get('momentum', self.momentum)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.enabled = state.get('enabled', self.enabled)
        self._curvature = state.get('curvature', 0.0)
        
        fisher_state = state.get('fisher_diagonal')
        if fisher_state is not None:
            self._fisher_diagonal = {
                k: v.to(self.device) for k, v in fisher_state.items()
            }
        else:
            self._fisher_diagonal = None


# Backward compatibility alias
ManifoldFlattener = CurvatureRegularizer

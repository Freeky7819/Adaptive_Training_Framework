"""
Meta-Learning Controller
========================

Epoch-level hyperparameter adaptation based on training dynamics.

This module implements an automated hyperparameter controller that
monitors training progress and makes adaptive adjustments to learning
rate and other parameters. It acts as a meta-learning layer on top
of the base training loop.

Design Philosophy:
------------------
The controller observes three types of signals:

1. Improvement Signal: Whether validation loss improved
2. Deterioration Signal: Whether validation loss worsened significantly
3. Stability Signal: The variance (β) of recent losses

Based on these signals, it can:
- Reduce learning rate when training stagnates
- Adjust oscillation amplitude (α) based on stability
- Adjust frequency (ω) based on loss curvature
- Trigger early stopping when no progress is made

Decision Rules:
---------------
1. Learning Rate Reduction:
   - If loss worsens for `worsen_patience` consecutive epochs
   - Reduce LR by `lr_drop` factor (default: 0.5)
   - Reset worsen counter after reduction

2. Amplitude Adjustment:
   - If β < 0.40 (stable): Increase α slightly (explore more)
   - If β > 0.75 (unstable): Decrease α slightly (conserve)
   - Changes bounded to [α_min, α_max]

3. Frequency Adjustment:
   - If R² of loss fit is low and curvature is high
   - Adjust ω to better match the loss landscape
   - Changes bounded to [ω_min, ω_max]

4. Early Stopping:
   - If no improvement for `no_improve_patience` epochs
   - Signal to terminate training

Integration Pattern:
--------------------
    controller = MetaLearningController()
    
    for epoch in range(max_epochs):
        train_loss = train(...)
        val_loss = validate(...)
        
        # Compute metrics
        history = {
            'epoch': epoch,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'beta': beta,
            'lr': current_lr,
        }
        
        # Get controller decision
        decision = controller.on_epoch_end(history, orchestrator)
        
        # Apply decisions
        if 'lr' in decision['apply']:
            set_lr(decision['apply']['lr'])
        if decision['early_stop']:
            break

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp value to [minimum, maximum] range."""
    return max(minimum, min(maximum, value))


@dataclass
class MetaControllerConfig:
    """
    Configuration for the Meta-Learning Controller.
    
    Attributes:
        min_lr: Minimum allowed learning rate
        max_lr: Maximum allowed learning rate
        alpha_min: Minimum oscillation amplitude
        alpha_max: Maximum oscillation amplitude
        omega_min: Minimum angular frequency
        omega_max: Maximum angular frequency
        worsen_patience: Epochs of worsening before LR reduction
        no_improve_patience: Epochs without improvement before early stop
        lr_drop: LR reduction factor
        alpha_step: Step size for amplitude adjustment
        omega_step: Step size for frequency adjustment
    """
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    alpha_min: float = 0.03
    alpha_max: float = 0.12
    omega_min: float = 5.2
    omega_max: float = 6.8
    worsen_patience: int = 2
    no_improve_patience: int = 4
    lr_drop: float = 0.5
    alpha_step: float = 0.01
    omega_step: float = 0.1


class MetaLearningController:
    """
    Epoch-level hyperparameter adaptation controller.
    
    This controller monitors training dynamics and makes adaptive
    decisions about learning rate, oscillation amplitude, and
    frequency. It also determines when training should stop.
    
    Parameters:
        config: Configuration object (default: MetaControllerConfig())
    
    Example:
        >>> controller = MetaLearningController()
        >>> 
        >>> for epoch in range(max_epochs):
        ...     # Training and validation
        ...     train_loss = train_epoch(...)
        ...     val_loss = validate(...)
        ...     
        ...     # Prepare metrics
        ...     history = {
        ...         'val_loss': val_loss,
        ...         'best_val_loss': best_val_loss,
        ...         'beta': loss_variance,
        ...         'lr': optimizer.param_groups[0]['lr'],
        ...     }
        ...     
        ...     # Get controller decision
        ...     decision = controller.on_epoch_end(history, orchestrator)
        ...     
        ...     # Apply adjustments
        ...     for key, value in decision['apply'].items():
        ...         if key == 'lr':
        ...             for pg in optimizer.param_groups:
        ...                 pg['lr'] = value
        ...     
        ...     # Check for early stopping
        ...     if decision['early_stop']:
        ...         print(f"Early stopping: {decision['notes']}")
        ...         break
    """
    
    def __init__(self, config: Optional[MetaControllerConfig] = None):
        self.config = config or MetaControllerConfig()
        
        # State tracking
        self._worsen_streak = 0
        self._no_improve_count = 0
        self._last_best = None
    
    def on_epoch_end(
        self,
        history: Dict[str, Any],
        orchestrator: Any
    ) -> Dict[str, Any]:
        """
        Process epoch results and return adjustment decisions.
        
        Args:
            history: Dictionary containing:
                - 'val_loss': Current validation loss
                - 'best_val_loss': Best validation loss so far
                - 'beta': Loss variance estimate (optional)
                - 'lr': Current learning rate
                - 'curv': Curvature estimate (optional)
                - 'r2_eiw': R² of periodic fit (optional)
            orchestrator: Training orchestrator with scheduler access
        
        Returns:
            Dictionary with:
                - 'apply': Dict of parameter adjustments to apply
                - 'early_stop': Boolean indicating if training should stop
                - 'notes': Human-readable explanation of decisions
        """
        cfg = self.config
        notes: list = []
        apply: Dict[str, float] = {}
        early_stop = False
        
        # Extract metrics
        val_loss = history.get('val_loss', float('inf'))
        best_loss = history.get('best_val_loss', val_loss)
        
        # Initialize best tracking
        if self._last_best is None:
            self._last_best = best_loss
        
        # --- 1. Track improvement/deterioration ---
        if val_loss > self._last_best + 1e-6:
            # Loss worsened
            self._worsen_streak += 1
            self._no_improve_count += 1
        elif val_loss < self._last_best - 1e-6:
            # New best
            self._worsen_streak = 0
            self._no_improve_count = 0
            self._last_best = val_loss
        else:
            # No change
            self._no_improve_count += 1
        
        # --- 2. Learning rate policy ---
        current_lr = history.get('lr', 1e-3)
        
        if self._worsen_streak >= cfg.worsen_patience:
            # Reduce LR due to deterioration
            new_lr = clamp(
                current_lr * cfg.lr_drop,
                cfg.min_lr,
                cfg.max_lr
            )
            if new_lr < current_lr:
                apply['lr'] = new_lr
                notes.append(
                    f"LR reduced to {new_lr:.2e} "
                    f"(deterioration streak: {self._worsen_streak})"
                )
            self._worsen_streak = 0  # Reset after action
        
        # --- 3. Amplitude adjustment based on stability ---
        beta = history.get('beta', 0.5)
        scheduler = getattr(orchestrator, 'periodic_scheduler', None)
        
        if scheduler is not None:
            current_alpha = getattr(scheduler, 'alpha', cfg.alpha_min)
            
            if 0 < beta < 0.40:
                # Stable training - can afford more exploration
                new_alpha = clamp(
                    current_alpha + cfg.alpha_step,
                    cfg.alpha_min,
                    cfg.alpha_max
                )
                if new_alpha != current_alpha:
                    apply['alpha'] = new_alpha
                    notes.append(
                        f"Amplitude increased to {new_alpha:.3f} "
                        f"(stable: β={beta:.2f})"
                    )
            elif beta > 0.75:
                # Unstable training - reduce exploration
                new_alpha = clamp(
                    current_alpha - cfg.alpha_step,
                    cfg.alpha_min,
                    cfg.alpha_max
                )
                if new_alpha != current_alpha:
                    apply['alpha'] = new_alpha
                    notes.append(
                        f"Amplitude decreased to {new_alpha:.3f} "
                        f"(unstable: β={beta:.2f})"
                    )
            
            # --- 3b. Frequency adjustment based on curvature ---
            r2 = history.get('r2_eiw', 1.0)
            curvature = history.get('curv', 0.0)
            current_omega = getattr(scheduler, 'omega', 6.0)
            
            if r2 < 0.6 and curvature > 1e-4:
                # Poor fit and high curvature - adjust frequency
                omega_adjust = cfg.omega_step if beta < 0.5 else -cfg.omega_step
                new_omega = clamp(
                    current_omega + omega_adjust,
                    cfg.omega_min,
                    cfg.omega_max
                )
                if new_omega != current_omega:
                    apply['omega'] = new_omega
                    notes.append(
                        f"Frequency adjusted to {new_omega:.2f} "
                        f"(R²={r2:.2f}, curv={curvature:.1e})"
                    )
        
        # --- 4. Early stopping check ---
        if self._no_improve_count >= cfg.no_improve_patience:
            early_stop = True
            notes.append(
                f"Early stop triggered (no improvement for "
                f"{self._no_improve_count} epochs)"
            )
        
        return {
            'apply': apply,
            'early_stop': early_stop,
            'notes': "; ".join(notes) if notes else "No adjustments"
        }
    
    def reset(self) -> None:
        """Reset controller state for a new training run."""
        self._worsen_streak = 0
        self._no_improve_count = 0
        self._last_best = None
    
    @property
    def worsen_streak(self) -> int:
        """Current deterioration streak count."""
        return self._worsen_streak
    
    @property
    def no_improve_count(self) -> int:
        """Epochs since last improvement."""
        return self._no_improve_count
    
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'worsen_streak': self._worsen_streak,
            'no_improve_count': self._no_improve_count,
            'last_best': self._last_best,
            'config': {
                'min_lr': self.config.min_lr,
                'max_lr': self.config.max_lr,
                'alpha_min': self.config.alpha_min,
                'alpha_max': self.config.alpha_max,
                'omega_min': self.config.omega_min,
                'omega_max': self.config.omega_max,
                'worsen_patience': self.config.worsen_patience,
                'no_improve_patience': self.config.no_improve_patience,
                'lr_drop': self.config.lr_drop,
            }
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self._worsen_streak = state.get('worsen_streak', 0)
        self._no_improve_count = state.get('no_improve_count', 0)
        self._last_best = state.get('last_best')
        
        if 'config' in state:
            cfg = state['config']
            self.config = MetaControllerConfig(
                min_lr=cfg.get('min_lr', self.config.min_lr),
                max_lr=cfg.get('max_lr', self.config.max_lr),
                alpha_min=cfg.get('alpha_min', self.config.alpha_min),
                alpha_max=cfg.get('alpha_max', self.config.alpha_max),
                omega_min=cfg.get('omega_min', self.config.omega_min),
                omega_max=cfg.get('omega_max', self.config.omega_max),
                worsen_patience=cfg.get('worsen_patience', self.config.worsen_patience),
                no_improve_patience=cfg.get('no_improve_patience', self.config.no_improve_patience),
                lr_drop=cfg.get('lr_drop', self.config.lr_drop),
            )


# Backward compatibility aliases
AutoCoach = MetaLearningController
AutoCoachCfg = MetaControllerConfig

"""
Convergence Monitor
===================

Adaptive early stopping and checkpoint management.

This module implements training monitoring that tracks validation loss,
manages checkpoints, and triggers early stopping based on convergence
criteria. It uses exponential moving average (EMA) smoothing to reduce
noise in the monitoring signal.

Features:
---------
1. Validation Loss Tracking:
   - Optional EMA smoothing for noisy datasets
   - Relative improvement threshold (min_delta)
   - Patience-based early stopping

2. Adaptive Learning Rate:
   - Automatic LR reduction on plateau
   - Configurable reduction factor and count

3. Checkpoint Management:
   - Automatic checkpoint on improvement
   - Keep top-K checkpoints only
   - Full state preservation

4. TensorBoard Integration:
   - Optional logging of all metrics
   - Loss curves and LR tracking

Integration Pattern:
--------------------
    monitor = ConvergenceMonitor(
        patience=5,
        min_delta=0.005,
        checkpoint_dir='checkpoints'
    )
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(...)
        val_loss = validate(...)
        
        action = monitor.on_epoch_end(
            epoch=epoch,
            val_loss=val_loss,
            train_loss=train_loss,
            model=model,
            optimizer=optimizer
        )
        
        if action['stop']:
            print(f"Early stopping: {action['reason']}")
            break
    
    monitor.finalize()

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import torch
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
import time
import json
from enum import Enum


class TrainingState(Enum):
    """
    Training state classification.
    
    Used for diagnostic reporting and visualization.
    """
    INITIALIZING = 'initializing'
    CONVERGING = 'converging'
    OSCILLATING = 'oscillating'
    DIVERGING = 'diverging'
    CONVERGED = 'converged'


class ConvergenceResult:
    """
    Container for convergence analysis results.
    
    Attributes:
        beta: Loss variance estimate
        omega: Current frequency parameter
        r2: Fit quality metric
        confidence: Confidence in current state
        state: Current training state
        epochs_to_convergence: Estimated epochs remaining (if applicable)
    """
    
    def __init__(
        self,
        beta: float = 0.0,
        omega: float = 0.0,
        r2: float = 0.0,
        confidence: float = 0.0,
        state: TrainingState = TrainingState.INITIALIZING,
        epochs_to_convergence: Optional[int] = None
    ):
        self.beta = beta
        self.omega = omega
        self.r2 = r2
        self.confidence = confidence
        self.state = state
        self.epochs_to_convergence = epochs_to_convergence


# TensorBoard availability check
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore


class ConvergenceMonitor:
    """
    Training convergence monitor with adaptive early stopping.
    
    This class monitors validation loss during training and implements:
    - Patience-based early stopping
    - Automatic learning rate reduction on plateau
    - Checkpoint management (keeps top-K)
    - Optional TensorBoard logging
    - EMA smoothing for noisy validation metrics
    
    Parameters:
        checkpoint_dir: Directory for saving checkpoints
        patience: Epochs without improvement before action
        min_delta: Minimum relative improvement (e.g., 0.005 = 0.5%)
        lr_reduction_factor: Factor to reduce LR (default: 0.5)
        min_lr: Minimum learning rate
        max_lr_reductions: Maximum LR reductions before stopping
        keep_top_k: Number of best checkpoints to keep
        verbose: Print status messages
        use_ema: Apply EMA smoothing to validation loss
        ema_alpha: EMA smoothing factor (0.3 = moderate smoothing)
        enable_tensorboard: Log metrics to TensorBoard
        tensorboard_dir: TensorBoard log directory
    
    Example:
        >>> monitor = ConvergenceMonitor(
        ...     patience=5,
        ...     min_delta=0.005,
        ...     max_lr_reductions=2
        ... )
        >>> 
        >>> for epoch in range(100):
        ...     train_loss = train_epoch(model, train_loader)
        ...     val_loss = validate(model, val_loader)
        ...     
        ...     action = monitor.on_epoch_end(
        ...         epoch=epoch,
        ...         val_loss=val_loss,
        ...         train_loss=train_loss,
        ...         model=model,
        ...         optimizer=optimizer
        ...     )
        ...     
        ...     if action['reduce_lr']:
        ...         print(f"LR reduced: {action['reason']}")
        ...     if action['stop']:
        ...         print(f"Stopping: {action['reason']}")
        ...         break
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        patience: int = 10,
        min_delta: float = 0.0,
        lr_reduction_factor: float = 0.5,
        min_lr: float = 1e-6,
        max_lr_reductions: int = 2,
        keep_top_k: int = 3,
        verbose: bool = True,
        use_ema: bool = False,
        ema_alpha: float = 0.3,
        enable_tensorboard: bool = True,
        tensorboard_dir: str = 'runs'
    ):
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Patience and thresholds
        self.patience = patience
        self.min_delta = min_delta
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.max_lr_reductions = max_lr_reductions
        self.keep_top_k = keep_top_k
        self.verbose = verbose
        
        # EMA settings
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self._ema_val_loss: Optional[float] = None
        
        # Tracking state
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0
        self.lr_reductions = 0
        self.current_lr: Optional[float] = None
        self._checkpoint_list: List[Dict] = []
        self._history: List[Dict] = []
        
        # TensorBoard
        self._writer = None
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            self._writer = SummaryWriter(tensorboard_dir)
            if self.verbose:
                print(f"📊 TensorBoard logging: {tensorboard_dir}")
    
    def _get_ema_loss(self, val_loss: float) -> float:
        """Apply EMA smoothing to validation loss."""
        if not self.use_ema:
            return val_loss
        
        if self._ema_val_loss is None:
            self._ema_val_loss = val_loss
        else:
            self._ema_val_loss = (
                self.ema_alpha * val_loss +
                (1.0 - self.ema_alpha) * self._ema_val_loss
            )
        return self._ema_val_loss
    
    def on_epoch_end(
        self,
        epoch: int,
        val_loss: float,
        train_loss: Optional[float] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Process end of epoch and return action dictionary.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            train_loss: Training loss (optional, for logging)
            model: Model for checkpointing (optional)
            optimizer: Optimizer for LR management (optional)
        
        Returns:
            Dictionary with:
                - 'reduce_lr': Whether LR was reduced
                - 'stop': Whether training should stop
                - 'reason': Explanation of action (if any)
                - 'checkpoint_saved': Whether a checkpoint was saved
                - 'convergence_result': ConvergenceResult object
        """
        # Apply EMA smoothing
        monitored_loss = self._get_ema_loss(val_loss)
        
        # Track current LR
        if optimizer is not None:
            self.current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        self._history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'monitored_loss': monitored_loss,
            'lr': self.current_lr,
            'timestamp': time.time()
        })
        
        # Create convergence result
        convergence_result = ConvergenceResult(
            state=(
                TrainingState.CONVERGING if monitored_loss < self.best_loss
                else TrainingState.OSCILLATING
            ),
            confidence=1.0 if monitored_loss < self.best_loss else 0.0
        )
        
        # Initialize action
        action = {
            'reduce_lr': False,
            'stop': False,
            'reason': None,
            'checkpoint_saved': False,
            'convergence_result': convergence_result
        }
        
        # Log to TensorBoard
        self._log_tensorboard(epoch, train_loss, val_loss, monitored_loss)
        
        # Check for improvement (relative threshold)
        improvement_threshold = self.best_loss * (1.0 - self.min_delta)
        
        if monitored_loss < improvement_threshold:
            # Improvement detected
            self.best_loss = monitored_loss
            self.epochs_since_improvement = 0
            
            # Save checkpoint
            if model is not None:
                self._save_checkpoint(model, optimizer, epoch, val_loss)
                action['checkpoint_saved'] = True
            
            if self.verbose:
                print(f"📊 Epoch {epoch}: Improved! Val Loss: {val_loss:.6f}")
        else:
            # No improvement
            self.epochs_since_improvement += 1
            
            if self.epochs_since_improvement >= self.patience:
                if self.lr_reductions >= self.max_lr_reductions:
                    # Max reductions reached - stop
                    action['stop'] = True
                    action['reason'] = (
                        f"Early stopping after {self.max_lr_reductions} "
                        f"LR reductions without improvement"
                    )
                    if self.verbose:
                        print(f"🛑 {action['reason']}")
                else:
                    # Reduce LR
                    if self._reduce_lr(optimizer):
                        action['reduce_lr'] = True
                        action['reason'] = (
                            f"LR reduction {self.lr_reductions}/"
                            f"{self.max_lr_reductions}"
                        )
                        self.epochs_since_improvement = 0
                        
                        if self.verbose:
                            print(f"📉 {action['reason']}: LR → {self.current_lr:.2e}")
            else:
                if self.verbose and epoch % 5 == 0:
                    print(
                        f"⏳ Epoch {epoch}: Patience "
                        f"{self.epochs_since_improvement}/{self.patience}"
                    )
        
        return action
    
    def _reduce_lr(self, optimizer: Optional[torch.optim.Optimizer]) -> bool:
        """Reduce learning rate."""
        if optimizer is None:
            return False
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr <= self.min_lr:
            return False
        
        new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.lr_reductions += 1
        self.current_lr = new_lr
        return True
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        loss: float
    ) -> None:
        """Save model checkpoint."""
        filename = f'checkpoint_epoch{epoch}_loss{loss:.6f}.pt'
        path = self.checkpoint_dir / filename
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'history': self._history
        }
        
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(state, path)
        
        self._checkpoint_list.append({
            'path': path,
            'loss': loss,
            'epoch': epoch
        })
        
        # Keep only top-K
        if len(self._checkpoint_list) > self.keep_top_k:
            self._checkpoint_list.sort(key=lambda x: x['loss'])
            to_remove = self._checkpoint_list[self.keep_top_k:]
            self._checkpoint_list = self._checkpoint_list[:self.keep_top_k]
            
            for ckpt in to_remove:
                try:
                    ckpt['path'].unlink()
                except Exception:
                    pass
        
        if self.verbose:
            print(f"💾 Saved: {filename}")
    
    def _log_tensorboard(
        self,
        epoch: int,
        train_loss: Optional[float],
        val_loss: float,
        monitored_loss: float
    ) -> None:
        """Log metrics to TensorBoard."""
        if self._writer is None:
            return
        
        if train_loss is not None:
            self._writer.add_scalar('Loss/Train', train_loss, epoch)
        self._writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        if self.use_ema:
            self._writer.add_scalar('Loss/Monitored_EMA', monitored_loss, epoch)
        
        if self.current_lr is not None:
            self._writer.add_scalar('Learning_Rate', self.current_lr, epoch)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return training statistics."""
        return {
            'best_loss': self.best_loss,
            'lr_reductions': self.lr_reductions,
            'epochs_trained': len(self._history),
            'ema_enabled': self.use_ema,
            'current_lr': self.current_lr
        }
    
    def save_history(self, path: Optional[str] = None) -> None:
        """Save training history to JSON."""
        if path is None:
            path = self.checkpoint_dir / 'training_history.json'
        else:
            path = Path(path)
        
        with open(path, 'w') as f:
            json.dump({
                'history': self._history,
                'statistics': self.get_statistics()
            }, f, indent=2)
    
    def finalize(self) -> None:
        """Finalize monitoring and close resources."""
        if self._writer:
            self._writer.close()
        
        self.save_history()
        
        if self.verbose:
            stats = self.get_statistics()
            print("\n" + "=" * 60)
            print("📊 TRAINING SUMMARY")
            print("=" * 60)
            print(f"Best validation loss: {stats['best_loss']:.6f}")
            print(f"LR reductions: {stats['lr_reductions']}")
            print(f"Total epochs: {stats['epochs_trained']}")
            print("=" * 60)
    
    def reset(self) -> None:
        """Reset monitor state for new training run."""
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0
        self.lr_reductions = 0
        self.current_lr = None
        self._ema_val_loss = None
        self._checkpoint_list.clear()
        self._history.clear()


# Backward compatibility aliases
ResonantCallback = ConvergenceMonitor
RCAResult = ConvergenceResult

__all__ = [
    'ConvergenceMonitor',
    'ConvergenceResult',
    'TrainingState',
    'ResonantCallback',
    'RCAResult'
]

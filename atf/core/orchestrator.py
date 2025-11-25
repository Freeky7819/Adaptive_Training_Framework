"""
Adaptive Training Orchestrator
==============================

Central coordinator for all training enhancement modules.

This module provides a unified interface for managing the various
training enhancement components. It handles initialization, coordinates
per-step and per-epoch operations, and manages the interaction between
different modules.

Architecture:
-------------
The orchestrator follows a modular design where each component can be
independently enabled or disabled through the TrainingConfig. This allows
for controlled experiments and gradual adoption.

Components managed:
- Harmonic Weight Initialization
- Curvature Regularization
- Gradient Feedback Controller
- Convergence Damper
- Temporal Feedback Buffer
- Periodic LR Scheduler
- Meta-Learning Controller

Lifecycle:
----------
1. Initialization: Create orchestrator with config and optimizer
2. Model setup: Call initialize_model() to apply weight initialization
3. Training loop:
   a. on_epoch_start(): Prepare for new epoch
   b. For each batch:
      - on_train_step(): Apply per-step enhancements
   c. on_eval_end(): Process validation results and adjust parameters
4. Finalization: Call get_summary_stats() for final report

Example:
--------
    config = TrainingConfig.full()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    model = orchestrator.initialize_model(model)
    
    for epoch in range(epochs):
        orchestrator.on_epoch_start(epoch)
        
        for step, (inputs, targets) in enumerate(train_loader):
            loss = criterion(model(inputs), targets)
            loss = orchestrator.on_train_step(model, loss, step)
            loss.backward()
            optimizer.step()
        
        val_loss = validate(model, val_loader)
        action, metrics = orchestrator.on_eval_end(epoch, val_loss)
        
        if action['stop']:
            break

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from .config import TrainingConfig
from ..schedulers.periodic_lr import PeriodicLRScheduler
from ..modules.gradient_feedback import GradientFeedbackController
from ..modules.curvature_regularizer import CurvatureRegularizer
from ..modules.convergence_damper import apply_convergence_damping
from ..modules.temporal_buffer import (
    TemporalFeedbackBuffer,
    compute_temporal_gain,
    suggest_amplitude_adjustment
)

# Optional imports
try:
    from ..modules.harmonic_init import (
        HarmonicWeightInitializer,
        WarmupScheduler,
        InitializationReport
    )
    HARMONIC_INIT_AVAILABLE = True
except ImportError:
    HARMONIC_INIT_AVAILABLE = False
    HarmonicWeightInitializer = None
    WarmupScheduler = None
    InitializationReport = None

try:
    from ..modules.meta_controller import (
        MetaLearningController,
        MetaControllerConfig
    )
    META_CONTROLLER_AVAILABLE = True
except ImportError:
    META_CONTROLLER_AVAILABLE = False
    MetaLearningController = None
    MetaControllerConfig = None


class AdaptiveTrainingOrchestrator:
    """
    Unified training enhancement coordinator.
    
    This class manages all training enhancement modules and provides
    a simple interface for integrating them into a training loop.
    
    Parameters:
        config: TrainingConfig instance specifying enabled features
        optimizer: PyTorch optimizer
    
    Attributes:
        config: Active configuration
        optimizer: Managed optimizer
        metrics: Current epoch metrics dictionary
        action: Current epoch actions (stop, reduce_lr, checkpoint)
    
    Example:
        >>> config = TrainingConfig(
        ...     use_convergence_analysis=True,
        ...     use_gradient_feedback=True,
        ...     use_periodic_lr=True
        ... )
        >>> orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
        >>> model = orchestrator.initialize_model(model)
        >>> 
        >>> for epoch in range(epochs):
        ...     orchestrator.on_epoch_start(epoch)
        ...     for step, batch in enumerate(dataloader):
        ...         loss = compute_loss(model, batch)
        ...         loss = orchestrator.on_train_step(model, loss, step)
        ...         loss.backward()
        ...         optimizer.step()
        ...     val_loss = validate(model)
        ...     action, metrics = orchestrator.on_eval_end(epoch, val_loss)
        ...     if action['stop']:
        ...         break
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        optimizer: torch.optim.Optimizer
    ):
        self.config = config
        self.optimizer = optimizer
        self.device = torch.device(config.device)
        
        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            'phase_coherence': 0.0,
            'curvature': 0.0,
            'curvature_reduction': 0.0,
            'beta': 0.0,
            'omega': config.lr_omega,
            'r2': 1.0,
            'confidence': 0.0,
            'damper_active': False,
            'current_lr': config.lr_base,
            'epoch_time': 0.0,
            'meta_notes': '',
        }
        
        # Actions for current epoch
        self.action: Dict[str, bool] = {
            'stop': False,
            'reduce_lr': False,
            'checkpoint': False,
        }
        
        # Convergence tracking
        self._loss_ema: Optional[float] = None
        self._loss_sq_ema: Optional[float] = None
        self.best_val_loss = float('inf')
        self.epochs_since_improvement = 0
        self.lr_reductions = 0
        
        # History for temporal buffer
        self._beta_history: List[float] = []
        self._omega_history: List[float] = []
        self._phi_history: List[float] = []
        
        # Initial curvature for reduction tracking
        self._initial_curvature: Optional[float] = None
        
        # Reports
        self.init_report: Optional[Any] = None
        
        # Initialize modules
        self._harmonic_init: Optional[Any] = None
        self._warmup: Optional[Any] = None
        self._curvature_reg: Optional[CurvatureRegularizer] = None
        self._gradient_feedback: Optional[GradientFeedbackController] = None
        self._periodic_scheduler: Optional[PeriodicLRScheduler] = None
        self._meta_controller: Optional[Any] = None
        self._temporal_buffer: Optional[TemporalFeedbackBuffer] = None
        
        self._init_modules()
    
    def _init_modules(self) -> None:
        """Initialize all enabled modules."""
        cfg = self.config
        
        # Curvature Regularizer
        if cfg.use_curvature_reg:
            self._curvature_reg = CurvatureRegularizer(
                threshold=cfg.curvature_threshold,
                strength=cfg.curvature_strength,
                momentum=cfg.curvature_momentum,
                enabled=cfg.use_fisher_diagonal,
                device=cfg.device
            )
        
        # Gradient Feedback Controller
        if cfg.use_gradient_feedback:
            self._gradient_feedback = GradientFeedbackController(
                alpha=cfg.gfc_alpha,
                omega=cfg.gfc_omega,
                phi=cfg.gfc_phi,
                ema_alpha=cfg.gfc_ema_alpha,
                clamp=cfg.gfc_clamp
            )
        
        # Periodic LR Scheduler
        if cfg.use_periodic_lr:
            self._periodic_scheduler = PeriodicLRScheduler(
                optimizer=self.optimizer,
                base_lr=cfg.lr_base,
                decay=cfg.lr_decay,
                amplitude=cfg.lr_amplitude,
                omega=cfg.lr_omega,
                phi=cfg.lr_phi,
                step_mode=cfg.lr_step_mode
            )
        
        # Meta-Learning Controller
        if cfg.use_meta_controller and META_CONTROLLER_AVAILABLE:
            self._meta_controller = MetaLearningController(
                MetaControllerConfig(
                    min_lr=cfg.meta_min_lr,
                    max_lr=cfg.meta_max_lr,
                    alpha_min=cfg.meta_alpha_range[0],
                    alpha_max=cfg.meta_alpha_range[1],
                    omega_min=cfg.meta_omega_range[0],
                    omega_max=cfg.meta_omega_range[1],
                    worsen_patience=cfg.meta_worsen_patience,
                    no_improve_patience=cfg.meta_no_improve_patience,
                    lr_drop=cfg.meta_lr_drop
                )
            )
        
        # Temporal Feedback Buffer
        if cfg.use_temporal_buffer:
            self._temporal_buffer = TemporalFeedbackBuffer(
                window=cfg.tfb_window,
                decay=cfg.tfb_decay,
                push=cfg.tfb_push_strength
            )
    
    def initialize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply weight initialization if enabled.
        
        Args:
            model: PyTorch model
        
        Returns:
            Initialized model (same object, modified in-place)
        """
        if not self.config.use_harmonic_init:
            return model
        
        if not HARMONIC_INIT_AVAILABLE:
            warnings.warn("Harmonic initialization requested but module not available")
            return model
        
        self._harmonic_init = HarmonicWeightInitializer(
            omega=self.config.harmonic_omega,
            phi=self.config.harmonic_phi,
            amplitude_weight=self.config.harmonic_amp_weight,
            amplitude_bias=self.config.harmonic_amp_bias,
            seed=self.config.seed,
            apply_to=self.config.harmonic_apply_to,
            mode=self.config.harmonic_mode
        )
        
        self._warmup = WarmupScheduler(warmup_epochs=self.config.warmup_epochs)
        
        self.init_report = self._harmonic_init.apply(model)
        
        if self.config.verbose and self.init_report:
            print(f"🔧 Harmonic Init: {self.init_report.layers_modified} layers, "
                  f"R²={self.init_report.pattern_fidelity:.4f}")
        
        return model
    
    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """
        Wrap optimizer (placeholder for future enhancements).
        
        Currently returns the optimizer unchanged. Future versions may
        add optimizer wrapping for additional functionality.
        
        Args:
            optimizer: PyTorch optimizer
        
        Returns:
            Wrapped optimizer
        """
        return optimizer
    
    def on_epoch_start(self, epoch: int) -> None:
        """
        Prepare for new training epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        # Reset epoch time tracking
        self.metrics['epoch_time'] = time.time()
        
        # Update warmup scheduler
        if self._warmup is not None:
            self._warmup.step(epoch)
    
    def on_train_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        global_step: int
    ) -> torch.Tensor:
        """
        Process a single training step.
        
        This method applies per-step enhancements:
        - Gradient feedback (loss modulation)
        - Curvature estimation and regularization
        - Per-iteration LR scheduling
        
        Args:
            model: Neural network model
            loss: Computed loss (before backward)
            global_step: Global training step number
        
        Returns:
            Modified loss tensor (apply backward to this)
        """
        # Gradient Feedback Controller
        if self._gradient_feedback is not None:
            feedback = self._gradient_feedback.step(loss.item(), global_step)
            if feedback != 0:
                loss = loss + feedback * torch.abs(loss)
        
        # Periodic LR Scheduler (per-iteration mode)
        if self._periodic_scheduler is not None:
            if self.config.lr_step_mode == 'iter':
                self._periodic_scheduler.step_iter()
                self.metrics['current_lr'] = self.optimizer.param_groups[0]['lr']
        
        # Curvature Regularization
        if self._curvature_reg is not None:
            fisher = self._curvature_reg.estimate_fisher(model, loss)
            curvature = self._curvature_reg.curvature
            
            # Track initial curvature for reduction metric
            if self._initial_curvature is None and curvature > 0:
                self._initial_curvature = curvature
            
            # Compute reduction percentage
            reduction = 0.0
            if self._initial_curvature and self._initial_curvature > 0:
                reduction = max(0.0, 100.0 * (self._initial_curvature - curvature) / self._initial_curvature)
            
            self.metrics['curvature'] = curvature
            self.metrics['curvature_reduction'] = reduction
            
            # Apply regularization if above threshold
            if curvature > self.config.curvature_threshold:
                loss = self._curvature_reg.apply(loss, model)
        
        return loss
    
    def on_eval_end(
        self,
        epoch: int,
        val_loss: float
    ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        """
        Process end of validation phase.
        
        This method handles:
        - Convergence analysis (improvement tracking)
        - Beta (loss variance) estimation
        - Temporal buffer updates
        - Convergence damping
        - Meta-learning adjustments
        - LR scheduling (per-epoch mode)
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
        
        Returns:
            Tuple of (action_dict, metrics_dict)
            - action_dict: 'stop', 'reduce_lr', 'checkpoint' flags
            - metrics_dict: Current metrics
        """
        cfg = self.config
        
        # Reset actions
        self.action = {'stop': False, 'reduce_lr': False, 'checkpoint': False}
        
        # --- Improvement tracking ---
        improved = val_loss < self.best_val_loss * (1.0 - cfg.ca_min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.epochs_since_improvement = 0
            self.action['checkpoint'] = True
        else:
            self.epochs_since_improvement += 1
        
        # --- Beta estimation (loss variance) ---
        if self._loss_ema is None:
            self._loss_ema = val_loss
            self._loss_sq_ema = val_loss * val_loss
        else:
            alpha = cfg.ca_ema_alpha
            self._loss_ema = (1.0 - alpha) * self._loss_ema + alpha * val_loss
            self._loss_sq_ema = (1.0 - alpha) * self._loss_sq_ema + alpha * (val_loss ** 2)
        
        variance = max(self._loss_sq_ema - self._loss_ema ** 2, 1e-12)
        beta = float(np.sqrt(variance))
        self.metrics['beta'] = beta
        
        # --- Get current scheduler params ---
        current_omega = cfg.lr_omega
        current_phi = cfg.lr_phi
        current_alpha = cfg.lr_amplitude
        
        if self._periodic_scheduler is not None:
            current_omega = self._periodic_scheduler.omega
            current_phi = self._periodic_scheduler.phi
            current_alpha = self._periodic_scheduler.amplitude
        
        # Record history
        self._beta_history.append(beta)
        self._omega_history.append(current_omega)
        self._phi_history.append(current_phi)
        
        # --- Temporal Feedback Buffer ---
        if cfg.use_temporal_buffer and self._periodic_scheduler is not None:
            gain = compute_temporal_gain(
                self._beta_history,
                self._omega_history,
                self._phi_history,
                window=cfg.tfb_window,
                decay=cfg.tfb_decay
            )
            new_alpha = suggest_amplitude_adjustment(
                current_alpha, gain, push=cfg.tfb_push_strength
            )
            if new_alpha != current_alpha:
                self._periodic_scheduler.set_params(amplitude=new_alpha)
                current_alpha = new_alpha
        
        # --- Periodic LR Scheduler (per-epoch mode) ---
        if self._periodic_scheduler is not None:
            if cfg.lr_step_mode == 'epoch':
                self._periodic_scheduler.step(epoch)
        
        self.metrics['current_lr'] = self.optimizer.param_groups[0]['lr']
        
        # --- Convergence Analysis (manual RCA if meta not enabled) ---
        if cfg.use_convergence_analysis and not cfg.use_meta_controller:
            if not improved and self.epochs_since_improvement >= cfg.ca_patience:
                if self.lr_reductions >= cfg.ca_max_lr_reductions:
                    self.action['stop'] = True
                else:
                    self.action['reduce_lr'] = True
                    self.lr_reductions += 1
                    self.epochs_since_improvement = 0
                    
                    # Apply LR reduction
                    for g in self.optimizer.param_groups:
                        g['lr'] = max(g['lr'] * cfg.ca_lr_factor, 1e-7)
                    
                    # Sync scheduler if present
                    if self._periodic_scheduler is not None:
                        new_lr = self.optimizer.param_groups[0]['lr']
                        self._periodic_scheduler.set_params(base_lr=new_lr)
                    
                    self.metrics['current_lr'] = self.optimizer.param_groups[0]['lr']
        
        # --- Meta-Learning Controller ---
        if cfg.use_meta_controller and self._meta_controller is not None:
            history = {
                'epoch': epoch,
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss,
                'beta': beta,
                'curv': self.metrics.get('curvature', 0.0),
                'r2_eiw': self.metrics.get('r2', 1.0),
                'lr': self.metrics['current_lr'],
            }
            
            decision = self._meta_controller.on_epoch_end(history, self)
            apply_changes = decision.get('apply', {})
            
            # Apply LR change
            if 'lr' in apply_changes:
                new_lr = apply_changes['lr']
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr
                if self._periodic_scheduler is not None:
                    self._periodic_scheduler.set_params(base_lr=new_lr)
                self.metrics['current_lr'] = new_lr
                self.action['reduce_lr'] = True
            
            # Apply amplitude change
            if 'alpha' in apply_changes and self._periodic_scheduler is not None:
                self._periodic_scheduler.set_params(amplitude=apply_changes['alpha'])
            
            # Apply frequency change
            if 'omega' in apply_changes and self._periodic_scheduler is not None:
                self._periodic_scheduler.set_params(omega=apply_changes['omega'])
            
            if decision.get('early_stop', False):
                self.action['stop'] = True
            
            self.metrics['meta_notes'] = decision.get('notes', '')
        
        # --- Convergence Damper ---
        if cfg.use_convergence_damper and not self.action['reduce_lr']:
            if beta < cfg.damper_beta_threshold:
                self.metrics['damper_active'] = True
                for g in self.optimizer.param_groups:
                    g['lr'] = apply_convergence_damping(
                        g['lr'], beta, alpha=cfg.damper_alpha
                    )
                if self._periodic_scheduler is not None:
                    new_lr = self.optimizer.param_groups[0]['lr']
                    self._periodic_scheduler.set_params(base_lr=new_lr)
                self.metrics['current_lr'] = self.optimizer.param_groups[0]['lr']
            else:
                self.metrics['damper_active'] = False
        
        # --- Finalize metrics ---
        self.metrics['epoch_time'] = time.time() - self.metrics['epoch_time']
        
        if self._periodic_scheduler is not None:
            self.metrics['omega'] = self._periodic_scheduler.omega
        
        self.metrics['confidence'] = 1.0 if self.action['checkpoint'] else 0.0
        
        return self.action, self.metrics
    
    @property
    def periodic_scheduler(self) -> Optional[PeriodicLRScheduler]:
        """Access to periodic LR scheduler (for meta-controller)."""
        return self._periodic_scheduler
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for final report.
        
        Returns:
            Dictionary with:
            - phase_coherence_r2: Pattern fidelity (if init applied)
            - curvature_reduction_final: Final curvature reduction %
            - init_report: Initialization report (if available)
            - meta_notes: Last meta-controller notes
        """
        init_data = None
        if self.init_report is not None:
            init_data = {
                'applied': True,
                'omega': self.init_report.omega,
                'phi': self.init_report.phi,
                'amplitude_weight': self.init_report.amplitude_weight,
                'amplitude_bias': self.init_report.amplitude_bias,
                'layers_modified': self.init_report.layers_modified,
                'parameters_modified': self.init_report.parameters_modified,
                'pattern_fidelity': round(self.init_report.pattern_fidelity, 4),
                'checksum': self.init_report.checksum
            }
        else:
            init_data = {'applied': False}
        
        return {
            'phase_coherence_r2': self.metrics['phase_coherence'],
            'curvature_reduction_final': self.metrics['curvature_reduction'],
            'initialization_report': init_data,
            'meta_notes': self.metrics.get('meta_notes', ''),
            'total_lr_reductions': self.lr_reductions,
            'best_val_loss': self.best_val_loss,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Return orchestrator state for checkpointing."""
        state = {
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'best_val_loss': self.best_val_loss,
            'epochs_since_improvement': self.epochs_since_improvement,
            'lr_reductions': self.lr_reductions,
            'loss_ema': self._loss_ema,
            'loss_sq_ema': self._loss_sq_ema,
            'beta_history': self._beta_history,
            'omega_history': self._omega_history,
            'phi_history': self._phi_history,
        }
        
        if self._periodic_scheduler is not None:
            state['periodic_scheduler'] = self._periodic_scheduler.state_dict()
        if self._gradient_feedback is not None:
            state['gradient_feedback'] = self._gradient_feedback.state_dict()
        if self._curvature_reg is not None:
            state['curvature_reg'] = self._curvature_reg.state_dict()
        if self._meta_controller is not None:
            state['meta_controller'] = self._meta_controller.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load orchestrator state from checkpoint."""
        self.metrics = state.get('metrics', self.metrics)
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.epochs_since_improvement = state.get('epochs_since_improvement', 0)
        self.lr_reductions = state.get('lr_reductions', 0)
        self._loss_ema = state.get('loss_ema')
        self._loss_sq_ema = state.get('loss_sq_ema')
        self._beta_history = state.get('beta_history', [])
        self._omega_history = state.get('omega_history', [])
        self._phi_history = state.get('phi_history', [])
        
        if 'periodic_scheduler' in state and self._periodic_scheduler is not None:
            self._periodic_scheduler.load_state_dict(state['periodic_scheduler'])
        if 'gradient_feedback' in state and self._gradient_feedback is not None:
            self._gradient_feedback.load_state_dict(state['gradient_feedback'])
        if 'curvature_reg' in state and self._curvature_reg is not None:
            self._curvature_reg.load_state_dict(state['curvature_reg'])
        if 'meta_controller' in state and self._meta_controller is not None:
            self._meta_controller.load_state_dict(state['meta_controller'])


# Backward compatibility alias
TrinityResonanceField = AdaptiveTrainingOrchestrator
TrinityField = AdaptiveTrainingOrchestrator

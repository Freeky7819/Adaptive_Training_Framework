"""
Adaptive Training Framework - Configuration
============================================

Central configuration dataclass for all training components.

This module provides a unified configuration interface that controls:
- Feature flags (enable/disable individual components)
- Hyperparameters for each module
- Training behavior and logging options

The configuration follows a hierarchical design where each component
can be independently enabled and configured, allowing for controlled
ablation studies and gradual feature adoption.

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class TrainingConfig:
    """
    Unified configuration for the Adaptive Training Framework.
    
    The framework consists of several optional components that can be
    independently enabled. This allows users to start with a baseline
    configuration and progressively add features.
    
    Component Overview:
    ------------------
    - Harmonic Weight Initialization: Applies periodic perturbations to 
      initial weights following sin(ω·log(1+t) + φ) pattern
    - Curvature Regularization: Fisher Information-based regularization
      that penalizes high-curvature regions of the loss landscape
    - Convergence Analysis: Monitors validation loss trends and triggers
      learning rate reductions or early stopping
    - Gradient Feedback Controller: Modulates gradients based on loss
      trajectory to accelerate convergence on difficult samples
    - Convergence Damper: Reduces learning rate when variance of loss
      approaches zero (near convergence)
    - Temporal Feedback Buffer: Maintains exponentially-weighted history
      of training metrics for adaptive parameter adjustment
    - Periodic LR Scheduler: Learning rate schedule with oscillating
      component: η(t) = η₀·exp(-kt)·(1 + α·sin(ω·log(1+t) + φ))
    - Meta-Learning Controller: Epoch-level hyperparameter adaptation
      based on training dynamics
    
    Attributes:
        use_harmonic_init: Enable harmonic weight initialization
        use_curvature_reg: Enable Fisher-based curvature regularization
        use_convergence_analysis: Enable convergence monitoring and early stopping
        use_gradient_feedback: Enable gradient feedback controller
        use_meta_controller: Enable epoch-level meta-learning
        use_convergence_damper: Enable LR damping near convergence
        use_temporal_buffer: Enable temporal feedback buffer
        use_periodic_lr: Enable periodic LR scheduler
        
    Example:
        >>> # Baseline configuration (all features disabled)
        >>> config = TrainingConfig()
        
        >>> # Enable only convergence analysis
        >>> config = TrainingConfig(use_convergence_analysis=True)
        
        >>> # Full configuration with all features
        >>> config = TrainingConfig(
        ...     use_harmonic_init=True,
        ...     use_curvature_reg=True,
        ...     use_convergence_analysis=True,
        ...     use_gradient_feedback=True,
        ...     use_meta_controller=True,
        ...     use_convergence_damper=True,
        ...     use_temporal_buffer=True,
        ...     use_periodic_lr=True,
        ... )
    """
    
    # ========================
    # Feature Flags
    # ========================
    use_harmonic_init: bool = False
    use_curvature_reg: bool = False
    use_convergence_analysis: bool = False
    use_gradient_feedback: bool = False
    use_meta_controller: bool = False
    use_convergence_damper: bool = False
    use_temporal_buffer: bool = False
    use_periodic_lr: bool = False
    
    # ========================
    # General Settings
    # ========================
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    verbose: bool = True
    
    # ========================
    # Harmonic Weight Initialization
    # ========================
    # Applies perturbation: w' = w · (1 + amp · sin(ω·log(1+i) + φ))
    # where i is the parameter index
    harmonic_omega: float = 6.0        # Angular frequency (rad)
    harmonic_phi: float = 1.0472       # Phase offset (≈ π/3 rad)
    harmonic_amp_weight: float = 0.02  # Amplitude for weight matrices
    harmonic_amp_bias: float = 0.01    # Amplitude for bias vectors
    harmonic_apply_to: str = "conv,linear,ln,bn,attn,qkv,proj"
    harmonic_mode: str = "sinlog"      # "sinlog" | "sinusoid" | "no-op"
    warmup_epochs: int = 1
    
    # ========================
    # Curvature Regularization (Fisher Information)
    # ========================
    # Estimates local curvature via E[∇L²] and adds regularization
    # when curvature exceeds threshold
    curvature_threshold: float = 1e-3   # Activation threshold
    curvature_strength: float = 0.05    # Regularization coefficient
    curvature_momentum: float = 0.9     # EMA momentum for Fisher estimate
    use_fisher_diagonal: bool = True    # Use diagonal Fisher approximation
    
    # ========================
    # Convergence Analysis
    # ========================
    # Monitors validation loss with optional EMA smoothing.
    # Triggers LR reduction after patience epochs without improvement.
    # Stops training after max_lr_reductions.
    ca_ema_alpha: float = 0.3           # EMA smoothing factor (0=no smoothing, 1=no memory)
    ca_patience: int = 5                # Epochs without improvement before action
    ca_min_delta: float = 0.005         # Minimum relative improvement (0.5%)
    ca_max_lr_reductions: int = 2       # Maximum LR reductions before stopping
    ca_lr_factor: float = 0.5           # LR reduction factor
    
    # ========================
    # Gradient Feedback Controller
    # ========================
    # Modulates loss: L' = L · (1 + sign(δ) · α · sin(ωt + φ))
    # where δ = EMA(L) - EMA_prev(L) indicates loss trend
    gfc_alpha: float = 0.05             # Feedback strength
    gfc_omega: float = 6.0              # Angular frequency
    gfc_phi: float = 0.3                # Phase offset
    gfc_ema_alpha: float = 0.3          # Loss EMA smoothing
    gfc_clamp: float = 0.2              # Maximum feedback magnitude
    
    # ========================
    # Convergence Damper
    # ========================
    # Reduces LR when loss variance (β) approaches zero.
    # Formula: lr' = lr · (1 - α_damp · exp(-β²/ε))
    damper_beta_threshold: float = 0.008  # Variance threshold for activation
    damper_alpha: float = 0.40            # Maximum damping factor
    damper_epsilon: float = 1e-4          # Numerical stability
    
    # ========================
    # Temporal Feedback Buffer
    # ========================
    # Maintains windowed history of metrics with exponential decay.
    # Computes: gain = (1/W) · Σ βᵢ · sin(ωᵢ·t + φᵢ) · exp(-λ(t-i))
    tfb_window: int = 12                # History window size
    tfb_decay: float = 0.05             # Exponential decay rate (λ)
    tfb_push_strength: float = 0.01     # Parameter adjustment strength
    
    # ========================
    # Periodic LR Scheduler
    # ========================
    # Formula: η(t) = η₀ · exp(-k·t) · (1 + α · sin(ω·log(1+t) + φ))
    lr_base: float = 1e-3               # Base learning rate (η₀)
    lr_decay: float = 0.003             # Exponential decay coefficient (k)
    lr_amplitude: float = 0.08          # Oscillation amplitude (α)
    lr_omega: float = 6.0               # Angular frequency (ω)
    lr_phi: float = 1.0472              # Phase offset (φ ≈ π/3)
    lr_step_mode: str = 'epoch'         # "epoch" (recommended) | "iter"
    
    # ========================
    # Meta-Learning Controller
    # ========================
    meta_min_lr: float = 1e-5
    meta_max_lr: float = 1e-2
    meta_alpha_range: tuple = (0.03, 0.12)
    meta_omega_range: tuple = (5.2, 6.8)
    meta_worsen_patience: int = 2       # Epochs of worsening before LR drop
    meta_no_improve_patience: int = 4   # Epochs without improvement → early stop
    meta_lr_drop: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    @classmethod
    def baseline(cls) -> 'TrainingConfig':
        """Create baseline configuration with all features disabled."""
        return cls()
    
    @classmethod
    def minimal(cls) -> 'TrainingConfig':
        """
        Create minimal configuration with only convergence analysis.
        Recommended starting point for evaluation.
        """
        return cls(
            use_convergence_analysis=True,
            use_convergence_damper=True,
            use_temporal_buffer=True,
            use_periodic_lr=True,
        )
    
    @classmethod
    def full(cls) -> 'TrainingConfig':
        """Create full configuration with all features enabled."""
        return cls(
            use_harmonic_init=True,
            use_curvature_reg=True,
            use_convergence_analysis=True,
            use_gradient_feedback=True,
            use_meta_controller=True,
            use_convergence_damper=True,
            use_temporal_buffer=True,
            use_periodic_lr=True,
        )
    
    def get_enabled_features(self) -> list:
        """Return list of enabled feature names."""
        features = []
        if self.use_harmonic_init:
            features.append("Harmonic Weight Initialization")
        if self.use_curvature_reg:
            features.append("Curvature Regularization")
        if self.use_convergence_analysis:
            features.append("Convergence Analysis")
        if self.use_gradient_feedback:
            features.append("Gradient Feedback Controller")
        if self.use_meta_controller:
            features.append("Meta-Learning Controller")
        if self.use_convergence_damper:
            features.append("Convergence Damper")
        if self.use_temporal_buffer:
            features.append("Temporal Feedback Buffer")
        if self.use_periodic_lr:
            features.append("Periodic LR Scheduler")
        return features
    
    @classmethod
    def nlp_minimal(cls) -> 'TrainingConfig':
        """
        NLP-optimized minimal configuration.
        
        Designed for transformer fine-tuning (BERT, GPT, etc.):
        - No LR modification (transformers have sensitive LR schedules)
        - No weight perturbation (preserves pretrained weights)
        - Convergence analysis for early stopping
        """
        return cls(
            use_harmonic_init=False,  # Don't perturb pretrained weights!
            use_curvature_reg=False,
            use_convergence_analysis=True,
            use_gradient_feedback=False,
            use_meta_controller=False,
            use_convergence_damper=False,
            use_temporal_buffer=False,
            use_periodic_lr=False,  # Don't modify transformer LR
            ca_patience=3,
            ca_min_delta=0.001,
        )
    
    @classmethod
    def nlp_full(cls) -> 'TrainingConfig':
        """
        NLP-optimized full configuration.
        
        All features enabled but with gentler settings appropriate
        for transformer fine-tuning:
        - No exponential decay (lr_decay=0)
        - Small oscillation amplitude
        - Gentle gradient feedback
        - Shorter patience for faster adaptation
        """
        return cls(
            use_harmonic_init=False,  # Don't perturb pretrained weights!
            use_curvature_reg=False,  # Too expensive for large models
            use_convergence_analysis=True,
            use_gradient_feedback=True,
            use_meta_controller=True,
            use_convergence_damper=True,
            use_temporal_buffer=True,
            use_periodic_lr=True,
            # NLP-specific gentler settings
            lr_decay=0.0,  # No exponential decay
            lr_amplitude=0.02,  # Small oscillation (vs 0.08 for vision)
            lr_omega=6.0,
            lr_step_mode='epoch',
            gf_alpha=0.02,  # Gentle feedback (vs 0.08)
            cd_threshold=0.005,
            ca_patience=2,  # Shorter patience for few epochs
            mc_patience=2,
        )


# ========================
# Preset Configurations
# ========================

PRESETS = {
    'baseline': TrainingConfig.baseline,
    'minimal': TrainingConfig.minimal,
    'full': TrainingConfig.full,
    'nlp_minimal': TrainingConfig.nlp_minimal,
    'nlp_full': TrainingConfig.nlp_full,
}


def get_preset(name: str) -> TrainingConfig:
    """
    Get a preset configuration by name.
    
    Available presets:
    - 'baseline': All features disabled (vanilla training)
    - 'minimal': Core features only (CA + damper + TFB + periodic LR)
    - 'full': All features enabled
    - 'nlp_minimal': NLP-optimized, convergence analysis only
    - 'nlp_full': NLP-optimized, all features with gentler settings
    
    Args:
        name: Preset name
        
    Returns:
        TrainingConfig instance
        
    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]()

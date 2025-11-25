"""
Adaptive Training Framework - Modules
=====================================

This package contains the individual training enhancement modules:

- GradientFeedbackController: Adaptive loss modulation based on training dynamics
- ConvergenceDamper: Learning rate reduction near convergence
- TemporalFeedbackBuffer: Exponentially-weighted metric history
- CurvatureRegularizer: Fisher Information-based regularization
- HarmonicWeightInitializer: Periodic weight perturbation
- MetaLearningController: Epoch-level hyperparameter adaptation

Each module can be used independently or combined through the
AdaptiveTrainingOrchestrator.
"""

from .gradient_feedback import GradientFeedbackController, SmartTeacher
from .convergence_damper import (
    ConvergenceDamper,
    apply_convergence_damping,
    stillness_amplifier
)
from .temporal_buffer import (
    TemporalFeedbackBuffer,
    compute_temporal_gain,
    suggest_amplitude_adjustment,
    trp_gain,
    apply_alpha_trp_push
)
from .curvature_regularizer import CurvatureRegularizer, ManifoldFlattener
from .harmonic_init import (
    HarmonicWeightInitializer,
    InitializationReport,
    WarmupScheduler,
    CMBPhaseInitializer,
    CMBReport,
    CMBWarmupScheduler
)
from .meta_controller import (
    MetaLearningController,
    MetaControllerConfig,
    AutoCoach,
    AutoCoachCfg
)

__all__ = [
    # Primary API
    'GradientFeedbackController',
    'ConvergenceDamper',
    'TemporalFeedbackBuffer',
    'CurvatureRegularizer',
    'HarmonicWeightInitializer',
    'MetaLearningController',
    'MetaControllerConfig',
    'InitializationReport',
    'WarmupScheduler',
    
    # Utility functions
    'apply_convergence_damping',
    'compute_temporal_gain',
    'suggest_amplitude_adjustment',
    
    # Backward compatibility aliases
    'SmartTeacher',
    'stillness_amplifier',
    'trp_gain',
    'apply_alpha_trp_push',
    'ManifoldFlattener',
    'CMBPhaseInitializer',
    'CMBReport',
    'CMBWarmupScheduler',
    'AutoCoach',
    'AutoCoachCfg',
]

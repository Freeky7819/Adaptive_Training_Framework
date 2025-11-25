"""
Adaptive Training Framework (ATF)
=================================

A modular framework for enhancing neural network training through
adaptive learning rate scheduling, convergence monitoring, and
gradient feedback mechanisms.

Quick Start:
------------
    from atf import TrainingConfig, AdaptiveTrainingOrchestrator
    
    # Create configuration
    config = TrainingConfig.minimal()  # or .full() for all features
    
    # Create orchestrator
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    
    # Initialize model (applies harmonic init if enabled)
    model = orchestrator.initialize_model(model)
    
    # Training loop
    for epoch in range(epochs):
        orchestrator.on_epoch_start(epoch)
        for step, batch in enumerate(train_loader):
            loss = compute_loss(model, batch)
            loss = orchestrator.on_train_step(model, loss, step)
            loss.backward()
            optimizer.step()
        
        val_loss = validate(model, val_loader)
        action, metrics = orchestrator.on_eval_end(epoch, val_loss)
        
        if action['stop']:
            break

Components:
-----------
- TrainingConfig: Central configuration dataclass
- AdaptiveTrainingOrchestrator: Main training coordinator
- ConvergenceMonitor: Checkpoint and early stopping management
- PeriodicLRScheduler: Oscillating learning rate schedule
- GradientFeedbackController: Adaptive loss modulation
- ConvergenceDamper: LR reduction near convergence
- TemporalFeedbackBuffer: Metric history with decay
- CurvatureRegularizer: Fisher Information regularization
- HarmonicWeightInitializer: Periodic weight perturbation
- MetaLearningController: Epoch-level hyperparameter adaptation

Author: Adaptive Training Framework Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Adaptive Training Framework Team"

# Core API
from .core import (
    TrainingConfig,
    AdaptiveTrainingOrchestrator,
    ConvergenceMonitor,
    ConvergenceResult,
    TrainingState,
    get_preset,
    PRESETS,
)

# Modules
from .modules import (
    GradientFeedbackController,
    ConvergenceDamper,
    TemporalFeedbackBuffer,
    CurvatureRegularizer,
    HarmonicWeightInitializer,
    MetaLearningController,
    MetaControllerConfig,
    InitializationReport,
    WarmupScheduler,
)

# Schedulers
from .schedulers import PeriodicLRScheduler

# Data utilities
from .data import (
    get_dataloaders,
    get_dataset_info,
    StratifiedBatchSampler,
    PeriodicShuffleSampler,
    get_sampler,
    create_stratified_sampler,
)

# Models
from .models import SimpleCNN, CIFAR10CNN, get_model, count_parameters

# Utilities
from .utils import TrainingLogger, MetricRegistry, format_time, format_number

# Backward compatibility aliases
from .core import TrinityResonanceField, TrinityField, ResonantCallback, RCAResult
from .modules import (
    SmartTeacher,
    ManifoldFlattener,
    CMBPhaseInitializer,
    AutoCoach,
)
from .schedulers import ResonantLRScheduler
from .data import ResonantStratifiedBatchSampler

__all__ = [
    # Version
    '__version__',
    
    # Core API
    'TrainingConfig',
    'AdaptiveTrainingOrchestrator',
    'ConvergenceMonitor',
    'ConvergenceResult',
    'TrainingState',
    'get_preset',
    'PRESETS',
    
    # Modules
    'GradientFeedbackController',
    'ConvergenceDamper',
    'TemporalFeedbackBuffer',
    'CurvatureRegularizer',
    'HarmonicWeightInitializer',
    'MetaLearningController',
    'MetaControllerConfig',
    'InitializationReport',
    'WarmupScheduler',
    
    # Schedulers
    'PeriodicLRScheduler',
    
    # Data
    'get_dataloaders',
    'get_dataset_info',
    'StratifiedBatchSampler',
    'PeriodicShuffleSampler',
    'get_sampler',
    'create_stratified_sampler',
    
    # Models
    'SimpleCNN',
    'CIFAR10CNN',
    'get_model',
    'count_parameters',
    
    # Utilities
    'TrainingLogger',
    'MetricRegistry',
    'format_time',
    'format_number',
    
    # Backward compatibility
    'TrinityResonanceField',
    'TrinityField',
    'ResonantCallback',
    'RCAResult',
    'SmartTeacher',
    'ManifoldFlattener',
    'CMBPhaseInitializer',
    'AutoCoach',
    'ResonantLRScheduler',
    'ResonantStratifiedBatchSampler',
]

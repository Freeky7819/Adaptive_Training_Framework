"""
Adaptive Training Framework - Core
==================================

Core components for the Adaptive Training Framework.
"""

from .config import TrainingConfig, get_preset, PRESETS
from .orchestrator import (
    AdaptiveTrainingOrchestrator,
    TrinityResonanceField,
    TrinityField
)
from .monitor import (
    ConvergenceMonitor,
    ConvergenceResult,
    TrainingState,
    ResonantCallback,
    RCAResult
)

__all__ = [
    # Primary API
    'TrainingConfig',
    'AdaptiveTrainingOrchestrator',
    'ConvergenceMonitor',
    'ConvergenceResult',
    'TrainingState',
    
    # Configuration
    'get_preset',
    'PRESETS',
    
    # Backward compatibility
    'TrinityResonanceField',
    'TrinityField',
    'ResonantCallback',
    'RCAResult',
]

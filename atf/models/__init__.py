"""
Adaptive Training Framework - Models
====================================

Neural network model definitions.

Vision:
- SimpleCNN: MNIST/Fashion-MNIST
- CIFAR10CNN: CIFAR-10
- CIFAR100CNN: CIFAR-100

NLP:
- BERTForSequenceClassification: BERT fine-tuning
- NanoGPT: Minimal GPT implementation
- GPT2ForLanguageModeling: HuggingFace GPT-2 wrapper
"""

from .cnn import SimpleCNN, CIFAR10CNN, CIFAR100CNN, get_model, count_parameters

# Try to import NLP models (requires transformers)
try:
    from .nlp import (
        BERTForSequenceClassification,
        NanoGPT,
        GPT2ForLanguageModeling,
        get_nlp_model
    )
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    BERTForSequenceClassification = None
    NanoGPT = None
    GPT2ForLanguageModeling = None
    get_nlp_model = None

__all__ = [
    # Vision
    'SimpleCNN',
    'CIFAR10CNN',
    'CIFAR100CNN',
    'get_model',
    'count_parameters',
    
    # NLP (if available)
    'BERTForSequenceClassification',
    'NanoGPT',
    'GPT2ForLanguageModeling',
    'get_nlp_model',
    'NLP_AVAILABLE',
]

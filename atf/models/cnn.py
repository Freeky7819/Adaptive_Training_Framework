"""
Convolutional Neural Network Models
===================================

Standard CNN architectures for image classification benchmarks.

This module provides ready-to-use CNN models for MNIST, Fashion-MNIST,
and CIFAR-10 datasets. The architectures are designed to be simple
yet effective for demonstrating training optimization techniques.

Models:
-------
- SimpleCNN: Lightweight model for MNIST/Fashion-MNIST (28x28 grayscale)
- CIFAR10CNN: Deeper model for CIFAR-10 (32x32 RGB)

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST and Fashion-MNIST.
    
    Architecture:
    - Conv(1→32, 3x3) → ReLU → MaxPool(2x2)
    - Conv(32→64, 3x3) → ReLU → MaxPool(2x2)
    - Dropout(0.25)
    - Flatten
    - Linear(64*5*5→128) → ReLU → Dropout(0.5)
    - Linear(128→10)
    
    Parameters:
        in_channels: Number of input channels (default: 1)
        num_classes: Number of output classes (default: 10)
        dropout: Dropout probability (default: 0.25)
    
    Example:
        >>> model = SimpleCNN(in_channels=1, num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([32, 10])
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.25,
        num_channels: int = None  # Alias for in_channels
    ):
        super().__init__()
        
        # Support both in_channels and num_channels
        if num_channels is not None:
            in_channels = num_channels
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 2)
        
        # For 28x28 input: after 2 pools → 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification.
    
    Architecture (VGG-style):
    - Block 1: Conv(3→64)×2 → MaxPool
    - Block 2: Conv(64→128)×2 → MaxPool
    - Block 3: Conv(128→256)×2 → MaxPool
    - FC(256*4*4→512) → FC(512→10)
    
    Uses batch normalization for training stability.
    
    Parameters:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 10)
        dropout: Dropout probability (default: 0.3)
    
    Example:
        >>> model = CIFAR10CNN(in_channels=3, num_classes=10)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([32, 10])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        # After 3 pools: 32 → 16 → 8 → 4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class CIFAR100CNN(nn.Module):
    """
    CNN for CIFAR-100 classification (100 classes).
    
    Deeper architecture than CIFAR-10 CNN for handling 100 classes.
    Uses ResNet-style skip connections for better gradient flow.
    
    Parameters:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 100)
        dropout: Dropout probability (default: 0.4)
    
    Example:
        >>> model = CIFAR100CNN()
        >>> x = torch.randn(32, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([32, 100])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 100,
        dropout: float = 0.4
    ):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 1: 64 channels
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        
        # Block 2: 128 channels
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.skip1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        
        # Block 3: 256 channels
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        self.skip2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        
        # Block 4: 512 channels
        self.conv5a = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm2d(512)
        self.skip3 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        
        self.dropout = nn.Dropout(dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Block 1 (residual)
        identity = x
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)
        
        # Block 2 (residual with downsample)
        identity = self.skip1(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = F.relu(x + identity)
        
        # Block 3 (residual with downsample)
        identity = self.skip2(x)
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = self.bn4b(self.conv4b(x))
        x = F.relu(x + identity)
        
        # Block 4 (residual with downsample)
        identity = self.skip3(x)
        x = F.relu(self.bn5a(self.conv5a(x)))
        x = self.bn5b(self.conv5b(x))
        x = F.relu(x + identity)
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating models.
    
    Args:
        name: Model name ("simple", "cifar10", "cifar100")
        in_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model arguments
    
    Returns:
        PyTorch model instance
    
    Example:
        >>> model = get_model("cifar100", in_channels=3, num_classes=100)
    """
    name = name.lower().strip()
    
    if name in ["simple", "simplecnn", "mnist"]:
        return SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.25)
        )
    
    elif name in ["cifar10", "cifar", "cifar10cnn"]:
        return CIFAR10CNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.3)
        )
    
    elif name in ["cifar100", "cifar100cnn"]:
        return CIFAR100CNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.4)
        )
    
    else:
        raise ValueError(f"Unknown model: {name}. Supported: simple, cifar10, cifar100")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

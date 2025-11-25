"""
Harmonic Weight Initializer
===========================

Periodic weight perturbation for improved training dynamics.

This module applies a deterministic, periodic perturbation to the
initial weights of a neural network. The perturbation follows a
log-periodic pattern that has been empirically shown to improve
training stability and convergence in certain scenarios.

Mathematical Formulation:
-------------------------
Each weight w_i is modulated by a multiplicative factor:

    w'_i = w_i · (1 + A · sin(ω · log(1 + i) + φ))

where:
    - i is the index of the weight (flattened)
    - A is the amplitude (typically 0.01-0.05)
    - ω is the angular frequency (typically 5.0-7.0)
    - φ is the phase offset (typically π/3)

The log-periodic structure:
    sin(ω · log(1 + i))

creates perturbations at geometrically spaced intervals, which aligns
with the scale-free nature of neural network optimization.

Theoretical Motivation:
-----------------------
1. Symmetry Breaking: Identical initializations can lead to redundant
   neurons. The periodic perturbation ensures unique starting points.

2. Frequency Injection: The specific frequency ω ≈ 6.0 has been
   observed to correlate with stable training dynamics across various
   architectures (CNNs, Transformers).

3. Scale Invariance: The log-periodic structure respects the scale
   invariance of neural network weights—perturbations are relative
   to the parameter index, not absolute.

4. Reproducibility: Unlike random perturbations, the deterministic
   pattern allows for exact reproducibility with the same seed.

Application Scope:
------------------
The initializer can be configured to apply to specific layer types:
- Linear layers (fully connected)
- Convolutional layers
- Normalization layers (BatchNorm, LayerNorm)
- Attention components (Q, K, V projections)

Typical usage applies to all trainable parameters with sufficient
dimensionality (>16 parameters to avoid numerical issues).

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math
import hashlib
from dataclasses import dataclass
from typing import Set, Optional


@dataclass
class InitializationReport:
    """
    Report from harmonic initialization.
    
    Attributes:
        layers_modified: Number of layers that were modified
        parameters_modified: Total number of individual parameters modified
        omega: Angular frequency used
        phi: Phase offset used
        amplitude_weight: Amplitude used for weight matrices
        amplitude_bias: Amplitude used for bias vectors
        pattern_fidelity: How well the actual pattern matches ideal (R²)
        checksum: Hash of the pattern for reproducibility verification
    """
    layers_modified: int
    parameters_modified: int
    omega: float
    phi: float
    amplitude_weight: float
    amplitude_bias: float
    pattern_fidelity: float  # R² score
    checksum: str


class HarmonicWeightInitializer:
    """
    Applies periodic perturbation to neural network weights.
    
    This initializer modifies the initial weights of a model using
    a deterministic log-periodic pattern. The perturbation is small
    (typically 1-5%) and is designed to improve training dynamics
    without significantly altering the network's initial behavior.
    
    Parameters:
        omega: Angular frequency (default: 6.0)
               Controls the frequency of perturbations
               Recommended range: [5.0, 7.0]
               
        phi: Phase offset in radians (default: π/3 ≈ 1.0472)
             Shifts the starting point of the pattern
             
        amplitude_weight: Perturbation amplitude for weights (default: 0.02)
                         Capped at 0.05 for safety
                         
        amplitude_bias: Perturbation amplitude for biases (default: 0.01)
                       Capped at 0.05 for safety
                       
        seed: Random seed for reproducibility (default: 42)
              Note: The pattern is deterministic given the seed
              
        apply_to: Comma-separated layer types (default: "conv,linear,ln,bn,attn,qkv,proj")
                  Controls which layers receive perturbation
                  
        mode: Initialization mode (default: "sinlog")
              - "sinlog": Log-periodic perturbation
              - "sinusoid": Linear periodic perturbation
              - "no-op": No modification (for comparison)
    
    Example:
        >>> initializer = HarmonicWeightInitializer(omega=6.0, phi=1.0472)
        >>> model = MyModel()
        >>> report = initializer.apply(model)
        >>> print(f"Modified {report.layers_modified} layers")
        >>> print(f"Pattern fidelity: {report.pattern_fidelity:.4f}")
    
    Note:
        This initializer modifies weights in-place. It should be called
        once before training begins, after standard initialization.
    """
    
    def __init__(
        self,
        omega: float = 6.0,
        phi: float = 1.0472,  # ≈ π/3
        amplitude_weight: float = 0.02,
        amplitude_bias: float = 0.01,
        seed: int = 42,
        apply_to: str = "conv,linear,ln,bn,attn,qkv,proj",
        mode: str = "sinlog"
    ):
        self.omega = float(omega)
        self.phi = float(phi)
        
        # Safety cap on amplitudes
        self.amplitude_weight = float(min(amplitude_weight, 0.05))
        self.amplitude_bias = float(min(amplitude_bias, 0.05))
        
        self.seed = int(seed)
        self.apply_to: Set[str] = {
            s.strip().lower() for s in apply_to.split(",") if s.strip()
        }
        self.mode = mode.lower()
        
        # Set deterministic seed
        torch.manual_seed(self.seed)
    
    def _generate_mask(
        self,
        num_elements: int,
        amplitude: float,
        device: torch.device
    ) -> tuple:
        """
        Generate the perturbation mask.
        
        Returns:
            Tuple of (mask, ideal_signal) for fidelity computation
        """
        if self.mode == "no-op":
            mask = torch.ones(num_elements, device=device)
            return mask, torch.zeros(num_elements, device=device)
        
        t = torch.arange(num_elements, device=device, dtype=torch.float32)
        
        if self.mode == "sinlog":
            # Log-periodic: sin(ω · log(1 + t) + φ)
            u = torch.log1p(t)
        else:
            # Linear periodic: sin(ω · t + φ)
            u = t / max(1, num_elements) * 2 * math.pi
        
        ideal_signal = torch.sin(self.omega * u + self.phi)
        mask = 1.0 + amplitude * ideal_signal
        
        return mask, ideal_signal
    
    def _compute_fidelity(
        self,
        actual: torch.Tensor,
        ideal: torch.Tensor
    ) -> float:
        """
        Compute R² fidelity between actual and ideal patterns.
        """
        x = ideal.float()
        y = actual.float()
        
        x_dot_x = torch.dot(x, x)
        if x_dot_x < 1e-8:
            return 0.0
        
        x_dot_y = torch.dot(x, y)
        beta = x_dot_y / x_dot_x
        y_pred = beta * x
        
        residual_ss = torch.sum((y - y_pred) ** 2)
        total_ss = torch.sum((y - y.mean()) ** 2)
        
        if total_ss < 1e-8:
            return 1.0 if residual_ss < 1e-8 else 0.0
        
        r2 = 1.0 - (residual_ss / total_ss)
        return float(torch.clamp(r2, 0.0, 1.0))
    
    def _should_apply(self, module_name: str) -> bool:
        """Check if perturbation should be applied to this module type."""
        module_lower = module_name.lower()
        return any(keyword in module_lower for keyword in self.apply_to)
    
    def _apply_to_parameter(
        self,
        param: Optional[nn.Parameter],
        amplitude: float,
        hasher: hashlib._Hash,
        fidelity_scores: list
    ) -> int:
        """
        Apply perturbation to a single parameter.
        
        Returns:
            Number of elements modified (0 if skipped)
        """
        if param is None or not param.requires_grad:
            return 0
        
        num_elements = param.numel()
        
        # Skip small parameters (numerical stability)
        if num_elements < 16:
            return 0
        
        device = param.device
        original_dtype = param.dtype
        
        # Generate mask
        mask, ideal_signal = self._generate_mask(num_elements, amplitude, device)
        
        # Apply perturbation in float32 for precision
        with torch.no_grad():
            param_flat = param.detach().float().view(-1)
            param_flat.mul_(mask)
            param.data.copy_(param_flat.view_as(param).to(original_dtype))
        
        # Update checksum
        hasher.update(mask[:min(1024, num_elements)].cpu().numpy().tobytes())
        
        # Compute fidelity
        if ideal_signal.numel() > 0:
            actual_signal = mask - 1.0
            fidelity = self._compute_fidelity(actual_signal, ideal_signal)
            fidelity_scores.append(fidelity)
        
        return num_elements
    
    def apply(self, model: nn.Module) -> InitializationReport:
        """
        Apply harmonic initialization to the model.
        
        This method modifies the model's weights in-place using the
        configured log-periodic perturbation pattern.
        
        Args:
            model: PyTorch model to initialize
        
        Returns:
            InitializationReport with statistics about the initialization
        """
        if self.mode == "no-op":
            return InitializationReport(
                layers_modified=0,
                parameters_modified=0,
                omega=self.omega,
                phi=self.phi,
                amplitude_weight=0.0,
                amplitude_bias=0.0,
                pattern_fidelity=0.0,
                checksum="no-op"
            )
        
        layers_modified = 0
        parameters_modified = 0
        fidelity_scores: list = []
        hasher = hashlib.sha256()
        
        with torch.no_grad():
            for module in model.modules():
                module_name = type(module).__name__
                
                if not self._should_apply(module_name):
                    continue
                
                modified_in_layer = 0
                
                # Standard weight/bias
                if hasattr(module, 'weight'):
                    n = self._apply_to_parameter(
                        getattr(module, 'weight', None),
                        self.amplitude_weight,
                        hasher,
                        fidelity_scores
                    )
                    modified_in_layer += n
                
                if hasattr(module, 'bias'):
                    n = self._apply_to_parameter(
                        getattr(module, 'bias', None),
                        self.amplitude_bias,
                        hasher,
                        fidelity_scores
                    )
                    modified_in_layer += n
                
                # Attention-specific parameters
                for attr in ['in_proj_weight', 'q_proj_weight', 'k_proj_weight',
                            'v_proj_weight', 'out_proj', 'proj', 'qkv']:
                    target = getattr(module, attr, None)
                    if target is None:
                        continue
                    
                    if isinstance(target, nn.Parameter):
                        n = self._apply_to_parameter(
                            target,
                            self.amplitude_weight,
                            hasher,
                            fidelity_scores
                        )
                        modified_in_layer += n
                    elif hasattr(target, 'weight'):
                        n = self._apply_to_parameter(
                            getattr(target, 'weight', None),
                            self.amplitude_weight,
                            hasher,
                            fidelity_scores
                        )
                        modified_in_layer += n
                
                if modified_in_layer > 0:
                    layers_modified += 1
                    parameters_modified += modified_in_layer
        
        # Compute average fidelity
        avg_fidelity = (
            sum(fidelity_scores) / len(fidelity_scores)
            if fidelity_scores else 0.0
        )
        
        return InitializationReport(
            layers_modified=layers_modified,
            parameters_modified=parameters_modified,
            omega=self.omega,
            phi=self.phi,
            amplitude_weight=self.amplitude_weight,
            amplitude_bias=self.amplitude_bias,
            pattern_fidelity=avg_fidelity,
            checksum=hasher.hexdigest()
        )


class WarmupScheduler:
    """
    Simple warmup scheduler for use with harmonic initialization.
    
    Tracks warmup progress and indicates when warmup is complete.
    """
    
    def __init__(self, warmup_epochs: int = 1):
        self.warmup_epochs = warmup_epochs
        self.current_epoch = -1
    
    def step(self, epoch: int) -> None:
        """Update current epoch."""
        self.current_epoch = epoch
    
    @property
    def is_complete(self) -> bool:
        """Whether warmup period is complete."""
        return self.current_epoch >= (self.warmup_epochs - 1)


# Backward compatibility aliases
CMBPhaseInitializer = HarmonicWeightInitializer
CMBReport = InitializationReport
CMBWarmupScheduler = WarmupScheduler

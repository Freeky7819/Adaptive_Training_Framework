"""
Basic tests for Adaptive Training Framework.

Run with: pytest tests/test_basic.py -v
"""

import pytest
import torch


class TestImports:
    """Test that all modules import correctly."""
    
    def test_core_imports(self):
        """Test core module imports."""
        from atf import (
            TrainingConfig,
            AdaptiveTrainingOrchestrator,
            ConvergenceMonitor,
        )
        assert TrainingConfig is not None
        assert AdaptiveTrainingOrchestrator is not None
        assert ConvergenceMonitor is not None
    
    def test_module_imports(self):
        """Test module imports."""
        from atf import (
            GradientFeedbackController,
            ConvergenceDamper,
            TemporalFeedbackBuffer,
            CurvatureRegularizer,
            HarmonicWeightInitializer,
            MetaLearningController,
        )
        assert GradientFeedbackController is not None
    
    def test_scheduler_imports(self):
        """Test scheduler imports."""
        from atf import PeriodicLRScheduler
        assert PeriodicLRScheduler is not None
    
    def test_data_imports(self):
        """Test data utility imports."""
        from atf import get_dataloaders, StratifiedBatchSampler
        assert get_dataloaders is not None
        assert StratifiedBatchSampler is not None
    
    def test_model_imports(self):
        """Test model imports."""
        from atf import SimpleCNN, CIFAR10CNN
        assert SimpleCNN is not None
        assert CIFAR10CNN is not None


class TestConfigs:
    """Test configuration presets."""
    
    def test_baseline_config(self):
        """Test baseline configuration."""
        from atf import TrainingConfig
        config = TrainingConfig.baseline()
        
        assert config.use_harmonic_init == False
        assert config.use_gradient_feedback == False
        assert config.use_periodic_lr == False
    
    def test_minimal_config(self):
        """Test minimal configuration."""
        from atf import TrainingConfig
        config = TrainingConfig.minimal()
        
        assert config.use_convergence_analysis == True
        assert config.use_convergence_damper == True
    
    def test_full_config(self):
        """Test full configuration."""
        from atf import TrainingConfig
        config = TrainingConfig.full()
        
        assert config.use_harmonic_init == True
        assert config.use_gradient_feedback == True
        assert config.use_periodic_lr == True
    
    def test_get_enabled_features(self):
        """Test feature listing."""
        from atf import TrainingConfig
        config = TrainingConfig.full()
        features = config.get_enabled_features()
        
        assert len(features) > 0
        assert "Gradient Feedback Controller" in features


class TestModules:
    """Test individual modules."""
    
    def test_gradient_feedback_controller(self):
        """Test gradient feedback controller."""
        from atf import GradientFeedbackController
        
        controller = GradientFeedbackController(alpha=0.05, omega=6.0)
        
        # First step returns 0 (no history)
        feedback = controller.step(loss=1.0, t=0)
        assert feedback == 0.0
        
        # Subsequent steps return non-zero
        feedback = controller.step(loss=0.9, t=1)
        assert isinstance(feedback, float)
    
    def test_periodic_lr_scheduler(self):
        """Test periodic LR scheduler."""
        from atf import PeriodicLRScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = PeriodicLRScheduler(
            optimizer,
            base_lr=1e-3,
            amplitude=0.1,
            omega=6.0
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step(1)
        new_lr = optimizer.param_groups[0]['lr']
        
        # LR should change due to oscillation
        assert isinstance(new_lr, float)
        assert new_lr > 0


class TestModels:
    """Test model architectures."""
    
    def test_simple_cnn_forward(self):
        """Test SimpleCNN forward pass."""
        from atf import SimpleCNN
        
        model = SimpleCNN(in_channels=1, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_simple_cnn_num_channels_alias(self):
        """Test num_channels parameter alias."""
        from atf import SimpleCNN
        
        model = SimpleCNN(num_channels=3, num_classes=10)
        x = torch.randn(4, 3, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_cifar10_cnn_forward(self):
        """Test CIFAR10CNN forward pass."""
        from atf import CIFAR10CNN
        
        model = CIFAR10CNN(in_channels=3, num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (4, 10)


class TestOrchestrator:
    """Test the training orchestrator."""
    
    def test_orchestrator_creation(self):
        """Test orchestrator creation."""
        from atf import TrainingConfig, AdaptiveTrainingOrchestrator
        
        config = TrainingConfig.minimal()
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
        assert orchestrator is not None
    
    def test_orchestrator_train_step(self):
        """Test orchestrator training step."""
        from atf import TrainingConfig, AdaptiveTrainingOrchestrator
        
        config = TrainingConfig(use_gradient_feedback=True)
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)
        loss = torch.nn.functional.mse_loss(model(x), y)
        
        modified_loss = orchestrator.on_train_step(model, loss, global_step=0)
        assert isinstance(modified_loss, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

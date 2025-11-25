#!/usr/bin/env python3
"""
Adaptive Training Framework - Command Line Interface
====================================================

Unified CLI for training models with the Adaptive Training Framework.

Usage:
------
    # Baseline training (no enhancements)
    python -m cli.run --dataset mnist --mode baseline --epochs 10
    
    # Minimal configuration (convergence analysis only)
    python -m cli.run --dataset cifar10 --mode minimal --epochs 50
    
    # Full configuration (all features)
    python -m cli.run --dataset mnist --mode full --epochs 20
    
    # Custom configuration
    python -m cli.run --dataset cifar10 --epochs 50 \\
        --use-convergence-analysis \\
        --use-gradient-feedback \\
        --use-periodic-lr \\
        --lr 0.001 \\
        --batch-size 128

Author: Adaptive Training Framework Team
License: MIT
"""

import argparse
import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atf.core import TrainingConfig, AdaptiveTrainingOrchestrator
from atf.data import get_dataloaders, create_stratified_sampler
from atf.models import SimpleCNN


class TrainingLogger:
    """Simple training logger with optional file output."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.log_file = log_file
        self._history = []
        
        if log_file:
            self._file = open(log_file, 'w')
        else:
            self._file = None
    
    def log(self, message: str) -> None:
        """Log a message."""
        if self.verbose:
            print(message)
        if self._file:
            self._file.write(message + '\n')
            self._file.flush()
        self._history.append(message)
    
    def record(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Record epoch metrics."""
        self._history.append({'epoch': epoch, **metrics})
    
    def close(self) -> None:
        """Close log file."""
        if self._file:
            self._file.close()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    orchestrator: Optional[AdaptiveTrainingOrchestrator] = None,
    epoch: int = 0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Neural network
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Computation device
        orchestrator: Training orchestrator (optional)
        epoch: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Calculate global step offset
    steps_per_epoch = len(loader)
    step_offset = epoch * steps_per_epoch
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Apply orchestrator enhancements
        if orchestrator is not None:
            global_step = step_offset + batch_idx
            loss = orchestrator.on_train_step(model, loss, global_step)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network
        loader: Test data loader
        criterion: Loss function
        device: Computation device
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Adaptive Training Framework CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === Dataset and Model ===
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='Test batch size')
    
    # === Training ===
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    
    # === Mode Presets ===
    parser.add_argument('--mode', type=str, default='full',
                        choices=['baseline', 'minimal', 'full', 'custom'],
                        help='Configuration preset')
    
    # === Feature Flags ===
    parser.add_argument('--use-harmonic-init', action='store_true',
                        help='Enable harmonic weight initialization')
    parser.add_argument('--use-curvature-reg', action='store_true',
                        help='Enable curvature regularization')
    parser.add_argument('--use-convergence-analysis', action='store_true',
                        help='Enable convergence analysis')
    parser.add_argument('--use-gradient-feedback', action='store_true',
                        help='Enable gradient feedback controller')
    parser.add_argument('--use-meta-controller', action='store_true',
                        help='Enable meta-learning controller')
    parser.add_argument('--use-convergence-damper', action='store_true',
                        help='Enable convergence damper')
    parser.add_argument('--use-temporal-buffer', action='store_true',
                        help='Enable temporal feedback buffer')
    parser.add_argument('--use-periodic-lr', action='store_true',
                        help='Enable periodic LR scheduler')
    
    # === Gradient Feedback Parameters ===
    parser.add_argument('--gfc-alpha', type=float, default=0.05,
                        help='Gradient feedback strength')
    parser.add_argument('--gfc-omega', type=float, default=6.0,
                        help='Gradient feedback frequency')
    parser.add_argument('--gfc-phi', type=float, default=0.3,
                        help='Gradient feedback phase')
    
    # === Periodic LR Parameters ===
    parser.add_argument('--lr-decay', type=float, default=0.003,
                        help='LR exponential decay coefficient')
    parser.add_argument('--lr-amplitude', type=float, default=0.08,
                        help='LR oscillation amplitude')
    parser.add_argument('--lr-omega', type=float, default=6.0,
                        help='LR oscillation frequency')
    parser.add_argument('--lr-phi', type=float, default=1.0472,
                        help='LR oscillation phase (default: π/3)')
    parser.add_argument('--lr-step-mode', type=str, default='epoch',
                        choices=['iter', 'epoch'],
                        help='LR update frequency')
    
    # === Convergence Damper Parameters ===
    parser.add_argument('--damper-beta-threshold', type=float, default=0.008,
                        help='Beta threshold for damper activation')
    parser.add_argument('--damper-alpha', type=float, default=0.40,
                        help='Maximum damping factor')
    
    # === Convergence Analysis Parameters ===
    parser.add_argument('--ca-patience', type=int, default=5,
                        help='Patience for LR reduction')
    parser.add_argument('--ca-min-delta', type=float, default=0.005,
                        help='Minimum improvement threshold')
    parser.add_argument('--ca-max-reductions', type=int, default=2,
                        help='Maximum LR reductions before stopping')
    parser.add_argument('--ca-ema-alpha', type=float, default=0.3,
                        help='EMA smoothing factor for validation loss')
    
    # === Sampler ===
    parser.add_argument('--sampler', type=str, default='shuffle',
                        choices=['shuffle', 'stratified'],
                        help='Batch sampling strategy')
    parser.add_argument('--sampler-alpha', type=float, default=0.0,
                        help='Log-periodic shuffle amplitude')
    
    # === Output ===
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Build TrainingConfig from command line arguments."""
    
    # Start with preset
    if args.mode == 'baseline':
        config = TrainingConfig.baseline()
    elif args.mode == 'minimal':
        config = TrainingConfig.minimal()
    elif args.mode == 'full':
        config = TrainingConfig.full()
    else:  # custom
        config = TrainingConfig()
    
    # Override with explicit flags
    if args.mode == 'custom' or args.use_harmonic_init:
        config.use_harmonic_init = args.use_harmonic_init
    if args.mode == 'custom' or args.use_curvature_reg:
        config.use_curvature_reg = args.use_curvature_reg
    if args.mode == 'custom' or args.use_convergence_analysis:
        config.use_convergence_analysis = args.use_convergence_analysis
    if args.mode == 'custom' or args.use_gradient_feedback:
        config.use_gradient_feedback = args.use_gradient_feedback
    if args.mode == 'custom' or args.use_meta_controller:
        config.use_meta_controller = args.use_meta_controller
    if args.mode == 'custom' or args.use_convergence_damper:
        config.use_convergence_damper = args.use_convergence_damper
    if args.mode == 'custom' or args.use_temporal_buffer:
        config.use_temporal_buffer = args.use_temporal_buffer
    if args.mode == 'custom' or args.use_periodic_lr:
        config.use_periodic_lr = args.use_periodic_lr
    
    # Set parameters
    config.seed = args.seed
    config.verbose = not args.quiet
    
    # Gradient feedback
    config.gfc_alpha = args.gfc_alpha
    config.gfc_omega = args.gfc_omega
    config.gfc_phi = args.gfc_phi
    
    # Periodic LR
    config.lr_base = args.lr
    config.lr_decay = args.lr_decay
    config.lr_amplitude = args.lr_amplitude
    config.lr_omega = args.lr_omega
    config.lr_phi = args.lr_phi
    config.lr_step_mode = args.lr_step_mode
    
    # Convergence damper
    config.damper_beta_threshold = args.damper_beta_threshold
    config.damper_alpha = args.damper_alpha
    
    # Convergence analysis
    config.ca_patience = args.ca_patience
    config.ca_min_delta = args.ca_min_delta
    config.ca_max_lr_reductions = args.ca_max_reductions
    config.ca_ema_alpha = args.ca_ema_alpha
    
    return config


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logger
    logger = TrainingLogger(log_file=args.log_file, verbose=not args.quiet)
    
    # Banner
    logger.log("=" * 60)
    logger.log("  ADAPTIVE TRAINING FRAMEWORK")
    logger.log("=" * 60)
    logger.log(f"Dataset: {args.dataset}")
    logger.log(f"Mode: {args.mode}")
    logger.log(f"Device: {device}")
    logger.log(f"Epochs: {args.epochs}")
    logger.log(f"Batch Size: {args.batch_size}")
    logger.log(f"Learning Rate: {args.lr}")
    logger.log("=" * 60)
    
    # Build config
    config = build_config(args)
    config.device = str(device)
    
    # Log enabled features
    features = config.get_enabled_features()
    if features:
        logger.log("Enabled features:")
        for f in features:
            logger.log(f"  ✓ {f}")
    else:
        logger.log("Mode: Baseline (no enhancements)")
    logger.log("")
    
    # Load data
    logger.log("Loading dataset...")
    
    if args.sampler == 'stratified':
        # Get dataset first for stratified sampling
        from torchvision import datasets, transforms
        if args.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = datasets.CIFAR10('./data', train=True, 
                                             download=True, transform=transform)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            ds_cls = datasets.MNIST if args.dataset == 'mnist' else datasets.FashionMNIST
            train_dataset = ds_cls('./data', train=True, download=True, transform=transform)
        
        sampler = create_stratified_sampler(
            train_dataset, 
            batch_size=args.batch_size,
            alpha=args.sampler_alpha,
            seed=args.seed
        )
        train_loader, test_loader, num_channels = get_dataloaders(
            args.dataset, args.batch_size, args.test_batch_size, sampler=sampler
        )
    else:
        train_loader, test_loader, num_channels = get_dataloaders(
            args.dataset, args.batch_size, args.test_batch_size
        )
    
    logger.log(f"Train batches: {len(train_loader)}")
    logger.log(f"Test batches: {len(test_loader)}")
    logger.log("")
    
    # Create model
    model = SimpleCNN(num_channels=num_channels, num_classes=10).to(device)
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # Create orchestrator
    orchestrator = None
    if args.mode != 'baseline':
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
        model = orchestrator.initialize_model(model)
    
    # Training loop
    logger.log("")
    logger.log("Starting training...")
    logger.log("-" * 60)
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Notify orchestrator
        if orchestrator is not None:
            orchestrator.on_epoch_start(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            orchestrator=orchestrator, epoch=epoch
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # Process epoch end
        if orchestrator is not None:
            action, metrics = orchestrator.on_eval_end(epoch, val_loss)
            current_lr = metrics['current_lr']
            beta = metrics['beta']
            
            # Check for early stopping
            if action['stop']:
                logger.log(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            current_lr = optimizer.param_groups[0]['lr']
            beta = 0.0
        
        # Update best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            marker = " ★"
        else:
            marker = ""
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        logger.log(
            f"Epoch {epoch + 1:3d}/{args.epochs} │ "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} │ "
            f"Acc: {val_acc:6.2f}%{marker} │ "
            f"LR: {current_lr:.2e} │ "
            f"β: {beta:.4f} │ "
            f"Time: {epoch_time:.1f}s"
        )
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.log("")
    logger.log("=" * 60)
    logger.log("  TRAINING COMPLETE")
    logger.log("=" * 60)
    logger.log(f"Best Accuracy: {best_acc:.2f}%")
    logger.log(f"Total Time: {total_time:.1f}s")
    
    if orchestrator is not None:
        stats = orchestrator.get_summary_stats()
        logger.log(f"LR Reductions: {stats['total_lr_reductions']}")
        logger.log(f"Best Val Loss: {stats['best_val_loss']:.6f}")
        
        if stats['initialization_report']['applied']:
            init = stats['initialization_report']
            logger.log(f"Init Pattern Fidelity: {init['pattern_fidelity']:.4f}")
    
    logger.log("=" * 60)
    
    logger.close()
    
    return best_acc


if __name__ == '__main__':
    main()

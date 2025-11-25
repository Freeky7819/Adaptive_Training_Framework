#!/usr/bin/env python3
"""
Comprehensive Benchmark Test Suite
===================================

Tests the Adaptive Training Framework across multiple benchmarks:

Vision:
- MNIST (99%+ baseline)
- Fashion-MNIST (89%+ baseline)
- CIFAR-10 (77%+ baseline)
- CIFAR-100 (45%+ baseline)
- SVHN (90%+ baseline)

NLP (requires transformers):
- GLUE SST-2 (92%+ with BERT)
- WikiText-2 Perplexity (30-50 with NanoGPT)

Run all benchmarks:
    python -m tests.benchmark_suite --all

Run specific benchmark:
    python -m tests.benchmark_suite --benchmark mnist
    python -m tests.benchmark_suite --benchmark bert_sst2
    python -m tests.benchmark_suite --benchmark nanogpt

Quick validation (fewer epochs):
    python -m tests.benchmark_suite --quick

Author: Adaptive Training Framework Team
License: MIT
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atf import (
    TrainingConfig,
    AdaptiveTrainingOrchestrator,
    get_dataloaders,
    SimpleCNN,
    CIFAR10CNN,
)
from atf.models.cnn import CIFAR100CNN


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    mode: str  # baseline, minimal, full
    accuracy: float
    best_loss: float
    epochs_trained: int
    total_time: float
    lr_reductions: int
    params: int
    timestamp: str
    config: Dict[str, Any]


VISION_BENCHMARKS = {
    "mnist": {
        "dataset": "mnist",
        "model": "simple",
        "num_classes": 10,
        "in_channels": 1,
        "epochs": {"quick": 3, "normal": 10, "full": 20},
        "batch_size": 64,
        "lr": 0.001,
        "baseline_target": 99.0,
    },
    "fashion_mnist": {
        "dataset": "fashion_mnist",
        "model": "simple",
        "num_classes": 10,
        "in_channels": 1,
        "epochs": {"quick": 3, "normal": 15, "full": 30},
        "batch_size": 64,
        "lr": 0.001,
        "baseline_target": 89.0,
    },
    "cifar10": {
        "dataset": "cifar10",
        "model": "cifar10",
        "num_classes": 10,
        "in_channels": 3,
        "epochs": {"quick": 5, "normal": 50, "full": 100},
        "batch_size": 128,
        "lr": 0.001,
        "baseline_target": 77.0,
    },
    "cifar100": {
        "dataset": "cifar100",
        "model": "cifar100",
        "num_classes": 100,
        "in_channels": 3,
        "epochs": {"quick": 5, "normal": 75, "full": 150},
        "batch_size": 128,
        "lr": 0.001,
        "baseline_target": 45.0,
    },
    "svhn": {
        "dataset": "svhn",
        "model": "cifar10",
        "num_classes": 10,
        "in_channels": 3,
        "epochs": {"quick": 3, "normal": 20, "full": 40},
        "batch_size": 128,
        "lr": 0.001,
        "baseline_target": 90.0,
    },
}

NLP_BENCHMARKS = {
    "bert_sst2": {
        "task": "sst2",
        "model": "bert-base-uncased",
        "epochs": {"quick": 1, "normal": 3, "full": 5},
        "batch_size": 32,
        "lr": 2e-5,
        "baseline_target": 92.0,
    },
    "bert_mrpc": {
        "task": "mrpc",
        "model": "bert-base-uncased",
        "epochs": {"quick": 1, "normal": 3, "full": 5},
        "batch_size": 32,
        "lr": 2e-5,
        "baseline_target": 84.0,
    },
    "nanogpt_shakespeare": {
        "dataset": "shakespeare",
        "model": "nanogpt_small",
        "epochs": {"quick": 2, "normal": 10, "full": 25},
        "batch_size": 64,
        "block_size": 64,
        "lr": 3e-4,
        "baseline_target": 1.5,  # Loss target (lower is better)
    },
    "nanogpt_wikitext": {
        "dataset": "wikitext-2",
        "model": "nanogpt_small",
        "epochs": {"quick": 1, "normal": 5, "full": 15},
        "batch_size": 32,
        "block_size": 128,
        "lr": 3e-4,
        "baseline_target": 35.0,  # Perplexity target
    },
}


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, orchestrator=None, epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    steps_per_epoch = len(loader)
    step_offset = epoch * steps_per_epoch
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        if orchestrator is not None:
            loss = orchestrator.on_train_step(model, loss, step_offset + batch_idx)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
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
    
    return total_loss / len(loader), 100.0 * correct / total


def get_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    """Get model by name."""
    if name == "simple":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif name == "cifar10":
        return CIFAR10CNN(in_channels=in_channels, num_classes=num_classes)
    elif name == "cifar100":
        return CIFAR100CNN(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


# =============================================================================
# Vision Benchmark Runner
# =============================================================================

def run_vision_benchmark(
    name: str,
    mode: str = "full",
    epoch_mode: str = "normal",
    verbose: bool = True
) -> BenchmarkResult:
    """
    Run a vision benchmark.
    
    Args:
        name: Benchmark name (mnist, fashion_mnist, cifar10, cifar100, svhn)
        mode: Training mode (baseline, minimal, full)
        epoch_mode: Epoch count (quick, normal, full)
        verbose: Print progress
    
    Returns:
        BenchmarkResult with metrics
    """
    if name not in VISION_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    
    cfg = VISION_BENCHMARKS[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg["epochs"][epoch_mode]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {name.upper()} - {mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Device: {device}, Epochs: {epochs}")
    
    # Load data
    train_loader, test_loader, _ = get_dataloaders(
        cfg["dataset"],
        batch_size=cfg["batch_size"],
        test_batch_size=cfg["batch_size"] * 2
    )
    
    # Create model
    model = get_model(
        cfg["model"],
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"]
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Setup config and orchestrator
    if mode == "baseline":
        config = TrainingConfig.baseline()
        orchestrator = None
    elif mode == "minimal":
        config = TrainingConfig.minimal()
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
        model = orchestrator.initialize_model(model)
    else:  # full
        config = TrainingConfig.full()
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
        model = orchestrator.initialize_model(model)
    
    # Training loop
    best_acc = 0.0
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        if orchestrator:
            orchestrator.on_epoch_start(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            orchestrator, epoch
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        if orchestrator:
            action, metrics = orchestrator.on_eval_end(epoch, val_loss)
            if action['stop']:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
        
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Acc={val_acc:.2f}%")
    
    total_time = time.time() - start_time
    lr_reductions = orchestrator.lr_reductions if orchestrator else 0
    
    if verbose:
        target = cfg["baseline_target"]
        status = "✅ PASS" if best_acc >= target else "❌ FAIL"
        print(f"\n  Result: {best_acc:.2f}% (target: {target}%) {status}")
        print(f"  Time: {total_time:.1f}s")
    
    return BenchmarkResult(
        name=name,
        mode=mode,
        accuracy=best_acc,
        best_loss=best_loss,
        epochs_trained=epoch + 1,
        total_time=total_time,
        lr_reductions=lr_reductions,
        params=n_params,
        timestamp=datetime.now().isoformat(),
        config=config.to_dict() if hasattr(config, 'to_dict') else {}
    )


# =============================================================================
# NLP Benchmark Runner (requires transformers)
# =============================================================================

def run_bert_benchmark(
    task: str = "sst2",
    mode: str = "full",
    epoch_mode: str = "normal",
    verbose: bool = True
) -> Optional[BenchmarkResult]:
    """Run BERT fine-tuning benchmark."""
    try:
        from atf.data.nlp_datasets import get_glue_dataloaders
        from atf.models.nlp import BERTForSequenceClassification
    except ImportError:
        if verbose:
            print("⚠️ BERT benchmark requires: pip install transformers datasets")
        return None
    
    cfg = NLP_BENCHMARKS.get(f"bert_{task}", NLP_BENCHMARKS["bert_sst2"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg["epochs"][epoch_mode]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  BERT {task.upper()} - {mode.upper()} MODE")
        print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, task_info = get_glue_dataloaders(
        task_name=task,
        model_name=cfg["model"],
        batch_size=cfg["batch_size"]
    )
    
    # Create model
    model = BERTForSequenceClassification(
        model_name=cfg["model"],
        num_labels=task_info["num_labels"]
    ).to(device)
    
    n_params = model.get_num_params()
    
    # Setup
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    if mode == "baseline":
        orchestrator = None
    else:
        config = TrainingConfig.minimal() if mode == "minimal" else TrainingConfig.full()
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    
    # Training
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        if orchestrator:
            orchestrator.on_epoch_start(epoch)
        
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = outputs["loss"]
            
            if orchestrator:
                loss = orchestrator.on_train_step(model, loss, batch_idx)
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device)
                )
                val_loss += outputs["loss"].item()
                preds = outputs["logits"].argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += batch["labels"].size(0)
        
        acc = 100.0 * correct / total
        val_loss /= len(val_loader)
        
        if orchestrator:
            action, _ = orchestrator.on_eval_end(epoch, val_loss)
            if action['stop']:
                break
        
        if acc > best_acc:
            best_acc = acc
        
        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs}: Acc={acc:.2f}%")
    
    total_time = time.time() - start_time
    
    return BenchmarkResult(
        name=f"bert_{task}",
        mode=mode,
        accuracy=best_acc,
        best_loss=val_loss,
        epochs_trained=epoch + 1,
        total_time=total_time,
        lr_reductions=orchestrator.lr_reductions if orchestrator else 0,
        params=n_params,
        timestamp=datetime.now().isoformat(),
        config={}
    )


def run_nanogpt_benchmark(
    dataset: str = "shakespeare",
    mode: str = "full",
    epoch_mode: str = "normal",
    verbose: bool = True
) -> Optional[BenchmarkResult]:
    """Run NanoGPT language modeling benchmark."""
    try:
        from atf.models.nlp import NanoGPT
        from atf.data.nlp_datasets import download_shakespeare, get_char_dataloaders
    except ImportError:
        if verbose:
            print("⚠️ NanoGPT benchmark requires: pip install transformers datasets")
        return None
    
    cfg = NLP_BENCHMARKS.get(f"nanogpt_{dataset}", NLP_BENCHMARKS["nanogpt_shakespeare"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg["epochs"][epoch_mode]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  NanoGPT {dataset.upper()} - {mode.upper()} MODE")
        print(f"{'='*60}")
    
    # Load data
    if dataset == "shakespeare":
        text_path = download_shakespeare()
        train_loader, val_loader, data_info = get_char_dataloaders(
            text_path,
            block_size=cfg["block_size"],
            batch_size=cfg["batch_size"]
        )
        vocab_size = data_info["vocab_size"]
    else:
        # WikiText uses tokenizer
        from atf.data.nlp_datasets import get_lm_dataloaders
        train_loader, val_loader, data_info = get_lm_dataloaders(
            dataset,
            block_size=cfg["block_size"],
            batch_size=cfg["batch_size"]
        )
        vocab_size = data_info["vocab_size"]
    
    # Create model
    model = NanoGPT.from_preset(
        "small",
        vocab_size=vocab_size,
        block_size=cfg["block_size"]
    ).to(device)
    
    n_params = model.get_num_params()
    
    # Setup
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    if mode == "baseline":
        orchestrator = None
    else:
        config = TrainingConfig.minimal() if mode == "minimal" else TrainingConfig.full()
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    
    # Training
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        if orchestrator:
            orchestrator.on_epoch_start(epoch)
        
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss = model(x, y)
            
            if orchestrator:
                loss = orchestrator.on_train_step(model, loss, batch_idx)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if orchestrator:
            action, _ = orchestrator.on_eval_end(epoch, val_loss)
            if action['stop']:
                break
        
        if val_loss < best_loss:
            best_loss = val_loss
        
        if verbose:
            ppl = 2.718281828 ** val_loss
            print(f"  Epoch {epoch + 1}/{epochs}: Loss={val_loss:.4f}, PPL={ppl:.2f}")
    
    total_time = time.time() - start_time
    
    return BenchmarkResult(
        name=f"nanogpt_{dataset}",
        mode=mode,
        accuracy=best_loss,  # Using loss as metric
        best_loss=best_loss,
        epochs_trained=epoch + 1,
        total_time=total_time,
        lr_reductions=orchestrator.lr_reductions if orchestrator else 0,
        params=n_params,
        timestamp=datetime.now().isoformat(),
        config={}
    )


# =============================================================================
# Main Test Suite
# =============================================================================

def run_all_benchmarks(
    modes: List[str] = ["baseline", "full"],
    epoch_mode: str = "normal",
    verbose: bool = True
) -> Dict[str, List[BenchmarkResult]]:
    """Run all benchmarks."""
    results = {}
    
    # Vision benchmarks
    for name in VISION_BENCHMARKS.keys():
        results[name] = []
        for mode in modes:
            try:
                result = run_vision_benchmark(name, mode, epoch_mode, verbose)
                results[name].append(result)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Error in {name}/{mode}: {e}")
    
    # NLP benchmarks (if available)
    for task in ["sst2", "mrpc"]:
        name = f"bert_{task}"
        results[name] = []
        for mode in modes:
            try:
                result = run_bert_benchmark(task, mode, epoch_mode, verbose)
                if result:
                    results[name].append(result)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Error in {name}/{mode}: {e}")
    
    # NanoGPT
    for dataset in ["shakespeare"]:
        name = f"nanogpt_{dataset}"
        results[name] = []
        for mode in modes:
            try:
                result = run_nanogpt_benchmark(dataset, mode, epoch_mode, verbose)
                if result:
                    results[name].append(result)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Error in {name}/{mode}: {e}")
    
    return results


def print_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<20} {'Mode':<10} {'Metric':>10} {'Time':>10} {'Status':>10}")
    print("-" * 80)
    
    for name, result_list in results.items():
        for r in result_list:
            # Determine target
            if name in VISION_BENCHMARKS:
                target = VISION_BENCHMARKS[name]["baseline_target"]
                metric = f"{r.accuracy:.2f}%"
                passed = r.accuracy >= target
            elif "nanogpt" in name:
                target = NLP_BENCHMARKS.get(name, {}).get("baseline_target", 2.0)
                metric = f"{r.accuracy:.4f}"  # Loss
                passed = r.accuracy <= target
            else:
                target = NLP_BENCHMARKS.get(name, {}).get("baseline_target", 90.0)
                metric = f"{r.accuracy:.2f}%"
                passed = r.accuracy >= target
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:<20} {r.mode:<10} {metric:>10} {r.total_time:>8.1f}s {status:>10}")
    
    print("=" * 80)


def save_results(results: Dict[str, List[BenchmarkResult]], path: str = "benchmark_results.json"):
    """Save results to JSON."""
    data = {
        name: [asdict(r) for r in result_list]
        for name, result_list in results.items()
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="ATF Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--benchmark", type=str, help="Run specific benchmark")
    parser.add_argument("--mode", type=str, default="full", 
                       choices=["baseline", "minimal", "full", "compare"])
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer epochs)")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    epoch_mode = "quick" if args.quick else "normal"
    verbose = not args.quiet
    
    if args.mode == "compare":
        modes = ["baseline", "full"]
    else:
        modes = [args.mode]
    
    if args.all:
        results = run_all_benchmarks(modes, epoch_mode, verbose)
        print_summary(results)
        save_results(results, args.output)
    
    elif args.benchmark:
        name = args.benchmark.lower()
        results = {name: []}
        
        if name in VISION_BENCHMARKS:
            for mode in modes:
                result = run_vision_benchmark(name, mode, epoch_mode, verbose)
                results[name].append(result)
        
        elif name.startswith("bert_"):
            task = name.replace("bert_", "")
            for mode in modes:
                result = run_bert_benchmark(task, mode, epoch_mode, verbose)
                if result:
                    results[name].append(result)
        
        elif name.startswith("nanogpt"):
            dataset = name.replace("nanogpt_", "") or "shakespeare"
            for mode in modes:
                result = run_nanogpt_benchmark(dataset, mode, epoch_mode, verbose)
                if result:
                    results[name].append(result)
        
        else:
            print(f"Unknown benchmark: {name}")
            print(f"Vision: {list(VISION_BENCHMARKS.keys())}")
            print(f"NLP: bert_sst2, bert_mrpc, nanogpt_shakespeare, nanogpt_wikitext")
            return
        
        print_summary(results)
    
    else:
        # Default: run quick vision benchmarks
        print("Running quick vision benchmarks...")
        results = {}
        for name in ["mnist", "cifar10"]:
            results[name] = []
            for mode in modes:
                result = run_vision_benchmark(name, mode, "quick", verbose)
                results[name].append(result)
        
        print_summary(results)


if __name__ == "__main__":
    main()

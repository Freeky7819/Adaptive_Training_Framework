#!/usr/bin/env python3
"""
BERT Fine-tuning Example with ATF
=================================

Example showing how to fine-tune BERT on GLUE tasks
using the Adaptive Training Framework.

Requirements:
    pip install transformers datasets

Usage:
    python examples/bert_glue.py --task sst2 --epochs 3
    python examples/bert_glue.py --task mrpc --mode full

Author: Adaptive Training Framework Team
License: MIT
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from atf import TrainingConfig, AdaptiveTrainingOrchestrator


def main():
    parser = argparse.ArgumentParser(description="BERT Fine-tuning with ATF")
    parser.add_argument("--task", type=str, default="sst2",
                       choices=["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "cola"])
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mode", type=str, default="minimal",
                       choices=["baseline", "minimal", "full"])
    parser.add_argument("--max-seq-length", type=int, default=128)
    args = parser.parse_args()
    
    # Check dependencies
    try:
        from atf.data.nlp_datasets import get_glue_dataloaders, GLUE_TASKS
        from atf.models.nlp import BERTForSequenceClassification
    except ImportError:
        print("Error: Please install required packages:")
        print("  pip install transformers datasets")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {args.task.upper()}")
    print(f"Mode: {args.mode}")
    print()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, task_info = get_glue_dataloaders(
        task_name=args.task,
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )
    
    print(f"Train samples: {task_info['train_size']}")
    print(f"Val samples: {task_info['val_size']}")
    print(f"Num labels: {task_info['num_labels']}")
    print()
    
    # Create model
    print("Loading BERT model...")
    model = BERTForSequenceClassification(
        model_name=args.model,
        num_labels=task_info["num_labels"]
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print()
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Setup ATF - NLP tasks need gentler settings
    if args.mode == "baseline":
        config = TrainingConfig.baseline()
        orchestrator = None
    elif args.mode == "minimal":
        # Minimal: only convergence analysis, no LR modification
        config = TrainingConfig(
            use_convergence_analysis=True,
            use_gradient_feedback=False,
            use_periodic_lr=False,  # Don't modify BERT's LR schedule
            use_convergence_damper=False,
            use_harmonic_init=False,
            ca_patience=3,
            ca_min_delta=0.001,
        )
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    else:  # full
        # Full but with NLP-appropriate settings
        config = TrainingConfig(
            use_convergence_analysis=True,
            use_gradient_feedback=True,
            use_periodic_lr=True,
            use_convergence_damper=True,
            use_harmonic_init=False,  # Don't perturb pretrained weights!
            use_curvature_regularizer=False,
            use_meta_controller=True,
            # NLP-specific: gentler settings
            lr_decay=0.0,  # No exponential decay for BERT
            lr_amplitude=0.02,  # Small oscillation
            lr_omega=6.0,
            lr_step_mode='epoch',  # Per-epoch, not per-step
            gf_alpha=0.02,  # Gentler feedback
            cd_threshold=0.005,
            ca_patience=2,
            verbose=True,
        )
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        if orchestrator:
            orchestrator.on_epoch_start(epoch)
        
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            
            loss = outputs["loss"]
            
            if orchestrator:
                step = epoch * len(train_loader) + batch_idx
                loss = orchestrator.on_train_step(model, loss, step)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
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
        
        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / total
        
        if orchestrator:
            action, metrics = orchestrator.on_eval_end(epoch, val_loss)
            lr = metrics["current_lr"]
            
            if action["stop"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            lr = optimizer.param_groups[0]["lr"]
        
        marker = " ★" if accuracy > best_acc else ""
        best_acc = max(best_acc, accuracy)
        
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Acc={accuracy:.2f}%{marker}, "
              f"LR={lr:.2e}")
    
    print("-" * 60)
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Metric: {task_info['metric']}")


if __name__ == "__main__":
    main()

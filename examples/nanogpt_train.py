#!/usr/bin/env python3
"""
NanoGPT Training Example with ATF
=================================

Example showing how to train a minimal GPT model for
character-level language modeling using the Adaptive
Training Framework.

This example trains on the tiny Shakespeare dataset.

Usage:
    python examples/nanogpt_train.py --epochs 10
    python examples/nanogpt_train.py --preset small --mode full
    python examples/nanogpt_train.py --generate --prompt "ROMEO:"

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
    parser = argparse.ArgumentParser(description="NanoGPT Training with ATF")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--preset", type=str, default="small",
                       choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--mode", type=str, default="minimal",
                       choices=["baseline", "minimal", "full"])
    parser.add_argument("--generate", action="store_true",
                       help="Generate text instead of training")
    parser.add_argument("--prompt", type=str, default="ROMEO:",
                       help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Import NLP modules
    try:
        from atf.models.nlp import NanoGPT
        from atf.data.nlp_datasets import download_shakespeare, get_char_dataloaders
    except ImportError:
        print("Error: NLP modules not available")
        return
    
    # Load data
    print("Loading Shakespeare dataset...")
    text_path = download_shakespeare()
    train_loader, val_loader, data_info = get_char_dataloaders(
        text_path,
        block_size=args.block_size,
        batch_size=args.batch_size
    )
    
    vocab_size = data_info["vocab_size"]
    stoi = data_info["stoi"]
    itos = data_info["itos"]
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train samples: {data_info['train_size']}")
    print(f"Val samples: {data_info['val_size']}")
    print()
    
    # Create model
    model = NanoGPT.from_preset(
        args.preset,
        vocab_size=vocab_size,
        block_size=args.block_size
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Generation mode
    if args.generate:
        print(f"\nGenerating from prompt: '{args.prompt}'")
        print("-" * 60)
        
        # Encode prompt
        context = torch.tensor(
            [stoi.get(ch, 0) for ch in args.prompt],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        
        # Generate
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=args.max_tokens,
                temperature=0.8,
                top_k=40
            )
        
        # Decode
        text = "".join([itos[i] for i in generated[0].tolist()])
        print(text)
        return
    
    # Training mode
    print(f"Training NanoGPT ({args.preset}) for {args.epochs} epochs")
    print(f"Mode: {args.mode}")
    print()
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Setup ATF - NanoGPT training from scratch uses standard settings
    # but with epoch-level LR scheduling for stability
    if args.mode == "baseline":
        orchestrator = None
    elif args.mode == "minimal":
        config = TrainingConfig.minimal()
        # Ensure epoch-level scheduling
        config.lr_step_mode = 'epoch'
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    else:  # full
        config = TrainingConfig.full()
        # NanoGPT from scratch can use standard settings
        # but epoch-level scheduling is more stable
        config.lr_step_mode = 'epoch'
        config.use_harmonic_init = True  # OK for training from scratch
        orchestrator = AdaptiveTrainingOrchestrator(config, optimizer)
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        if orchestrator:
            orchestrator.on_epoch_start(epoch)
        
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss = model(x, y)
            
            if orchestrator:
                step = epoch * len(train_loader) + batch_idx
                loss = orchestrator.on_train_step(model, loss, step)
            
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
            action, metrics = orchestrator.on_eval_end(epoch, val_loss)
            lr = metrics["current_lr"]
            
            if action["stop"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            lr = optimizer.param_groups[0]["lr"]
        
        # Calculate perplexity
        train_ppl = 2.718281828 ** train_loss
        val_ppl = 2.718281828 ** val_loss
        
        marker = " ★" if val_loss < best_loss else ""
        if val_loss < best_loss:
            best_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), "nanogpt_best.pt")
        
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f} (PPL={train_ppl:.2f}), "
              f"Val Loss={val_loss:.4f} (PPL={val_ppl:.2f}){marker}, "
              f"LR={lr:.2e}")
    
    print("-" * 60)
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Best Perplexity: {2.718281828 ** best_loss:.2f}")
    print()
    
    # Generate sample
    print("Sample generation:")
    print("-" * 60)
    
    context = torch.tensor(
        [stoi.get(ch, 0) for ch in "ROMEO:"],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40
        )
    
    text = "".join([itos[i] for i in generated[0].tolist()])
    print(text)


if __name__ == "__main__":
    main()

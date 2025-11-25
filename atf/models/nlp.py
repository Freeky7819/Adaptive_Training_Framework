"""
NLP Models
==========

Neural network models for Natural Language Processing tasks.

Includes:
- BERTForSequenceClassification: BERT fine-tuning wrapper
- NanoGPT: Minimal GPT implementation for language modeling
- GPTWrapper: HuggingFace GPT-2 wrapper

These models are designed to work seamlessly with the
Adaptive Training Framework orchestrator.

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BERT for Sequence Classification
# =============================================================================

class BERTForSequenceClassification(nn.Module):
    """
    BERT model wrapper for sequence classification tasks.
    
    Wraps HuggingFace BERT models for GLUE-style classification.
    Compatible with ATF orchestrator for enhanced training.
    
    Parameters:
        model_name: HuggingFace model name (default: "bert-base-uncased")
        num_labels: Number of output classes
        dropout: Classification head dropout
        cache_dir: Model cache directory
    
    Example:
        >>> model = BERTForSequenceClassification(
        ...     model_name="bert-base-uncased",
        ...     num_labels=2
        ... )
        >>> outputs = model(input_ids, attention_mask)
        >>> logits = outputs["logits"]
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )
        
        self.num_labels = num_labels
        
        # Load pretrained BERT
        self.config = AutoConfig.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.bert = AutoModel.from_pretrained(
            model_name, 
            config=self.config,
            cache_dir=cache_dir
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize classifier
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional)
            labels: Ground truth labels (optional, for loss computation)
        
        Returns:
            Dictionary with "logits" and optionally "loss"
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                # Classification
                loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# NanoGPT - Minimal GPT Implementation
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    Uses efficient scaled dot-product attention with causal mask
    to prevent attending to future tokens.
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Attention output
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Simple feed-forward network with GELU activation.
    """
    
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """
    Minimal GPT implementation for language modeling.
    
    A clean, from-scratch implementation of the GPT architecture
    suitable for learning and experimentation. Works with the
    ATF orchestrator for enhanced training.
    
    Parameters:
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        n_layer: Number of transformer layers (default: 6)
        n_head: Number of attention heads (default: 6)
        n_embd: Embedding dimension (default: 384)
        dropout: Dropout probability (default: 0.1)
    
    Model Sizes (approximate):
        - Tiny:  n_layer=4,  n_head=4,  n_embd=128  (~0.8M params)
        - Small: n_layer=6,  n_head=6,  n_embd=384  (~10M params)
        - Medium: n_layer=8, n_head=8,  n_embd=512  (~25M params)
        - Large: n_layer=12, n_head=12, n_embd=768 (~85M params)
    
    Example:
        >>> model = NanoGPT(
        ...     vocab_size=50257,  # GPT-2 vocab
        ...     block_size=128,
        ...     n_layer=6,
        ...     n_head=6,
        ...     n_embd=384
        ... )
        >>> x = torch.randint(0, 50257, (4, 128))
        >>> logits, loss = model(x)
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = self.get_num_params()
        print(f"NanoGPT initialized with {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with small values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: Exclude embedding parameters (for comparison)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices (optional, for loss)
        
        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided
        """
        device = idx.device
        B, T = idx.size()
        
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block size {self.block_size}"
        )
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal)
            top_k: Top-k sampling (None = no filtering)
        
        Returns:
            Generated token indices [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    @classmethod
    def from_preset(cls, preset: str, vocab_size: int, block_size: int = 128) -> 'NanoGPT':
        """
        Create model from preset configuration.
        
        Args:
            preset: Size preset (tiny, small, medium, large, xl)
            vocab_size: Vocabulary size
            block_size: Maximum sequence length
        
        Returns:
            NanoGPT model instance
        """
        presets = {
            "tiny":   {"n_layer": 4,  "n_head": 4,  "n_embd": 128},
            "small":  {"n_layer": 6,  "n_head": 6,  "n_embd": 384},
            "medium": {"n_layer": 8,  "n_head": 8,  "n_embd": 512},
            "large":  {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "xl":     {"n_layer": 24, "n_head": 16, "n_embd": 1024},
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        config = presets[preset]
        return cls(vocab_size=vocab_size, block_size=block_size, **config)


# =============================================================================
# GPT-2 Wrapper (HuggingFace)
# =============================================================================

class GPT2ForLanguageModeling(nn.Module):
    """
    GPT-2 wrapper for language modeling with ATF integration.
    
    Wraps HuggingFace GPT-2 for use with ATF orchestrator.
    
    Parameters:
        model_name: GPT-2 variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        cache_dir: Model cache directory
    
    Example:
        >>> model = GPT2ForLanguageModeling("gpt2")
        >>> x = torch.randint(0, 50257, (4, 128))
        >>> logits, loss = model(x, x)
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )
        
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.config = self.model.config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            labels: Target token IDs (for loss computation)
            attention_mask: Attention mask
        
        Returns:
            Tuple of (logits, loss)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss if labels is not None else None
        return outputs.logits, loss
    
    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate text using HuggingFace generate."""
        return self.model.generate(input_ids, **kwargs)


# =============================================================================
# Factory Functions
# =============================================================================

def get_nlp_model(
    model_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function for NLP models.
    
    Args:
        model_type: Model type
            - "bert": BERTForSequenceClassification
            - "nanogpt": NanoGPT (from scratch)
            - "gpt2": GPT2ForLanguageModeling (pretrained)
        **kwargs: Model-specific arguments
    
    Returns:
        PyTorch model instance
    
    Example:
        >>> # BERT for classification
        >>> model = get_nlp_model("bert", num_labels=2)
        
        >>> # NanoGPT from scratch
        >>> model = get_nlp_model("nanogpt", vocab_size=50257, n_layer=6)
        
        >>> # Pretrained GPT-2
        >>> model = get_nlp_model("gpt2", model_name="gpt2-medium")
    """
    model_type = model_type.lower()
    
    if model_type == "bert":
        return BERTForSequenceClassification(**kwargs)
    
    elif model_type in ["nanogpt", "nano", "gpt"]:
        return NanoGPT(**kwargs)
    
    elif model_type in ["gpt2", "gpt2lm"]:
        return GPT2ForLanguageModeling(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: bert, nanogpt, gpt2"
        )

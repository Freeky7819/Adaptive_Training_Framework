"""
Logging and Registry Utilities
==============================

Training logging, experiment tracking, and metric registry.

This module provides:
- TrainingLogger: Structured logging for training runs
- MetricRegistry: Thread-safe metric storage and retrieval
- Formatting utilities for console output

Author: Adaptive Training Framework Team
License: MIT
"""

from __future__ import annotations
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
import threading


class TrainingLogger:
    """
    Structured logging for training experiments.
    
    Provides formatted console output and optional file logging
    for training metrics, configuration, and progress.
    
    Parameters:
        name: Experiment name (used in log filenames)
        log_dir: Directory for log files (default: ./logs)
        verbose: Print to console (default: True)
        log_to_file: Write to log file (default: False)
        timestamp: Add timestamp to log filename (default: True)
    
    Example:
        >>> logger = TrainingLogger("cifar10_experiment")
        >>> logger.info("Starting training")
        >>> logger.log_metrics(epoch=1, loss=0.5, accuracy=0.85)
        >>> logger.finalize()
    """
    
    def __init__(
        self,
        name: str = "training",
        log_dir: str = "logs",
        verbose: bool = True,
        log_to_file: bool = False,
        timestamp: bool = True
    ):
        self.name = name
        self.verbose = verbose
        self.log_to_file = log_to_file
        
        # Setup log directory
        self.log_dir = Path(log_dir)
        if log_to_file:
            self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log filename
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = self.log_dir / f"{name}_{ts}.log"
        else:
            self.log_filename = self.log_dir / f"{name}.log"
        
        # Metric history
        self._history: List[Dict[str, Any]] = []
        self._start_time = time.time()
        
        # File handle
        self._file = None
        if log_to_file:
            self._file = open(self.log_filename, 'w')
    
    def _write(self, message: str) -> None:
        """Write message to outputs."""
        if self.verbose:
            print(message)
        if self._file:
            self._file.write(message + "\n")
            self._file.flush()
    
    def info(self, message: str) -> None:
        """Log informational message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"[{timestamp}] INFO: {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"[{timestamp}] ⚠️ WARNING: {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"[{timestamp}] ❌ ERROR: {message}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration dictionary."""
        self._write("\n" + "=" * 60)
        self._write("CONFIGURATION")
        self._write("=" * 60)
        for key, value in config.items():
            self._write(f"  {key}: {value}")
        self._write("=" * 60 + "\n")
    
    def log_metrics(self, **kwargs) -> None:
        """
        Log training metrics for current epoch/step.
        
        Args:
            **kwargs: Metric name-value pairs
        
        Example:
            >>> logger.log_metrics(
            ...     epoch=5,
            ...     train_loss=0.123,
            ...     val_loss=0.145,
            ...     accuracy=0.956,
            ...     lr=0.001
            ... )
        """
        # Add timestamp
        record = {'_timestamp': time.time(), **kwargs}
        self._history.append(record)
        
        # Format output
        epoch = kwargs.get('epoch', '?')
        parts = [f"Epoch {epoch}"]
        
        for key, value in kwargs.items():
            if key == 'epoch':
                continue
            if isinstance(value, float):
                if 'loss' in key.lower():
                    parts.append(f"{key}={value:.4f}")
                elif 'acc' in key.lower():
                    parts.append(f"{key}={value:.2%}")
                elif 'lr' in key.lower():
                    parts.append(f"{key}={value:.2e}")
                else:
                    parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        
        self._write(" | ".join(parts))
    
    def log_progress(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        bar_length: int = 30
    ) -> None:
        """
        Print progress bar.
        
        Args:
            current: Current progress value
            total: Total value
            prefix: Prefix string
            suffix: Suffix string
            bar_length: Length of progress bar
        """
        if not self.verbose:
            return
        
        percent = current / total
        filled = int(bar_length * percent)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1%} {suffix}")
        sys.stdout.flush()
        
        if current >= total:
            sys.stdout.write("\n")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Return copy of metric history."""
        return self._history.copy()
    
    def save_history(self, path: Optional[str] = None) -> None:
        """
        Save metric history to JSON file.
        
        Args:
            path: Output path (default: log_dir/name_history.json)
        """
        if path is None:
            path = self.log_dir / f"{self.name}_history.json"
        
        with open(path, 'w') as f:
            json.dump(self._history, f, indent=2)
    
    def finalize(self) -> None:
        """Finalize logging and print summary."""
        elapsed = time.time() - self._start_time
        
        self._write("\n" + "=" * 60)
        self._write("TRAINING COMPLETE")
        self._write("=" * 60)
        self._write(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        if self._history:
            # Find best metrics
            val_losses = [h.get('val_loss') for h in self._history if 'val_loss' in h]
            accuracies = [h.get('accuracy') or h.get('val_accuracy') for h in self._history]
            accuracies = [a for a in accuracies if a is not None]
            
            if val_losses:
                self._write(f"Best val_loss: {min(val_losses):.4f}")
            if accuracies:
                self._write(f"Best accuracy: {max(accuracies):.2%}")
        
        self._write("=" * 60)
        
        if self._file:
            self._file.close()
            self._file = None
    
    def __del__(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()


class MetricRegistry:
    """
    Thread-safe registry for training metrics.
    
    Provides centralized storage for metrics from multiple sources
    (training loop, callbacks, schedulers) with thread-safe access.
    
    Example:
        >>> registry = MetricRegistry()
        >>> registry.update('loss', 0.5)
        >>> registry.update('accuracy', 0.85)
        >>> print(registry.get('loss'))  # 0.5
        >>> print(registry.get_all())  # {'loss': 0.5, 'accuracy': 0.85}
    """
    
    _instance: Optional['MetricRegistry'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'MetricRegistry':
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._metrics: Dict[str, Any] = {}
                cls._instance._history: Dict[str, List] = defaultdict(list)
                cls._instance._metric_lock = threading.Lock()
            return cls._instance
    
    def update(self, key: str, value: Any, track_history: bool = True) -> None:
        """
        Update a metric value.
        
        Args:
            key: Metric name
            value: Metric value
            track_history: Whether to append to history
        """
        with self._metric_lock:
            self._metrics[key] = value
            if track_history:
                self._history[key].append(value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a metric value."""
        with self._metric_lock:
            return self._metrics.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._metric_lock:
            return self._metrics.copy()
    
    def get_history(self, key: str) -> List:
        """Get history for a specific metric."""
        with self._metric_lock:
            return self._history.get(key, []).copy()
    
    def clear(self) -> None:
        """Clear all metrics and history."""
        with self._metric_lock:
            self._metrics.clear()
            self._history.clear()
    
    def reset_history(self) -> None:
        """Clear history but keep current values."""
        with self._metric_lock:
            self._history.clear()


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(n: Union[int, float]) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n < 1000:
        return str(int(n))
    elif n < 1_000_000:
        return f"{n/1000:.1f}K"
    elif n < 1_000_000_000:
        return f"{n/1_000_000:.1f}M"
    else:
        return f"{n/1_000_000_000:.1f}B"

"""
ATF Training Server
Real-time training backend with WebSocket communication

Usage:
    pip install fastapi uvicorn websockets
    
    # Option 1: Run from atf root directory
    python gui/server.py
    
    # Option 2: Run from gui directory
    cd gui && python server.py
    
Then open atf_dashboard_live.html in browser
"""

import asyncio
import json
import time
import threading
import sys
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

# Add parent directory to path for ATF imports
# This allows running from either gui/ or root directory
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# PyTorch (optional import)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not found - using simulation mode")

# ATF imports (optional)
try:
    from atf import TrainingConfig, AdaptiveTrainingOrchestrator
    from atf.data import get_dataloaders
    from atf.models import SimpleCNN, CIFAR10CNN, CIFAR100CNN
    ATF_AVAILABLE = True
    print("✅ ATF loaded successfully")
    
    # Try to import NLP components
    try:
        from atf.data.nlp_datasets import get_glue_dataloaders, get_lm_dataloaders, get_char_dataloaders, download_shakespeare, GLUE_TASKS
        from atf.models.nlp import BERTForSequenceClassification, NanoGPT
        NLP_AVAILABLE = True
        print("✅ NLP components loaded")
    except ImportError as e:
        NLP_AVAILABLE = False
        print(f"⚠️  NLP components not available: {e}")
        
except ImportError as e:
    ATF_AVAILABLE = False
    NLP_AVAILABLE = False
    print(f"⚠️  ATF not found - using simulation mode")
    print(f"   Import error: {e}")
    print(f"   Searched in: {parent_dir}")


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TrainingState:
    status: TrainingStatus = TrainingStatus.IDLE
    epoch: int = 0
    total_epochs: int = 50
    batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    best_accuracy: float = 0.0
    best_epoch: int = 0
    learning_rate: float = 0.001
    # ATF metrics
    beta: float = 0.0
    omega: float = 6.0
    r2: float = 0.0
    confidence: float = 1.0
    stillness: bool = False
    curvature: float = 0.0
    lr_reduced: bool = False
    checkpoint: bool = False
    early_stopped: bool = False
    # NLP metrics
    perplexity: float = 0.0
    message: str = ""


class TrainingManager:
    """Manages training state and execution"""
    
    def __init__(self):
        self.state = TrainingState()
        self.config: Dict[str, Any] = {}
        self.should_stop = False
        self.should_pause = False
        self.training_thread: Optional[threading.Thread] = None
        self.websocket: Optional[WebSocket] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # PyTorch objects
        self.model = None
        self.optimizer = None
        self.orchestrator = None
        self.train_loader = None
        self.val_loader = None
        self.device = None
    
    def set_websocket(self, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.websocket = ws
        self.loop = loop
    
    async def send_state(self):
        """Send current state to websocket"""
        if self.websocket:
            try:
                await self.websocket.send_json({
                    "type": "state",
                    "data": asdict(self.state)
                })
            except:
                pass
    
    def send_state_sync(self):
        """Send state from sync context"""
        if self.websocket and self.loop:
            asyncio.run_coroutine_threadsafe(self.send_state(), self.loop)
    
    def send_log(self, message: str, level: str = "info"):
        """Send log message"""
        if self.websocket and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_json({
                    "type": "log",
                    "data": {"message": message, "level": level, "time": time.strftime("%H:%M:%S")}
                }),
                self.loop
            )
    
    def configure(self, config: Dict[str, Any]):
        """Configure training parameters"""
        self.config = config
        self.state.total_epochs = config.get("epochs", 50)
        self.send_log(f"Configuration updated: {config.get('dataset', 'unknown')} dataset")
    
    def start(self):
        """Start training in background thread"""
        if self.state.status == TrainingStatus.RUNNING:
            return
        
        self.should_stop = False
        self.should_pause = False
        
        if self.state.status != TrainingStatus.PAUSED:
            self.state = TrainingState(total_epochs=self.config.get("epochs", 50))
        
        self.state.status = TrainingStatus.RUNNING
        self.send_state_sync()
        
        # Start training thread
        if TORCH_AVAILABLE and ATF_AVAILABLE and self.config.get("useRealTraining", False):
            self.training_thread = threading.Thread(target=self._real_training_loop)
        else:
            self.training_thread = threading.Thread(target=self._simulation_loop)
        
        self.training_thread.start()
        self.send_log("Training started", "success")
    
    def pause(self):
        """Pause training"""
        self.should_pause = True
        self.state.status = TrainingStatus.PAUSED
        self.send_state_sync()
        self.send_log("Training paused", "warning")
    
    def stop(self):
        """Stop training"""
        self.should_stop = True
        self.state.status = TrainingStatus.IDLE
        self.state.epoch = 0
        self.send_state_sync()
        self.send_log("Training stopped", "warning")
    
    def _simulation_loop(self):
        """Simulated training for demo purposes"""
        import math
        import random
        
        epochs = self.config.get("epochs", 50)
        lr = self.config.get("learningRate", 0.001)
        omega = self.config.get("lrOmega", 6.0)
        amplitude = self.config.get("lrAmplitude", 0.08)
        decay = self.config.get("lrDecay", 0.003)
        use_periodic = self.config.get("usePeriodicLR", True)
        dataset = self.config.get("dataset", "mnist")
        
        # Determine if NLP dataset
        is_bert = dataset.startswith('bert_')
        is_nanogpt = dataset.startswith('nanogpt_')
        is_nlp = is_bert or is_nanogpt
        
        batches_per_epoch = 50 if is_nlp else 100  # NLP has fewer batches
        
        for epoch in range(self.state.epoch, epochs):
            if self.should_stop:
                break
            
            while self.should_pause and not self.should_stop:
                time.sleep(0.1)
            
            if self.should_stop:
                break
            
            self.state.epoch = epoch + 1
            self.state.total_batches = batches_per_epoch
            
            # Simulate batches
            epoch_loss = 0.0
            for batch in range(batches_per_epoch):
                if self.should_stop or self.should_pause:
                    break
                
                self.state.batch = batch + 1
                
                # Simulate loss decay with noise
                if is_nanogpt:
                    # Language model loss (higher, slower decay)
                    base_loss = 4.0 * math.exp(-0.03 * (epoch * batches_per_epoch + batch) / batches_per_epoch) + 1.5
                elif is_bert:
                    # BERT fine-tuning (faster convergence)
                    base_loss = 0.7 * math.exp(-0.08 * (epoch * batches_per_epoch + batch) / batches_per_epoch) + 0.15
                else:
                    # Vision
                    base_loss = 2.5 * math.exp(-0.04 * (epoch * batches_per_epoch + batch) / batches_per_epoch) + 0.2
                
                noise = random.gauss(0, 0.05 if not is_nlp else 0.02)
                if use_periodic:
                    periodic = 0.02 * math.sin(omega * math.log(1 + epoch + batch/batches_per_epoch))
                else:
                    periodic = 0
                
                self.state.train_loss = max(0.1, base_loss + noise + periodic)
                epoch_loss += self.state.train_loss
                
                # Learning rate
                base_lr = lr * math.exp(-decay * epoch)
                if use_periodic:
                    self.state.learning_rate = base_lr * (1 + amplitude * math.sin(omega * math.log(1 + epoch)))
                else:
                    self.state.learning_rate = base_lr
                
                # Send update every 10 batches
                if batch % 10 == 0:
                    self.send_state_sync()
                
                time.sleep(0.02 if not is_nlp else 0.05)  # NLP is slower
            
            if self.should_stop:
                break
            
            # End of epoch - validation
            self.state.train_loss = epoch_loss / batches_per_epoch
            self.state.val_loss = self.state.train_loss * 1.1 + random.gauss(0, 0.02)
            
            # Metrics simulation based on dataset type
            if is_nanogpt:
                # Perplexity for language models
                self.state.perplexity = max(1, math.exp(self.state.val_loss) + random.gauss(0, 0.5))
                self.state.accuracy = 0  # Not applicable
            elif is_bert:
                # BERT accuracy (classification)
                self.state.accuracy = min(95, 65 + 28 * (1 - math.exp(-0.3 * (epoch + 1))) + random.gauss(0, 1.5))
                self.state.perplexity = 0
            else:
                # Vision accuracy
                self.state.accuracy = min(99, 45 + 50 * (1 - math.exp(-0.08 * (epoch + 1))) + random.gauss(0, 1.5))
                self.state.perplexity = 0
            
            # Beta (convergence indicator)
            self.state.beta = max(0.001, 0.15 * math.exp(-0.12 * (epoch + 1)) + random.gauss(0, 0.005))
            
            # Additional ATF metrics
            self.state.omega = self.config.get("lrOmega", 6.0)
            self.state.r2 = min(1.0, max(0, 0.1 + 0.8 * (1 - math.exp(-0.1 * (epoch + 1))) + random.gauss(0, 0.05)))
            self.state.confidence = max(0, 1.0 - 0.5 * self.state.beta + random.gauss(0, 0.05))
            self.state.stillness = self.state.beta < 0.02
            self.state.curvature = max(0, 0.0001 * math.exp(-0.15 * (epoch + 1)) + random.gauss(0, 0.00001))
            self.state.lr_reduced = (epoch + 1) % 7 == 0
            
            # Track best (based on metric type)
            if is_nanogpt:
                # Lower perplexity is better
                is_best = self.state.perplexity < (self.state.best_accuracy or float('inf'))
                self.state.checkpoint = is_best
                if is_best or self.state.best_accuracy == 0:
                    self.state.best_accuracy = self.state.perplexity  # Reuse field
                    self.state.best_epoch = epoch + 1
                    self.send_log(f"★ New best perplexity: {self.state.perplexity:.2f} at epoch {epoch + 1}", "success")
            else:
                self.state.checkpoint = self.state.accuracy > self.state.best_accuracy
                if self.state.accuracy > self.state.best_accuracy:
                    self.state.best_accuracy = self.state.accuracy
                    self.state.best_epoch = epoch + 1
                    self.send_log(f"★ New best: {self.state.accuracy:.2f}% at epoch {epoch + 1}", "success")
            
            self.send_state_sync()
            status_str = f"{'STILL ' if self.state.stillness else ''}{'LR↓ ' if self.state.lr_reduced else ''}{'CKPT' if self.state.checkpoint else ''}"
            
            if is_nanogpt:
                self.send_log(f"Epoch {epoch + 1}/{epochs} | loss={self.state.train_loss:.4f} ppl={self.state.perplexity:.2f} | β={self.state.beta:.3f} | {status_str.strip()}")
            else:
                self.send_log(f"Epoch {epoch + 1}/{epochs} | loss={self.state.train_loss:.4f} acc={self.state.accuracy:.2f}% | β={self.state.beta:.3f} | {status_str.strip()}")
        
        if not self.should_stop:
            self.state.status = TrainingStatus.COMPLETED
            self.send_state_sync()
            if is_nanogpt:
                self.send_log(f"Training complete! Best perplexity: {self.state.best_accuracy:.2f}", "success")
            else:
                self.send_log(f"Training complete! Best accuracy: {self.state.best_accuracy:.2f}%", "success")
    
    def _real_training_loop(self):
        """Real PyTorch training loop"""
        try:
            self._setup_training()
            self._run_training()
        except Exception as e:
            self.state.status = TrainingStatus.ERROR
            self.state.message = str(e)
            self.send_state_sync()
            self.send_log(f"Error: {str(e)}", "error")
    
    def _setup_training(self):
        """Setup PyTorch training components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.send_log(f"Using device: {self.device}")
        
        dataset = self.config.get("dataset", "mnist")
        batch_size = self.config.get("batchSize", 128)
        
        # Determine dataset type
        self.dataset_type = 'vision'  # default
        self.is_language_model = False
        
        if dataset.startswith('bert_'):
            self.dataset_type = 'bert'
            task = dataset.replace('bert_', '')
            self._setup_bert(task, batch_size)
        elif dataset.startswith('nanogpt_'):
            self.dataset_type = 'nanogpt'
            self.is_language_model = True
            task = dataset.replace('nanogpt_', '')
            self._setup_nanogpt(task, batch_size)
        else:
            # Vision dataset
            self._setup_vision(dataset, batch_size)
        
        # Setup optimizer
        lr = self.config.get("learningRate", 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup ATF
        self._setup_atf()
    
    def _setup_vision(self, dataset, batch_size):
        """Setup vision dataset and model"""
        dataset_name = dataset.lower().replace("-", "_")
        
        self.train_loader, self.val_loader, _ = get_dataloaders(
            name=dataset_name,
            batch_size=batch_size
        )
        
        if dataset_name in ["mnist", "fashion_mnist", "fashionmnist"]:
            self.model = SimpleCNN(num_classes=10).to(self.device)
        elif dataset_name == "cifar10":
            self.model = CIFAR10CNN(num_classes=10).to(self.device)
        elif dataset_name == "cifar100":
            self.model = CIFAR100CNN(num_classes=100).to(self.device)
        elif dataset_name == "svhn":
            self.model = CIFAR10CNN(num_classes=10).to(self.device)
        elif dataset_name == "stl10":
            self.model = CIFAR100CNN(num_classes=10).to(self.device)
        else:
            self.model = SimpleCNN(num_classes=10).to(self.device)
        
        self.send_log(f"Loaded {dataset_name} (vision)")
    
    def _setup_bert(self, task, batch_size):
        """Setup BERT for GLUE task"""
        if not NLP_AVAILABLE:
            raise ImportError("NLP components not available. Install transformers and datasets.")
        
        max_seq_length = self.config.get("maxSeqLength", 128)
        
        # Get GLUE dataloaders
        self.train_loader, self.val_loader, info = get_glue_dataloaders(
            task_name=task,
            batch_size=batch_size,
            max_seq_length=max_seq_length
        )
        
        # Get number of labels for task
        num_labels = GLUE_TASKS.get(task, {}).get('num_labels', 2)
        
        # Create BERT model
        self.model = BERTForSequenceClassification(
            model_name="bert-base-uncased",
            num_labels=num_labels
        ).to(self.device)
        
        self.send_log(f"Loaded BERT for {task.upper()} (labels={num_labels})")
    
    def _setup_nanogpt(self, task, batch_size):
        """Setup NanoGPT for language modeling"""
        if not NLP_AVAILABLE:
            raise ImportError("NLP components not available. Install transformers and datasets.")
        
        block_size = self.config.get("blockSize", 256)
        
        if task == 'shakespeare':
            # Character-level Shakespeare - download first
            shakespeare_path = download_shakespeare("shakespeare.txt")
            self.train_loader, self.val_loader, info = get_char_dataloaders(
                text_path=shakespeare_path,
                block_size=block_size,
                batch_size=batch_size
            )
            vocab_size = info.get('vocab_size', 65)
        else:
            # Token-level WikiText
            dataset_name = 'wikitext-2' if task == 'wikitext2' else 'wikitext-103'
            self.train_loader, self.val_loader, info = get_lm_dataloaders(
                dataset_name=dataset_name,
                batch_size=batch_size,
                block_size=block_size
            )
            vocab_size = info.get('vocab_size', 50257)
        
        # Create NanoGPT model
        self.model = NanoGPT(
            vocab_size=vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=6,
            block_size=block_size,
            dropout=0.1
        ).to(self.device)
        
        self.send_log(f"Loaded NanoGPT for {task} (vocab={vocab_size}, block={block_size})")
    
    def _setup_atf(self):
        """Setup ATF orchestrator"""
        mode = self.config.get("mode", "full")
        if mode == "baseline":
            self.orchestrator = None
            self.send_log("Mode: baseline (no ATF)")
        else:
            # For NLP, use gentler settings
            is_nlp = self.dataset_type in ['bert', 'nanogpt']
            is_bert = self.dataset_type == 'bert'
            use_early_stop = self.config.get("useEarlyStopping", True)
            
            # BERT needs aggressive early stopping (few epochs, prone to collapse)
            default_patience = 1 if is_bert else (2 if is_nlp else 5)
            default_max_reductions = 1 if is_bert else 2
            
            atf_config = TrainingConfig(
                # Module toggles
                use_convergence_analysis=self.config.get("useConvergenceAnalysis", True),
                use_gradient_feedback=self.config.get("useGradientFeedback", True) and not is_nlp,
                use_periodic_lr=self.config.get("usePeriodicLR", True) and not is_bert,
                use_convergence_damper=self.config.get("useConvergenceDamper", True),
                use_temporal_buffer=self.config.get("useTemporalBuffer", True),
                use_harmonic_init=self.config.get("useHarmonicInit", True) and not is_nlp,
                use_meta_controller=self.config.get("useMetaController", True),
                use_curvature_reg=self.config.get("useCurvatureRegularizer", False),
                
                # Convergence Analysis + Early Stopping
                ca_patience=self.config.get("caPatience", default_patience),
                ca_min_delta=self.config.get("caMinDelta", 0.001 if is_bert else 0.005),
                ca_ema_alpha=self.config.get("caEmaAlpha", 0.3),
                ca_lr_factor=self.config.get("caLrFactor", 0.5),
                ca_max_lr_reductions=self.config.get("caMaxLrReductions", default_max_reductions) if use_early_stop else 999,
                
                # Gradient Feedback Controller
                gfc_alpha=self.config.get("gfAlpha", 0.02 if is_nlp else 0.05),
                gfc_omega=self.config.get("gfOmega", 6.0),
                gfc_phi=self.config.get("gfPhi", 0.3),
                gfc_ema_alpha=self.config.get("gfEmaAlpha", 0.3),
                gfc_clamp=self.config.get("gfClamp", 0.2),
                
                # Periodic LR
                lr_omega=self.config.get("lrOmega", 6.0),
                lr_amplitude=self.config.get("lrAmplitude", 0.02 if is_nlp else 0.08),
                lr_decay=self.config.get("lrDecay", 0.0 if is_nlp else 0.003),
                lr_phi=self.config.get("lrPhi", 1.0472),
                lr_step_mode=self.config.get("lrStepMode", "epoch"),
                
                # Convergence Damper
                damper_beta_threshold=self.config.get("cdThreshold", 0.008),
                damper_alpha=self.config.get("cdAlphaDamp", 0.4),
                damper_epsilon=self.config.get("cdEpsilon", 1e-4),
                
                # Temporal Feedback Buffer
                tfb_window=self.config.get("tbSize", 12),
                tfb_decay=self.config.get("tbDecay", 0.05),
                tfb_push_strength=self.config.get("tbPushStrength", 0.01),
                
                # Harmonic Init
                harmonic_omega=self.config.get("harmonicOmega", 6.0),
                harmonic_phi=self.config.get("harmonicPhi", 1.0472),
                harmonic_amp_weight=self.config.get("harmonicAmpWeight", 0.02),
                harmonic_amp_bias=self.config.get("harmonicAmpBias", 0.01),
                harmonic_mode=self.config.get("harmonicMode", "sinlog"),
                warmup_epochs=self.config.get("warmupEpochs", 1),
                
                # Meta Controller
                meta_worsen_patience=self.config.get("mcPatience", 2),
                meta_no_improve_patience=self.config.get("mcNoImprovePatience", 4) if use_early_stop else 999,
                meta_lr_drop=self.config.get("mcLrFactor", 0.5),
                meta_min_lr=self.config.get("mcMinLr", 1e-5),
                meta_max_lr=self.config.get("mcMaxLr", 1e-2),
                
                # Curvature Regularizer
                curvature_threshold=self.config.get("curvatureThreshold", 1e-3),
                curvature_strength=self.config.get("curvatureLambda", 0.05),
                curvature_momentum=self.config.get("curvatureMomentum", 0.9),
                use_fisher_diagonal=self.config.get("useFisherDiagonal", True),
                
                verbose=True,
            )
            self.orchestrator = AdaptiveTrainingOrchestrator(atf_config, self.optimizer)
            
            enabled = atf_config.get_enabled_features()
            early_stop_msg = "Early Stop ON" if use_early_stop else "Early Stop OFF"
            self.send_log(f"ATF: {', '.join(enabled)} | {early_stop_msg}")
    
    def _run_training(self):
        """Execute training loop"""
        import math
        
        criterion = nn.CrossEntropyLoss()
        epochs = self.config.get("epochs", 50)
        
        # Ensure dataset_type is set
        if not hasattr(self, 'dataset_type'):
            self.dataset_type = 'vision'
        if not hasattr(self, 'is_language_model'):
            self.is_language_model = False
        
        for epoch in range(self.state.epoch, epochs):
            if self.should_stop:
                break
            
            while self.should_pause and not self.should_stop:
                time.sleep(0.1)
            
            if self.should_stop:
                break
            
            self.state.epoch = epoch + 1
            
            # Epoch start
            if self.orchestrator:
                self.orchestrator.on_epoch_start(epoch)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            self.state.total_batches = len(self.train_loader)
            
            for batch_idx, batch in enumerate(self.train_loader):
                if self.should_stop or self.should_pause:
                    break
                
                self.state.batch = batch_idx + 1
                
                # Handle different batch formats
                if self.dataset_type == 'bert':
                    # BERT batch is a dict
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    token_type_ids = batch.get('token_type_ids')
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask, token_type_ids, labels)
                    loss = outputs['loss']
                    
                elif self.dataset_type == 'nanogpt':
                    # NanoGPT batch is (x, y) tensors
                    if isinstance(batch, (list, tuple)):
                        data, target = batch[0].to(self.device), batch[1].to(self.device)
                    else:
                        data, target = batch.to(self.device), batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    logits, loss = self.model(data, target)
                    
                else:
                    # Vision batch is (data, target)
                    data, target = batch[0].to(self.device), batch[1].to(self.device)
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                
                # ATF step
                if self.orchestrator:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    loss = self.orchestrator.on_train_step(self.model, loss, global_step)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                self.state.train_loss = loss.item()
                self.state.learning_rate = self.optimizer.param_groups[0]['lr']
                
                # Perplexity for LM
                if self.is_language_model:
                    self.state.perplexity = math.exp(min(loss.item(), 10))  # Cap to avoid overflow
                
                # Update every 20 batches
                if batch_idx % 20 == 0:
                    self.send_state_sync()
                
                # Intra-epoch validation (for long epochs like BERT)
                use_intra_val = self.config.get('useIntraEpochVal', False)
                intra_steps = self.config.get('intraEpochSteps', 100)
                if use_intra_val and (batch_idx + 1) % intra_steps == 0 and batch_idx > 0:
                    # Quick validation
                    self.model.eval()
                    intra_correct, intra_total = 0, 0
                    intra_batches = min(10, len(self.val_loader))  # Check only 10 batches
                    
                    with torch.no_grad():
                        for i, val_batch in enumerate(self.val_loader):
                            if i >= intra_batches:
                                break
                            
                            if self.dataset_type == 'bert':
                                input_ids = val_batch['input_ids'].to(self.device)
                                attention_mask = val_batch['attention_mask'].to(self.device)
                                labels = val_batch['labels'].to(self.device)
                                token_type_ids = val_batch.get('token_type_ids')
                                if token_type_ids is not None:
                                    token_type_ids = token_type_ids.to(self.device)
                                outputs = self.model(input_ids, attention_mask, token_type_ids, labels)
                                pred = outputs['logits'].argmax(dim=1)
                                intra_correct += pred.eq(labels).sum().item()
                                intra_total += labels.size(0)
                            else:
                                data, target = val_batch[0].to(self.device), val_batch[1].to(self.device)
                                output = self.model(data)
                                pred = output.argmax(dim=1)
                                intra_correct += pred.eq(target).sum().item()
                                intra_total += target.size(0)
                    
                    intra_acc = 100.0 * intra_correct / max(intra_total, 1)
                    self.send_log(f"⏱️ Intra-epoch val @ step {batch_idx+1}: {intra_acc:.2f}%")
                    
                    # Check for early stop within epoch
                    if intra_acc > self.state.best_accuracy:
                        self.state.best_accuracy = intra_acc
                        self.state.best_epoch = epoch + 1
                        self.send_log(f"★ New best: {intra_acc:.2f}% @ epoch {epoch+1} step {batch_idx+1}", "success")
                    
                    self.model.train()  # Back to training mode
            
            if self.should_stop:
                break
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    if self.dataset_type == 'bert':
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        token_type_ids = batch.get('token_type_ids')
                        if token_type_ids is not None:
                            token_type_ids = token_type_ids.to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask, token_type_ids, labels)
                        val_loss += outputs['loss'].item()
                        pred = outputs['logits'].argmax(dim=1)
                        correct += pred.eq(labels).sum().item()
                        total += labels.size(0)
                        
                    elif self.dataset_type == 'nanogpt':
                        if isinstance(batch, (list, tuple)):
                            data, target = batch[0].to(self.device), batch[1].to(self.device)
                        else:
                            data, target = batch.to(self.device), batch.to(self.device)
                        
                        logits, loss = self.model(data, target)
                        val_loss += loss.item()
                        # For LM, we don't compute accuracy the same way
                        total += 1
                        
                    else:
                        data, target = batch[0].to(self.device), batch[1].to(self.device)
                        output = self.model(data)
                        val_loss += criterion(output, target).item()
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
            
            self.state.train_loss = train_loss / len(self.train_loader)
            self.state.val_loss = val_loss / len(self.val_loader)
            
            if self.is_language_model:
                self.state.perplexity = math.exp(min(self.state.val_loss, 10))
                self.state.accuracy = 0  # Not applicable
            else:
                self.state.accuracy = 100.0 * correct / max(total, 1)
            
            # ATF epoch end
            if self.orchestrator:
                actions, metrics = self.orchestrator.on_eval_end(epoch, self.state.val_loss)
                self.state.beta = metrics.get('beta', 0.0)
                self.state.r2 = metrics.get('r2', 0.0)
                self.state.lr_reduced = actions.get('reduce_lr', False)
                
                # Check for early stopping (only if enabled)
                use_early_stop = self.config.get('useEarlyStopping', True)
                if use_early_stop and actions.get('stop', False):
                    self.state.early_stopped = True
                    self.state.status = TrainingStatus.COMPLETED
                    self.send_state_sync()
                    self.send_log(f"🛑 EARLY STOP: Model converged! Best={self.state.best_accuracy:.2f}% at epoch {self.state.best_epoch}", "success")
                    break
                
                if self.state.lr_reduced:
                    self.send_log(f"📉 LR reduced to {self.state.learning_rate:.2e}", "warning")
            
            # Track best
            if self.is_language_model:
                # For LM, track best perplexity (lower is better)
                if self.state.perplexity > 0 and (self.state.best_accuracy == 0 or self.state.perplexity < self.state.best_accuracy):
                    self.state.best_accuracy = self.state.perplexity
                    self.state.best_epoch = epoch + 1
                    self.state.checkpoint = True
                    self.send_log(f"★ New best perplexity: {self.state.perplexity:.2f}", "success")
                else:
                    self.state.checkpoint = False
            else:
                if self.state.accuracy > self.state.best_accuracy:
                    self.state.best_accuracy = self.state.accuracy
                    self.state.best_epoch = epoch + 1
                    self.state.checkpoint = True
                    self.send_log(f"★ New best: {self.state.accuracy:.2f}%", "success")
                else:
                    self.state.checkpoint = False
            
            self.send_state_sync()
            self.send_log(f"Epoch {epoch + 1}/{epochs}: Loss={self.state.train_loss:.4f}, Acc={self.state.accuracy:.2f}%")
        
        if not self.should_stop and not self.state.early_stopped:
            self.state.status = TrainingStatus.COMPLETED
            self.send_state_sync()
            if self.is_language_model:
                self.send_log(f"Training complete! Best perplexity: {self.state.best_accuracy:.2f}", "success")
            else:
                self.send_log(f"Training complete! Best accuracy: {self.state.best_accuracy:.2f}%", "success")


# FastAPI App
app = FastAPI(title="ATF Training Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global training manager
manager = TrainingManager()


@app.get("/")
async def root():
    return {"status": "ATF Training Server running", "torch": TORCH_AVAILABLE, "atf": ATF_AVAILABLE}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    manager.set_websocket(websocket, loop)
    
    # Send initial state
    await manager.send_state()
    
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            
            if command == "configure":
                manager.configure(data.get("config", {}))
                await manager.send_state()
            
            elif command == "start":
                manager.start()
            
            elif command == "pause":
                manager.pause()
            
            elif command == "stop":
                manager.stop()
            
            elif command == "get_state":
                await manager.send_state()
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("ATF Training Server")
    print("=" * 60)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"ATF available: {ATF_AVAILABLE}")
    print()
    print("Starting server at http://localhost:8000")
    print("Open atf_dashboard_live.html in browser")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 🖥️ ATF GUI Dashboard

Real-time training visualization and control interface for the Adaptive Training Framework.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Interface Overview](#-interface-overview)
- [Quick Config System](#-quick-config-system)
- [Presets Reference](#-presets-reference)
- [CLI Aliases](#-cli-aliases)
- [Live Architecture Diagram](#-live-architecture-diagram)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-time Metrics** | Live loss, accuracy, learning rate, β convergence |
| **6 Interactive Charts** | Train loss, val loss, accuracy, LR, β, curvature |
| **Live Architecture Diagram** | Visual module status with animations |
| **Quick Config** | CLI-style command parsing |
| **15+ Presets** | One-click configurations for common tasks |
| **Multi-dataset Support** | Vision (MNIST, CIFAR) + NLP (BERT, GPT) |
| **Export Results** | JSON export of training runs |
| **Intra-epoch Validation** | Mid-epoch checkpoints for long epochs |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn websockets torch torchvision
pip install transformers datasets  # For NLP tasks
```

### 2. Start Server

```bash
cd gui
python server.py
```

### 3. Open Dashboard

Navigate to `http://localhost:8000` in your browser.

### 4. Run Training

1. Select dataset from dropdown (e.g., CIFAR-10)
2. Choose a preset or configure manually
3. Click **Start** button
4. Watch real-time metrics and charts

---

## 📊 Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ATF Training Dashboard                                     🟢 Connected    │
├──────────────────────┬──────────────────────────────────────────────────────┤
│                      │                                                      │
│  📊 DATASET          │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  ┌────────────────┐  │   │92.4% │ │0.234 │ │0.001 │ │0.008 │ │ E15  │     │
│  │ CIFAR-10     ▼ │  │   │Best  │ │Loss  │ │ LR   │ │  β   │ │Best@ │     │
│  └────────────────┘  │   └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
│                      │                                                      │
│  ⚡ QUICK CONFIG     │   📈 CHARTS                                         │
│  ┌────────────────┐  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │ Presets...   ▼ │  │   │ Train Loss  │ │  Accuracy   │ │Learning Rate│  │
│  ├────────────────┤  │   │    ╲____    │ │      ____/  │ │   ~~~~~     │  │
│  │--omega 6.0    │  │   └─────────────┘ └─────────────┘ └─────────────┘  │
│  │--amp 0.08     │  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │--patience 5   │  │   │  Val Loss   │ │ β Converge  │ │  Curvature  │  │
│  └────────────────┘  │   │   \___/     │ │    ___      │ │     ~~~     │  │
│  [Apply] [Export]    │   └─────────────┘ └─────────────┘ └─────────────┘  │
│                      │                                                      │
│  🔧 ATF MODULES      │   🔄 ATF LIVE ARCHITECTURE                          │
│  ┌────────────────┐  │   ┌──────────────────────────────────────────────┐  │
│  │ ☑ Conv.Analysis│  │   │  Training ──▶ HarmonicInit ──▶ ConvAnalysis │  │
│  │ ☑ Gradient FB  │  │   │      │              │               │        │  │
│  │ ☑ Periodic LR  │  │   │      ▼              ▼               ▼        │  │
│  │ ☑ Conv.Damper  │  │   │  PeriodicLR    GradientFB      ConvDamper   │  │
│  │ ☑ Temp.Buffer  │  │   │      │              │               │        │  │
│  │ ☑ Harmonic Init│  │   │      └──────────────┴───────────────┘        │  │
│  │ ☑ Meta Control │  │   │                     ▼                        │  │
│  │ ☐ Curvature Reg│  │   │              MetaController ──▶ STOP?       │  │
│  └────────────────┘  │   └──────────────────────────────────────────────┘  │
│                      │                                                      │
│  ▶ START  ⏸ PAUSE   │   📋 LOG                                            │
│  ■ STOP   📥 EXPORT │   [14:32:15] Epoch 15/50 - Acc: 92.4% - Loss: 0.234 │
│                      │   [14:32:10] ★ New best: 92.4% @ epoch 15          │
│                      │   [14:31:45] LR reduced: 0.001 → 0.0005            │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

### Sections

| Section | Description |
|---------|-------------|
| **Dataset** | Select training dataset and task |
| **Quick Config** | CLI commands and presets |
| **ATF Modules** | Toggle individual modules on/off |
| **Metrics Row** | Real-time key metrics |
| **Charts** | 6 interactive time-series plots |
| **Live Architecture** | Animated module diagram |
| **Log** | Training messages and events |

---

## ⚡ Quick Config System

The Quick Config panel accepts CLI-style commands for rapid configuration.

### Usage

1. **Select Preset**: Choose from dropdown (e.g., "cifar10-atf")
2. **Or Type Command**: Enter CLI string directly
3. **Click Apply**: Parameters are instantly applied
4. **Click Export**: Copy current config as CLI string

### Supported Formats

**CLI Format:**
```bash
--omega 6.0 --amp 0.08 --patience 5 --epochs 50
```

**JSON Format:**
```json
{"lrOmega": 6.0, "lrAmplitude": 0.08, "caPatience": 5, "epochs": 50}
```

### Special Flags

| Flag | Effect |
|------|--------|
| `--baseline` | Disable ALL ATF modules (pure training) |
| `--atf` | Enable ALL ATF modules |
| `--full` | Same as `--atf` |

---

## 📋 Presets Reference

### Vision Presets

| Preset | Command | Description |
|--------|---------|-------------|
| `mnist-baseline` | `--baseline --epochs 20 --batch 128 --lr 0.001` | MNIST without ATF |
| `mnist-atf` | `--atf --epochs 20 --batch 128 --lr 0.001 --omega 6.0 --amp 0.08 --patience 5` | MNIST with full ATF |
| `cifar10-baseline` | `--baseline --epochs 50 --batch 128 --lr 0.001` | CIFAR-10 baseline |
| `cifar10-atf` | `--atf --epochs 50 --batch 128 --lr 0.001 --omega 6.0 --amp 0.08 --patience 5` | CIFAR-10 optimized |
| `cifar100-baseline` | `--baseline --epochs 75 --batch 128 --lr 0.001` | CIFAR-100 baseline |
| `cifar100-atf` | `--atf --epochs 75 --batch 128 --lr 0.001 --omega 6.0 --amp 0.08 --patience 5` | CIFAR-100 optimized |

### NLP Presets (BERT)

| Preset | Command | Description |
|--------|---------|-------------|
| `bert-baseline` | `--baseline --epochs 3 --batch 32 --lr 0.00002` | BERT fine-tuning baseline |
| `bert-atf` | `--atf --epochs 3 --batch 32 --lr 0.00002 --patience 1 --omega 6.0 --amp 0.02 --intra-val --val-steps 100` | BERT with ATF |
| `bert-safe` | `--atf --epochs 3 --batch 32 --lr 0.00002 --patience 1 --amp 0.01 --gfc off --hi off --intra-val` | BERT minimal ATF |
| `bert-mrpc` | `--atf --epochs 5 --batch 32 --lr 0.00002 --patience 3 --amp 0.01 --hi off --gfc off` | BERT MRPC optimized |

### NLP Presets (NanoGPT)

| Preset | Command | Description |
|--------|---------|-------------|
| `gpt-baseline` | `--baseline --epochs 20 --batch 64 --lr 0.0003 --block 256` | GPT baseline |
| `gpt-atf` | `--atf --epochs 20 --batch 64 --lr 0.0003 --block 256 --omega 6.0 --amp 0.04 --patience 2` | GPT with ATF |

### Experimental Presets

| Preset | Command | Description |
|--------|---------|-------------|
| `aggressive` | `--atf --omega 7.0 --amp 0.12 --decay 0.005 --patience 3 --gf 0.1` | High oscillation |
| `conservative` | `--atf --omega 5.5 --amp 0.04 --decay 0.001 --patience 8` | Stable, slow |
| `fast-converge` | `--atf --omega 6.0 --amp 0.1 --patience 3 --max-red 1` | Quick results |

---

## 📖 CLI Aliases

Complete reference of all CLI parameter aliases.

### Core ATF Parameters

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--omega`, `-w`, `--lr-omega` | `lrOmega` | 6.0 | Angular frequency |
| `--amp`, `-a`, `--lr-amp` | `lrAmplitude` | 0.08 | LR oscillation amplitude |
| `--decay`, `-k`, `--lr-decay` | `lrDecay` | 0.003 | Exponential decay rate |
| `--phi`, `--lr-phi` | `lrPhi` | 1.047 | Phase offset (π/3) |

### Early Stopping

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--patience`, `-p` | `caPatience` | 5 | Epochs before LR reduction |
| `--min-delta`, `--delta` | `caMinDelta` | 0.005 | Minimum improvement (0.5%) |
| `--lr-factor` | `caLrFactor` | 0.5 | LR multiplier on reduction |
| `--max-red`, `--max-lr-red` | `caMaxLrReductions` | 2 | Max reductions before stop |

### Gradient Feedback

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--gf-alpha`, `--gf`, `--gfa` | `gfAlpha` | 0.05 | Feedback strength |
| `--gf-omega`, `--gfw` | `gfOmega` | 6.0 | GF oscillation frequency |
| `--gf-phi` | `gfPhi` | 0.3 | Phase offset |
| `--gf-clamp` | `gfClamp` | 0.2 | Max feedback magnitude |

### Convergence Damper

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--beta`, `--cd-threshold`, `--cd-t` | `cdThreshold` | 0.008 | β threshold for damping |
| `--cd-alpha`, `--cd-damp` | `cdAlphaDamp` | 0.4 | Damping strength |

### Temporal Buffer

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--tb-size`, `--buffer` | `tbSize` | 12 | History window size |
| `--tb-decay` | `tbDecay` | 0.05 | Decay for old samples |
| `--tb-push`, `--push` | `tbPushStrength` | 0.01 | Parameter adjustment |

### Harmonic Init

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--h-omega`, `--hi-omega` | `harmonicOmega` | 6.0 | Init frequency |
| `--h-phi`, `--hi-phi` | `harmonicPhi` | 1.047 | Phase offset |
| `--h-amp`, `--hi-amp` | `harmonicAmpWeight` | 0.02 | Perturbation amplitude |
| `--warmup` | `warmupEpochs` | 1 | Warmup epochs |

### Meta Controller

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--mc-patience` | `mcPatience` | 2 | Epochs before action |
| `--mc-lr` | `mcLrFactor` | 0.5 | LR drop factor |
| `--min-lr`, `--mc-min` | `mcMinLr` | 1e-5 | Minimum LR |
| `--max-lr`, `--mc-max` | `mcMaxLr` | 1e-2 | Maximum LR |

### Curvature Regularizer

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--curv`, `--lambda`, `--curv-lambda` | `curvatureLambda` | 0.05 | Regularization strength |
| `--curv-threshold` | `curvatureThreshold` | 0.001 | Activation threshold |
| `--curv-mom` | `curvatureMomentum` | 0.9 | EMA momentum |

### Training Basics

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--epochs`, `-e` | `epochs` | 10 | Training epochs |
| `--batch`, `--bs`, `--batch-size` | `batchSize` | 128 | Batch size |
| `--lr`, `--learning-rate` | `learningRate` | 0.001 | Base learning rate |
| `--seed`, `-s` | `seed` | 42 | Random seed |

### NLP Specific

| Aliases | Config Key | Default | Description |
|---------|------------|---------|-------------|
| `--seq-len`, `--seq`, `--max-seq` | `maxSeqLength` | 128 | BERT max sequence |
| `--block-size`, `--block`, `--ctx` | `blockSize` | 256 | GPT context window |
| `--intra-val`, `--intra`, `--mid-epoch` | `useIntraEpochVal` | false | Mid-epoch validation |
| `--val-steps`, `--intra-steps` | `intraEpochSteps` | 100 | Steps between validations |

### Module Toggles

| Aliases | Config Key | Values |
|---------|------------|--------|
| `--ca` | `useConvergenceAnalysis` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--gfc` | `useGradientFeedback` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--plr` | `usePeriodicLR` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--cd` | `useConvergenceDamper` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--tb` | `useTemporalBuffer` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--hi` | `useHarmonicInit` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--mc` | `useMetaController` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--cr` | `useCurvatureRegularizer` | `on`/`off`, `true`/`false`, `1`/`0` |
| `--early-stop`, `--es` | `useEarlyStopping` | `on`/`off`, `true`/`false`, `1`/`0` |

---

## 🔄 Live Architecture Diagram

The Live Architecture section shows real-time module status with visual indicators.

### Status Indicators

| Color | Meaning |
|-------|---------|
| 🟢 Green | Active and working |
| 🟡 Yellow/Orange | Running/Processing |
| 🔴 Red | Triggered (LR reduced, early stop) |
| ⚪ Gray | Disabled |

### Animations

- **Pulse**: Module is actively processing
- **Blink**: Module just triggered an action
- **Flow lines**: Data flowing between modules

### Module Display

Each module box shows:
- Module name with icon
- Key parameter values
- Current status (active/standby/triggered/OFF)

---

## 🛠️ Troubleshooting

### Connection Issues

**Problem**: Dashboard shows "Disconnected"

**Solutions**:
1. Ensure server is running: `python server.py`
2. Check port 8000 is not in use
3. Try refreshing the page
4. Check browser console for errors

### Training Doesn't Start

**Problem**: Click Start but nothing happens

**Solutions**:
1. Check Log panel for error messages
2. Ensure dataset is downloaded
3. For BERT: check `transformers` is installed
4. Check GPU memory (BERT needs ~4GB)

### Slow Training

**Problem**: Training is slower than expected

**Solutions**:
1. Check if CUDA is available (`Using device: cuda` in log)
2. Reduce batch size if GPU memory is limited
3. Disable intra-epoch validation for vision tasks
4. Close other GPU-intensive applications

### Charts Not Updating

**Problem**: Charts are stuck

**Solutions**:
1. Training might be paused - check status
2. WebSocket might have disconnected - refresh page
3. Check if training completed or stopped early

### Export Not Working

**Problem**: Export button doesn't save file

**Solutions**:
1. Wait for at least one epoch to complete
2. Check browser's download permissions
3. Try a different browser

---

## 📁 Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI WebSocket server with training logic |
| `atf_dashboard_live.html` | Main dashboard (single-file React app) |
| `atf_dashboard_autotune.html` | AutoTune variant with hyperparameter search |
| `index.html` | Simple landing page |

---

## 🔧 Configuration

### Server Settings

Edit `server.py` to change:
- Port number (default: 8000)
- Dataset paths
- Model architectures

### Dashboard Customization

The dashboard is a single HTML file using:
- **React 18** (via CDN)
- **Recharts** for charts
- **Tailwind CSS** for styling
- **Lucide** for icons

All dependencies are loaded from CDN - no build step required.

---

## 📖 See Also

- [Main README](../README.md) - Full framework documentation
- [Examples](../examples/) - Training scripts
- [API Reference](../atf/core/orchestrator.py) - Core classes

---

<p align="center">
  <strong>Part of the Adaptive Training Framework</strong><br>
  Built with 💙 by Damjan Žakelj
</p>

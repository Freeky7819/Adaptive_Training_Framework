# 📊 ATF Benchmark Results

Comprehensive benchmark results comparing ATF against baseline training across vision and NLP tasks.

---

## 📋 Test Configuration

| Parameter | Value |
|-----------|-------|
| **Hardware** | NVIDIA RTX GPU, CUDA 12+ |
| **PyTorch** | 2.0+ |
| **Seed** | 42 (all experiments) |
| **Optimizer** | AdamW |
| **ATF Modules** | CA, GF, PLR, CD, TB, HI, MC (unless noted) |

---

## 🖼️ Vision Tasks

### MNIST (20 epochs)

| Metric | Baseline | ATF | Δ |
|--------|----------|-----|---|
| **Best Accuracy** | 99.44% | 99.33% | -0.11% |
| **Best Epoch** | 20 | 13 | -7 |
| **Training Time** | 3m 51s | 3m 53s | ≈ |
| **Early Stopped** | No | Yes (epoch 17) | ✓ |

```
Accuracy over Epochs:
100% ┤ ─────────────────────────────────────────────────── Baseline
     │                                              ████████ ATF (stopped)
 99% ┤ ████████████████████████████████████████████
     │
 98% ┤
     │
 97% ┤
     └────────────────────────────────────────────────────────
       1    3    5    7    9   11   13   15   17   19   20
                            Epoch

Verdict: TIE - MNIST is a "solved" problem (~99.5% ceiling)
         ATF correctly detected plateau and stopped early
```

---

### CIFAR-10 (50 epochs)

| Metric | Baseline | ATF | Δ |
|--------|----------|-----|---|
| **Best Accuracy** | 90.54% | 90.45% | -0.09% |
| **Best Epoch** | 43 | 26 | -17 |
| **Training Time** | 14m 11s | 9m 8s | **-36%** |
| **Early Stopped** | No | Yes | ✓ |

```
Accuracy over Epochs:
 92% ┤
     │                              ┌─────────────────────── Baseline (90.54%)
 90% ┤               ██████████████████████████████████████
     │         ██████████████████████
 88% ┤    █████                     ATF stopped @ 26 (90.45%)
     │  ██
 86% ┤██
     └────────────────────────────────────────────────────────
       1    5    10   15   20   25   30   35   40   45   50
                            Epoch

Training Time Comparison:
Baseline: ████████████████████████████████████████████████ 14m 11s
ATF:      ██████████████████████████████                   9m 8s (-36%)

Verdict: ATF WINS ⏱️ - Same accuracy, 36% faster
```

---

### CIFAR-100 (75 epochs)

| Metric | Baseline | ATF | Δ |
|--------|----------|-----|---|
| **Best Accuracy** | 68.58% | **69.33%** | **+0.75%** 🏆 |
| **Best Epoch** | 67 | 31 | -36 |
| **Training Time** | 30m 27s | 12m 36s | **-59%** |
| **Early Stopped** | No | Yes | ✓ |

```
Accuracy over Epochs:
 70% ┤                                       ┌──────────── Baseline (68.58%)
     │                              █████████
 68% ┤               ██████████████████████████████████████ ATF (69.33%) 🏆
     │         ██████
 66% ┤    █████      ATF stopped @ 31
     │  ██
 64% ┤██
     │
 62% ┤
     └────────────────────────────────────────────────────────
       1    10   20   30   40   50   60   70   75
                            Epoch

Training Time Comparison:
Baseline: ████████████████████████████████████████████████████████████ 30m 27s
ATF:      ████████████████████████                                     12m 36s (-59%)

Verdict: ATF WINS 🏆 - Better accuracy (+0.75%) AND 59% faster!
```

---

### Fashion-MNIST (20 epochs)

| Metric | Baseline | ATF | Δ |
|--------|----------|-----|---|
| **Best Accuracy** | 92.59% | 92.80% | +0.21% |
| **Training Time** | ~4m | ~3m | -23% |
| **Early Stopped** | No | Yes | ✓ |

```
Verdict: ATF WINS - Slightly better accuracy, faster training
```

---

## 📝 NLP Tasks (BERT Fine-tuning)

### SST-2 - Sentiment Analysis (5 epochs)

| Metric | Baseline | ATF | Δ |
|--------|----------|-----|---|
| **Best Accuracy** | **93.00%** | 92.43% | -0.57% |
| **Best Epoch** | 5 | 2 | -3 |
| **Training Time** | 29m 29s | 24m 6s | -18% |
| **Early Stopped** | No | Yes (epoch 2) | ⚠️ |

```
Accuracy over Epochs:
 94% ┤
     │                              ┌────────────── Baseline (93.00%) 🏆
 93% ┤                   ███████████
     │              █████
 92% ┤    ██████████      ATF (92.43%)
     │  ██           ATF stopped early!
 91% ┤██
     │
 90% ┤
     └────────────────────────────────────────
       1         2         3         4         5
                       Epoch

Analysis:
- SST-2 has 67k training samples (large dataset)
- BERT is already pre-trained and well-optimized
- ATF oscillations can destabilize fine-tuning
- Early stopping triggered too soon (patience=2)

Verdict: BASELINE WINS - Use baseline for large NLP datasets
```

---

### MRPC - Paraphrase Detection (5 epochs)

| Metric | Baseline | ATF | Δ |
|--------|----------|-----|---|
| **Best Accuracy** | 84.80% | **87.25%** | **+2.45%** 🏆 |
| **Best Epoch** | 2 | 3 | +1 |
| **Training Time** | 5m 49s | 4m 54s | -16% |
| **Early Stopped** | No | No | - |

```
Accuracy over Epochs:
 88% ┤                   ┌──────────────────── ATF (87.25%) 🏆
     │              █████
 86% ┤         █████
     │    █████       ┌──────────────────────── Baseline (84.80%)
 84% ┤████████████████
     │
 82% ┤
     └────────────────────────────────────────
       1         2         3         4         5
                       Epoch

Analysis:
- MRPC has only 3.7k training samples (small dataset)
- High risk of overfitting
- ATF oscillations act as regularization
- Convergence damper prevents overfitting

Verdict: ATF WINS 🏆 - +2.45% accuracy on small dataset!
```

---

## 📈 Summary Charts

### Overall Results

```
                        Accuracy Comparison
                 
CIFAR-100    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 69.33% ATF 🏆
             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 68.58% Base

CIFAR-10     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 90.45% ATF
             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 90.54% Base

MNIST        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 99.33% ATF
             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 99.44% Base

BERT MRPC    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 87.25% ATF 🏆
             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 84.80% Base

BERT SST-2   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 92.78% ATF
             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 93.23% Base 🏆

             0%       25%       50%       75%      100%
```

### Time Savings

```
                        Training Time Reduction

CIFAR-100    ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ -59% ⚡⚡⚡
CIFAR-10     █████████████████████████████░░░░░░░░░░░░░░░░░░░░ -36% ⚡⚡
F-MNIST      ██████████████████████████████████░░░░░░░░░░░░░░░ -23% ⚡
BERT SST-2   ███████████████████████████████████████░░░░░░░░░░ -18%
BERT MRPC    ████████████████████████████████████████░░░░░░░░░ -16%
MNIST        ██████████████████████████████████████████████████  ≈0%

             0%                                               100%
             ◀──── Time Saved ────▶                    Full Time
```

---

## 🏆 Win/Loss Summary

| Dataset | Size | Complexity | Winner | Why |
|---------|------|------------|--------|-----|
| CIFAR-100 | 50k | High (100 classes) | **ATF** 🏆 | +0.75% acc, -59% time |
| CIFAR-10 | 50k | Medium | **ATF** ⏱️ | Same acc, -36% time |
| MNIST | 60k | Low | Tie | Solved problem |
| Fashion-MNIST | 60k | Low-Med | **ATF** | +0.21%, -23% time |
| BERT MRPC | 3.7k | High (small data) | **ATF** 🏆 | +2.45% acc |
| BERT SST-2 | 67k | Low (large data) | Baseline | ATF early-stopped too soon |

### Score

```
ATF Wins:      4  (CIFAR-100, CIFAR-10, Fashion-MNIST, MRPC)
Baseline Wins: 1  (SST-2)
Ties:          1  (MNIST)

ATF Success Rate: 67%
```

---

## 💡 Key Insights

### When to Use ATF

✅ **Use ATF for:**
- Complex tasks with many classes (CIFAR-100)
- Small datasets prone to overfitting (MRPC)
- Training from scratch (not fine-tuning)
- When you need faster training with early stopping

❌ **Use Baseline for:**
- Pre-trained models on large datasets (BERT on SST-2)
- Already well-optimized training setups
- Very simple tasks (MNIST-level)

### ATF Module Recommendations

| Scenario | Recommended Modules |
|----------|---------------------|
| Vision (general) | CA + PLR + MC + CD + GF + TB + HI |
| BERT (small data) | CA + MC + CD + PLR (no HI, GF, TB) |
| BERT (large data) | Baseline (no ATF) |
| Aggressive | All + high ω (7.0) and amp (0.12) |
| Conservative | CA + MC + PLR + CD with low amp (0.04) |

---

## 🔬 Training Collapse Prevention

One of ATF's key strengths is preventing catastrophic training collapse.

### Example: CIFAR-10 Collapse Scenario

Under certain conditions (bad initialization, aggressive LR), baseline training can collapse:

```
Without ATF (collapsed run):
Epoch  1: 45.2%  ████████████████████
Epoch  5: 52.3%  ██████████████████████████
Epoch 10: 48.1%  ████████████████████████
Epoch 15: 43.2%  ███████████████████
Epoch 20: 40.7%  ██████████████████          ← Collapsed to near-random!

With ATF (same conditions):
Epoch  1: 44.8%  ████████████████████
Epoch  5: 61.2%  ████████████████████████████████
Epoch 10: 72.4%  ████████████████████████████████████████
Epoch 15: 77.8%  ██████████████████████████████████████████
Epoch 20: 80.0%  ████████████████████████████████████████████  ✓ Stable!

Difference: 40.65% → 80.03% = +39.38 percentage points!
```

ATF's convergence damper and meta controller detect divergence early and correct course.

---

## 📖 Reproducing Results

All experiments can be reproduced using the GUI or CLI:

```bash
# CIFAR-100 Baseline
python -m atf.cli.run --dataset cifar100 --epochs 75 --baseline

# CIFAR-100 ATF
python -m atf.cli.run --dataset cifar100 --epochs 75 --atf --omega 6.0 --amp 0.08 --patience 5

# BERT MRPC ATF
python -m atf.cli.run --dataset bert_mrpc --epochs 5 --atf --patience 3 --amp 0.01 --hi off --gfc off
```

Or use the GUI presets for one-click reproduction.

---

<p align="center">
  <strong>Results by Damjan Žakelj</strong><br>
  <em>All experiments conducted with seed=42 for reproducibility</em>
</p>

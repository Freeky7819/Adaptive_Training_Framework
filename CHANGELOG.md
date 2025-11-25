# Changelog

All notable changes to the Adaptive Training Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-25

### Added

#### Core Framework
- **ATFOrchestrator**: Main orchestrator class coordinating all modules
- **TrainingConfig**: Comprehensive configuration dataclass with 40+ parameters
- **ConvergenceMonitor**: Real-time convergence detection via β metric

#### 8 Training Modules
- **Convergence Analysis (CA)**: Monitors training stability and triggers early stopping
- **Periodic LR Scheduler (PLR)**: Oscillating learning rate with universal frequency ω≈6.0
- **Gradient Feedback Controller (GF)**: Adaptive loss modulation based on gradient dynamics
- **Convergence Damper (CD)**: Stabilizes training near convergence points
- **Temporal Feedback Buffer (TB)**: Historical gradient analysis for trend detection
- **Harmonic Weight Initialization (HI)**: Resonance-based weight perturbation
- **Meta Controller (MC)**: High-level training decisions (LR reduction, early stop)
- **Curvature Regularizer (CR)**: Second-order smoothing for loss landscape

#### Dataset Support
- **Vision**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN, STL-10
- **NLP (BERT)**: SST-2, MRPC, CoLA, MNLI, QQP, QNLI, RTE (GLUE benchmark)
- **NLP (GPT)**: WikiText-2, WikiText-103, Shakespeare

#### GUI Dashboard
- Real-time training visualization via WebSocket
- 6 interactive charts (loss, accuracy, LR, β, curvature, val loss)
- Live ATF Architecture diagram with animated module status
- Quick Config system with CLI parsing
- 15+ presets for common training scenarios
- Intra-epoch validation for long epochs (BERT)
- JSON export of training results

#### Documentation
- Comprehensive README with results and usage examples
- GUI-specific documentation
- CLI reference with 50+ parameter aliases
- Theoretical background on resonance-based training

### Benchmark Results

| Dataset | Baseline | ATF | Improvement |
|---------|----------|-----|-------------|
| CIFAR-100 | 68.58% | 69.33% | +0.75% acc, -59% time |
| CIFAR-10 | 90.54% | 90.45% | ≈ acc, -36% time |
| MNIST | 99.44% | 99.33% | ≈ |
| BERT MRPC | 84.80% | 87.25% | +2.45% |
| BERT SST-2 | 93.23% | 92.78% | -0.45% (baseline better) |

### Key Discoveries
- Universal frequency ω ≈ 6.0 works across domains
- ATF excels on complex tasks (CIFAR-100) and small datasets (MRPC)
- Pre-trained models on large datasets prefer baseline training

---

## [0.9.0] - 2025-11-20 (Pre-release)

### Added
- Initial module implementations
- Basic CLI interface
- Vision dataset support

### Changed
- Refactored parameter naming (gf_alpha → gfc_alpha, etc.)
- Unified API across all modules

---

## [0.8.0] - 2025-11-15 (Development)

### Added
- Core orchestrator design
- Convergence analysis algorithm
- Periodic LR scheduler with phase offset

### Experimental
- Testing various ω values (4.0 - 8.0)
- Discovered optimal ω ≈ 6.0

---

## Future Roadmap

### Planned for v1.1.0
- [ ] ImageNet validation
- [ ] Ablation studies (α, ω, φ sweeps)
- [ ] Additional GLUE tasks
- [ ] Theory paper

### Planned for v2.0.0
- [ ] AI Coach integration (LLM-powered training controller)
- [ ] Distributed training support
- [ ] AutoML integration
- [ ] Model zoo with pre-tuned configs

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

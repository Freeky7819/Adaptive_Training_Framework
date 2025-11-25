# Contributing to ATF

Thank you for your interest in contributing to the Adaptive Training Framework!

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Freeky7819/Adaptive_Training_Framework.git
cd Adaptive_Training_Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

---

## 📋 Types of Contributions

### 🐛 Bug Reports

Found a bug? Please open an issue with:

1. **Description**: What happened?
2. **Expected behavior**: What should have happened?
3. **Steps to reproduce**: How can we replicate this?
4. **Environment**: Python version, PyTorch version, OS, GPU
5. **Logs**: Any error messages or stack traces

### ✨ Feature Requests

Have an idea? Open an issue with:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How would it work?
3. **Alternatives**: Other approaches you considered

### 🔧 Code Contributions

#### Small Changes (typos, docs)

1. Fork the repository
2. Make your changes
3. Submit a pull request

#### Larger Changes (new features, bug fixes)

1. Open an issue first to discuss
2. Fork the repository
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Add tests if applicable
6. Submit a pull request

---

## 📁 Project Structure

```
atf/
├── core/           # Core orchestrator and config
├── modules/        # ATF modules (CA, PLR, GF, etc.)
├── schedulers/     # Learning rate schedulers
├── data/           # Dataset loaders
├── models/         # Model architectures
└── utils/          # Utilities

gui/                # Dashboard interface
tests/              # Test suite
examples/           # Example scripts
```

---

## 🎯 Coding Guidelines

### Style

- Follow PEP 8
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused and small

### Example

```python
def calculate_beta(
    gradient_norm: float,
    history: list[float],
    alpha: float = 0.7
) -> float:
    """
    Calculate convergence metric β using EMA.
    
    Args:
        gradient_norm: Current gradient L2 norm
        history: Previous gradient norms
        alpha: EMA smoothing factor (0-1)
    
    Returns:
        β value between 0 (converged) and 1 (active learning)
    """
    if not history:
        return 1.0
    
    ema = alpha * gradient_norm + (1 - alpha) * history[-1]
    max_ema = max(history) if history else ema
    
    return ema / max(max_ema, 1e-8)
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `ConvergenceAnalysis`)
- Functions: `snake_case` (e.g., `calculate_beta`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_OMEGA`)
- Private methods: `_leading_underscore`

---

## 🧪 Testing

### Running Tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_basic.py

# With coverage
python -m pytest tests/ --cov=atf
```

### Writing Tests

```python
import pytest
from atf import ATFOrchestrator, TrainingConfig

def test_orchestrator_initialization():
    """Test that orchestrator initializes correctly."""
    config = TrainingConfig(lr_omega=6.0)
    orch = ATFOrchestrator(None, config)
    
    assert orch.config.lr_omega == 6.0
    assert orch.state.epoch == 0

def test_periodic_lr_oscillation():
    """Test that LR oscillates correctly."""
    scheduler = PeriodicLRScheduler(omega=6.0, amplitude=0.1)
    
    lr1 = scheduler.get_lr(step=0)
    lr2 = scheduler.get_lr(step=10)
    
    # LR should differ due to oscillation
    assert lr1 != lr2
```

---

## 📝 Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Keep CHANGELOG.md updated

---

## 🔄 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run tests** locally before submitting
4. **Write clear commit messages**
5. **Reference related issues** in PR description

### Commit Message Format

```
type: Short description (max 50 chars)

Longer description if needed. Explain what and why,
not how (the code shows how).

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## 🏷️ Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 💬 Questions?

- Open an issue for questions
- Tag with `question` label

---

<p align="center">
  <strong>Thank you for contributing! 💙</strong>
</p>

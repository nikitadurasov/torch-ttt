<div align="center">
  <img src="https://raw.githubusercontent.com/nikitadurasov/torch-ttt/refs/heads/main/docs/source/_static/images/torch-ttt.png" alt="TorchTTT" width="500">
</div>

<br>

<br>

<div style="gap: 0px; flex-wrap: wrap; align-items: center;">
<p align="center">
    <a href="https://pypi.org/project/torch-ttt/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/torch-ttt.svg">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/stargazers" style="margin: 2px;">
        <img src="https://img.shields.io/github/stars/nikitadurasov/torch-ttt.svg?style=social" alt="GitHub stars" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/network" style="margin: 2px;">
        <img src="https://img.shields.io/github/forks/nikitadurasov/torch-ttt.svg?color=blue" alt="GitHub forks" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/actions/workflows/deploy-docs.yml" style="margin: 2px;">
        <img src="https://github.com/nikitadurasov/torch-ttt/actions/workflows/deploy-docs.yml/badge.svg" alt="Documentation" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/actions/workflows/run-tests.yml" style="margin: 2px;">
        <img src="https://github.com/nikitadurasov/torch-ttt/actions/workflows/run-tests.yml/badge.svg" alt="Testing" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://pepy.tech/project/torch-ttt" style="margin: 2px;">
        <img src="https://pepy.tech/badge/torch-ttt" alt="Downloads" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://pepy.tech/project/torch-ttt">
        <img alt="Monthly Downloads" src="https://static.pepy.tech/badge/torch-ttt/month">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/nikitadurasov/torch-ttt">
    </a>
    <a href="https://github.com/nikitadurasov/masksembles/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/nikitadurasov/masksembles">
    </a>
    </p>

# torch-ttt

**torch-ttt** is a comprehensive PyTorch library for [Test-Time Training (TTT)](https://arxiv.org/abs/1909.13231) and Test-Time Adaptation techniques. It helps make your neural networks more robust and generalizable to distribution shifts, corruptions, and out-of-distribution data—without requiring access to training data or labels at test time.

The library is designed to be **modular**, **easy to integrate** into existing PyTorch pipelines, and **collaborative**—we aim to include as many TTT methods as possible. If you've developed a TTT method, reach out to add yours!

<p align="center">
    >> You can find our webpage and documentation here:</strong> 
    <a href="https://torch-ttt.github.io">torch-ttt.github.io</a>
</p>

> **torch-ttt** is under active development. The API may change as we add new features and methods. Contributions are highly welcome! If you encounter any bugs or have feature requests, please [submit an issue](https://github.com/nikitadurasov/torch-ttt/issues).

## What is Test-Time Training?

Test-Time Training (TTT) is a paradigm where models adapt to test data during inference by optimizing self-supervised auxiliary objectives—without accessing training data or test labels. This helps models handle distribution shifts, corruptions, and out-of-distribution data.

<div align="center">
  <img src="https://raw.githubusercontent.com/nikitadurasov/torch-ttt/refs/heads/main/docs/_static/images/teaser_ssl_code.svg" alt="Test-Time Training Schema" width="700">
</div>

**torch-ttt** implements TTT methods through a unified **Engine** abstraction. Each Engine encapsulates the complete adaptation logic of a specific TTT method, allowing you to:
- Wrap any PyTorch model with a single line of code
- Switch between different TTT methods seamlessly
- Adapt models at inference time without modifying your existing pipeline

This modular design makes it easy to experiment with different adaptation strategies and find the best approach for your specific use case.

## Key Features

**torch-ttt** provides a streamlined API through **Engines**—lightweight wrappers around your PyTorch models. All Engines follow the same interface, making them easy to use and highly modular. 

<div align="center">
  <img src="https://raw.githubusercontent.com/nikitadurasov/torch-ttt/refs/heads/main/docs/_static/images/teaser.svg" alt="Test-Time Training Schema" width="630">
</div>

You can add test-time adaptation with just a few lines of code, and switch between methods seamlessly. The library includes comprehensive tutorials and examples for every method, with efficient implementations suitable for both research and production deployment.

Check out the [Quick Start](https://torch-ttt.github.io/quickstart.html) guide or the [API reference](https://torch-ttt.github.io/api.html) for more details.

## Supported Methods

**torch-ttt** includes implementations of the following test-time training and adaptation methods:

| Method | Class | Paper | Description |
|--------|-------|-------|-------------|
| **TTT** | `TTTEngine` | [Sun et al. 2020](https://arxiv.org/abs/1909.13231) | Original test-time training with self-supervised rotation prediction |
| **TTT++** | `TTTPPEngine` | [Liu et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html) | Improved TTT with contrastive learning |
| **Masked TTT** | `MaskedTTTEngine` | [Gandelsman et al. 2022](https://arxiv.org/abs/2209.07522) | Self-supervised masked reconstruction for adaptation |
| **TENT** | `TentEngine` | [Wang et al. 2021](https://arxiv.org/abs/2006.10726) | Entropy minimization for test-time adaptation |
| **EATA** | `EataEngine` | [Niu et al. 2022](https://arxiv.org/abs/2204.02610) | Efficient anti-catastrophic adaptation |
| **MEMO** | `MemoEngine` | [Zhang et al. 2022](https://arxiv.org/abs/2110.09506) | Marginal entropy minimization with one test point |
| **ActMAD** | `ActMADEngine` | [Mirza et al. 2022](https://arxiv.org/abs/2211.12870) | Activation matching for domain adaptation |
| **DeYO** | `DeYOEngine` | [Mummadi et al. 2021](https://arxiv.org/abs/2104.14504) | Test-time training with deep Y-shaped networks |
| **IT3** | `IT3Engine` | [Eastwood et al. 2024](https://arxiv.org/abs/2403.12682) | Iterative test-time training |

> Want to see your method here? We welcome contributions!

## Installation

**Requirements:** Python 3.10+, PyTorch 1.12+

Install from PyPI:
```bash
pip install torch-ttt
```

Install from source:
```bash
pip install git+https://github.com/nikitadurasov/torch-ttt.git
```


## Quick Start

Here's a minimal example showing how to use **torch-ttt** to adapt a model at test time:

```python
import torch
import torchvision.models as models
from torch_ttt.engine.tent_engine import TentEngine

# Load your pre-trained model
model = models.resnet50(pretrained=True)

# Wrap it with a TTT Engine (e.g., TENT for entropy minimization)
engine = TentEngine(
    model=model,
    optimization_parameters={
        "lr": 2e-3,
        "num_steps": 1
    }
)

# Switch to eval mode
engine.eval()

# At test time, adapt to new data
test_images = torch.randn(8, 3, 224, 224)

# The engine automatically adapts the model during forward
adapted_output = engine(test_images)
```

## Documentation

Comprehensive documentation is available at [torch-ttt.github.io](https://torch-ttt.github.io), including:

- [API Reference](https://torch-ttt.github.io/api.html) - Detailed documentation of all engines and utilities
- [Tutorials](https://torch-ttt.github.io/auto_examples/index.html) - Hands-on examples (MNIST, ImageNet, BERT, etc.)
- [Paper References](https://torch-ttt.github.io/papers.html) - Links to original papers

## Contributing

We welcome contributions! To add a new TTT method, report bugs, or improve documentation:

1. Fork the repository
2. Create a new engine inheriting from `BaseEngine`
3. Add tests and documentation
4. Submit a pull request

See [GitHub Issues](https://github.com/nikitadurasov/torch-ttt/issues) for bug reports and feature requests.

## Citation

If you use **torch-ttt** in your research, please cite:

```bibtex
@software{durasov2024torchttt,
  author    = {Durasov, Nikita},
  title     = {torch-ttt: A Unified PyTorch Library for Test-Time Training},
  year      = {2024},
  doi       = {10.5281/zenodo.17620711},
  url       = {https://github.com/nikitadurasov/torch-ttt},
}
```

Also cite the original papers of the methods you use. See our [Papers page](https://torch-ttt.github.io/papers.html).

## License

MIT License - see the [LICENSE](LICENSE) file for details.


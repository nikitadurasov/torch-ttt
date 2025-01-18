<div align="center">
  <img src="docs/source/_static/images/torch-ttt.png" alt="TorchTTT" width="500">
</div>

<div style="display: flex; gap: 0px; flex-wrap: wrap; align-items: center;">
    <a href="https://github.com/nikitadurasov/torch-ttt/stargazers" style="margin: 2px;">
        <img src="https://img.shields.io/github/stars/nikitadurasov/torch-ttt.svg?style=social" alt="GitHub stars" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/network" style="margin: 2px;">
        <img src="https://img.shields.io/github/forks/nikitadurasov/torch-ttt.svg?color=blue" alt="GitHub forks" style="display: inline-block; margin: 0;">
    </a>
    <a href="https://github.com/nikitadurasov/torch-ttt/actions/workflows/deploy-docs.yml" style="margin: 2px;">
        <img src="https://github.com/nikitadurasov/torch-ttt/actions/workflows/deploy-docs.yml/badge.svg" alt="Documentation" style="display: inline-block; margin: 0;">
    </a>
</div>

# torch-ttt

**torch-ttt** is a package designed to work with [Test-Time Training (TTT)](https://arxiv.org/abs/1909.13231) techniques and make your networks more generalizable. It aims to be modular, easy to integrate into existing pipelines, and collaborative— including as many methods as possible. Reach out to add yours!

<p align="center">
    >> You can find our webpage and documentation here:</strong> 
    <a href="https://torch-ttt.github.io">torch-ttt.github.io</a>
</p>

> **torch-ttt** is in its early stages, so changes are expected. Contributions are welcome—feel free to get involved! If you run into any bugs or issues, don’t hesitate to submit an issue.

---

This package provides a streamlined API for a variety of TTT methods through *Engines*, which are lightweight wrappers around your model. These Engines are:

- **Easy to use** – The internal logic of TTT methods is fully encapsulated, so you only need to wrap your model with an Engine, and you're ready to go.  
- **Highly modular and standardized** – Each Engine follows the same interface, allowing methods to be used interchangeably, making it easy to find the best fit for your application.  
- **Minimal changes required** – Enabling a TTT method for your model requires only a few additional lines of code.  

Check out the [Quick Start]() guide or the [API reference]() for a more detailed explanation of how Engines work and their core concepts.

# Installation

**torch-ttt** requires Python 3.10 or greater. Install the desired PyTorch version in your environment. 

For the latest development version you can run,

```console
pip install git+https://github.com/nikitadurasov/torch-ttt.git
```

While we do not support PyPI yet, support is expected very soon!

# Quickstart

We provide a **Quick Start** guide to help you get started smoothly. Take a look here: [Quick Start]() and see how to integrate TTT methods into your project.

<!-- # Implemented TTTs

## Baselines


| TTT-Method                                     | Image | Text | Graph | Audio |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| [TTT](https://arxiv.org/abs/1909.13231)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [MaskedTTT](https://arxiv.org/abs/2209.07522)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [TTT++](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [ActMAD](https://arxiv.org/abs/2211.12870)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [SHOT](https://arxiv.org/abs/2002.08546)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [TENT](https://arxiv.org/abs/2006.10726)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           | -->

# Tutorials

We offer a variety of tutorials to help users gain a deeper understanding of the implemented methods and see how they can be applied to different tasks. Visit the [Tutorials](https://torch-ttt.github.io) page to explore them.

# Documentation

Our aim is to provide comprehensive documentation for all included TTT methods, covering their theoretical foundations, practical benefits, and efficient integration into your project. We also offer tutorials that illustrate their applications and guide you through their effective usage.

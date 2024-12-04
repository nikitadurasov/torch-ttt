<div align="center">
  <img src="docs/source/_static/images/torch-ttt.png" alt="TorchTTT" width="500">
</div>


# torch-ttt

**torch-ttt** is a package designed to work with [Test-Time Training (TTT)](https://arxiv.org/abs/1909.13231) techniques and make your networks more generalizable. It aims at being modular and as easy to integrate into existing pipelines as possible, also it aims at being collaborative and including as many methods as possible, so reach out to add yours!

# Installation


**torch-ttt** requires Python 3.10 or greater. Install the desired PyTorch version in your environment. 

For the latest development version you can run,

```console
pip install git+https://github.com/nikitadurasov/torch-ttt.git
```

# Implemented TTTs

## Baselines


| TTT-Method                                     | Image | Text | Graph | Audio |
|-----------------------------------------------|:----------:|:--------------:|:------------:|:---------------------:|
| [TTT](https://arxiv.org/abs/1909.13231)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [MaskedTTT](https://arxiv.org/abs/2209.07522)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [TTT++](https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [ActMAD](https://arxiv.org/abs/2211.12870)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [SHOT](https://arxiv.org/abs/2002.08546)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |
| [TENT](https://arxiv.org/abs/2006.10726)                      |     ⏳     |       ⏳       |      ⏳      |          ⏳           |


# Documentation

Our goal is to offer comprehensive documentation for all included TTT methods, covering their theoretical background, along with tutorials that demonstrate these methods using popular datasets.

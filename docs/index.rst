.. torch-ttt master file, created by
   sphinx-quickstart on Mon Dec  2 00:44:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to |torch-ttt|'s documentation!
=================================================

**Deployment & Documentation & Stats & License**

.. raw:: html

   <div style="display: flex; gap: 0px; flex-wrap: wrap; align-items: center;">
       <a href="https://pypi.org/project/pyod/" style="margin: 2px;">
           <img src="https://img.shields.io/pypi/v/pyod.svg?color=brightgreen" alt="PyPI version" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://anaconda.org/conda-forge/pyod" style="margin: 2px;">
           <img src="https://anaconda.org/conda-forge/pyod/badges/version.svg" alt="Anaconda version" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://pyod.readthedocs.io/en/latest/?badge=latest" style="margin: 2px;">
           <img src="https://readthedocs.org/projects/pyod/badge/?version=latest" alt="Documentation status" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://github.com/yzhao062/pyod/stargazers" style="margin: 2px;">
           <img src="https://img.shields.io/github/stars/yzhao062/pyod.svg" alt="GitHub stars" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://github.com/yzhao062/pyod/network" style="margin: 2px;">
           <img src="https://img.shields.io/github/forks/yzhao062/pyod.svg?color=blue" alt="GitHub forks" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://pepy.tech/project/pyod" style="margin: 2px;">
           <img src="https://pepy.tech/badge/pyod" alt="Downloads" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://github.com/yzhao062/pyod/actions/workflows/testing.yml" style="margin: 2px;">
           <img src="https://github.com/yzhao062/pyod/actions/workflows/testing.yml/badge.svg" alt="Testing" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://coveralls.io/github/yzhao062/pyod" style="margin: 2px;">
           <img src="https://coveralls.io/repos/github/yzhao062/pyod/badge.svg" alt="Coverage Status" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://codeclimate.com/github/yzhao062/Pyod/maintainability" style="margin: 2px;">
           <img src="https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability" alt="Maintainability" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://github.com/yzhao062/pyod/blob/master/LICENSE" style="margin: 2px;">
           <img src="https://img.shields.io/github/license/yzhao062/pyod.svg" alt="License" style="display: inline-block; margin: 0;">
       </a>
       <a href="https://github.com/Minqi824/ADBench" style="margin: 2px;">
           <img src="https://img.shields.io/badge/ADBench-benchmark_results-pink" alt="Benchmark" style="display: inline-block; margin: 0;">
       </a>
   </div>

----

About |torch-ttt|
^^^^^^^^^^^^^^^^^^^^^^^

Welcome to |torch-ttt|, a comprehensive and easy-to-use Python library for applying *test-time training* methods to your PyTorch models. Whether you're tackling small projects or managing large datasets, |torch-ttt| provides a diverse set of algorithms to accommodate different needs. Our intuitive API seamlessly integrates into existing training / inference pipelines, ensuring smooth and efficient deployment without disrupting your workflow.

|torch-ttt| includes implementations of various TTT methods, from classical TTT :cite:`sun19ttt`, to more recent methods like TTT++ :cite:`liu2021ttt++` and ActMAD :cite:`mirza2023actmad`. The full list of the implemented methods can be found in a table below.

+-----------+--------------------------+------+------------------------+
| Method    | Engine Class             | Year | Reference              |
+===========+==========================+======+========================+
| TTT       | :class:`TTTEngine`       | 2018 | :cite:`sun19ttt`       |
+-----------+--------------------------+------+------------------------+
| TTT++     | :class:`TTTPPEngine`     | 2021 | :cite:`liu2021ttt++`   |
+-----------+--------------------------+------+------------------------+
| ActMAD    | :class:`ActMADEngine`    | 2023 | :cite:`mirza2023actmad`|
+-----------+--------------------------+------+------------------------+

.. +-----------+-----------------------------------------------------+------+------------------------+
.. | Method    | Engine Class                                        | Year | Reference              |
.. +===========+=====================================================+======+========================+
.. | TTT       | :class:`torch_ttt.engine.ttt_engine.TTTEngine`      | 2018 | :cite:`sun19ttt`       |
.. +-----------+-----------------------------------------------------+------+------------------------+
.. | TTT++     | :class:`torch_ttt.engine.ttt_pp_engine.TTTPPEngine` | 2021 | :cite:`liu2021ttt++`   |
.. +-----------+-----------------------------------------------------+------+------------------------+
.. | ActMAD    | :class:`torch_ttt.engine.actmad_engine.ActMADEngine`| 2023 | :cite:`mirza2023actmad`|
.. +-----------+-----------------------------------------------------+------+------------------------+


|torch-ttt| stands out for:

* **Seamless Integration**: A user-friendly API designed to effortlessly work with existing PyTorch models and pipelines.

* **Versatile Algorithms**: A comprehensive collection of test-time training methods tailored to a wide range of use cases.

* **Efficiency & Scalability**: Optimized implementations for robust performance on diverse hardware setups and datasets.

* **Flexibility in Application**: Supporting various domains and use cases, ensuring adaptability to different requirements with minimal effort.

**Minimal Test-Time Training Example**:

.. code-block:: python

    # Example: Using TTT 
    from torch_ttt.engine.ttt_engine import TTTEngine

    network = Net()
    engine = TTTEngine(network, "layer_name")  # Wrap your model with the Engine
    optimizer = optim.Adam(engine.parameters(), lr=learning_rate)

    engine.train()
    ...  # Train your model as usual 

    engine.eval()
    ...  # Test your model as usual

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started:

   installation
   quickstart
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation:

   api

.. raw:: html

   <br><br>

Indices and tables
=======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: References

.. bibliography::
   :cited:
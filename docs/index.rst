.. torch-ttt master file, created by
   sphinx-quickstart on Mon Dec  2 00:44:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to |torch-ttt|!
=================================================

**Deployment & Documentation & Stats & License**

.. raw:: html

   <div style="gap: 0px; flex-wrap: wrap; align-items: center;">
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
----

About |torch-ttt|
^^^^^^^^^^^^^^^^^^^^^^^

Welcome to |torch-ttt|, a comprehensive and easy-to-use Python library for applying *test-time training* methods to your PyTorch models. Whether you're tackling small projects or managing large datasets, |torch-ttt| provides a diverse set of algorithms to accommodate different needs. Our intuitive API seamlessly integrates into existing training / inference pipelines, ensuring smooth and efficient deployment without disrupting your workflow.

|torch-ttt| includes implementations of various TTT methods, from classical TTT :cite:`sun19ttt`, to more recent methods like DeYO :cite:`lee2024entropy` and ActMAD :cite:`mirza2023actmad`. The full list of the implemented methods can be found in a table below.

+-----------+-----------------------------------------------+------+---------------------------+
| Method    | Engine Class                                  | Year | Reference                 |
+===========+===============================================+======+===========================+
| TTT       | :class:`TTTEngine`                            | 2018 | :cite:`sun19ttt`          |
+-----------+-----------------------------------------------+------+---------------------------+
| TTT++     | :class:`TTTPPEngine`                          | 2021 | :cite:`liu2021ttt++`      |
+-----------+-----------------------------------------------+------+---------------------------+
| TENT      | :class:`TentEngine`                           | 2021 | :cite:`wang2021tent`      |
+-----------+-----------------------------------------------+------+---------------------------+
| EATA      | :class:`EataEngine`                           | 2022 | :cite:`niu2022efficient`  |
+-----------+-----------------------------------------------+------+---------------------------+
| MEMO      | :class:`MemoEngine`                           | 2022 | :cite:`zhang2022memo`     |
+-----------+-----------------------------------------------+------+---------------------------+
| Masked TTT| :class:`MaskedTTTEngine`                      | 2022 | :cite:`gandelsman2022test`|
+-----------+-----------------------------------------------+------+---------------------------+
| ActMAD    | :class:`ActMADEngine`                         | 2023 | :cite:`mirza2023actmad`   |
+-----------+-----------------------------------------------+------+---------------------------+
| DeYO      | :class:`DeYOEngine`                           | 2024 | :cite:`lee2024entropy`    |
+-----------+-----------------------------------------------+------+---------------------------+

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
   papers

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation:

   api

.. .. raw:: html

..    <br><br>

.. Indices and tables
.. =======================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. rubric:: References

.. bibliography::
   :cited:
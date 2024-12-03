Installation
============

.. role:: bash(code)
    :language: bash


You can install the package either from PyPI or from source. Choose the latter if you
want to access the files included the `experiments <https://github.com/ENSTA-U2IS-AI/torch-uncertainty/tree/main/experiments>`_
folder or if you want to contribute to the project.

From source
-----------

To install the project from source, you can use pip directly.

Again, with PyTorch already installed, clone the repository with:

.. parsed-literal::

    git clone https://github.com/nikitadurasov/torch-ttt.git
    cd torch-ttt

Create a new conda environment and activate it:

.. parsed-literal::

    conda create -n ttt python=3.10
    conda activate ttt

Install the package using pip in editable mode:

.. parsed-literal::

    pip install -e .

.. _installation:

============
Installation
============

Last updated: 2025-11-8

This guide will walk you through setting up dFactory and installing
its dependencies.

Prerequisites
=============

- Git
- Python 3.10 or higher
- CUDA 12.4 or higher

Clone the Repository
====================

First, clone the project repository from GitHub. The ``--recursive`` flag
is important as it ensures that all necessary submodules are
downloaded as well.

.. code-block:: bash

   git clone --recursive https://code.alipay.com/xbox/nexus_veomni.git
   cd nexus_veomni

Environment Setup and Dependencies
==================================

We offer two methods for installation. We recommend using **uv** for its
speed, but you can also use **pip** with a standard virtual environment.

Option A: Using uv (Recommended)
--------------------------------

uv is an extremely fast Python package installer and resolver.

Install uv
~~~~~~~~~~

If you don't have uv installed, run:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

.. note::
   
   You may need to restart your terminal or source your shell
   profile (``source ~/.bashrc``, ``source ~/.zshrc``, etc.) for the uv
   command to become available.

Create Environment and Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``uv sync`` command will create a virtual environment in the ``.venv``
directory and install all dependencies specified in the ``pyproject.toml``
file. The ``--extra gpu`` flag includes packages required for GPU support
(e.g., PyTorch with CUDA).

.. code-block:: bash

   # From VeOmni's root directory (dFactory/VeOmni)
   uv sync --extra gpu

Activate Environment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   source .venv/bin/activate

Your command prompt should now be prefixed with ``(.venv)``, indicating
that the environment is active.

Option B: Using pip and venv
----------------------------

This is the classic approach using Python's built-in tools.

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # From the project root directory (dFactory)
   python -m venv .venv

Activate Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   source .venv/bin/activate

Install Project in Editable Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will install the VeOmni package. The ``-e`` flag (editable mode)
allows you to make changes to the source code without needing to
reinstall. The ``[gpu]`` part installs the extra dependencies for GPU
support.

.. code-block:: bash

   pip install -e "VeOmni[gpu]"

Next Steps
==========

You are now ready to use dFactory.

============
Installation
============

dfactory requires Python 3.11 through 3.13 on x86-64 Linux servers with NVIDIA
GPUs and CUDA 12.8. macOS, MPS, and CPU training are not supported. VeOmni and
its GPU dependencies are installed as normal package dependencies; the
repository has no git submodules.

CUDA environment
================

.. code-block:: bash

   git clone --branch develop https://github.com/inclusionAI/dFactory.git
   cd dFactory
   uv sync --locked --extra dev
   source .venv/bin/activate

For an editable pip installation:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel packaging ninja
   pip install --index-url https://download.pytorch.org/whl/cu128 \
     torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
   pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
     --no-build-isolation -e .

The two-stage pip flow is required because ``flash-attn`` imports Torch while
building its CUDA extension. ``uv sync`` supplies the same build dependency
from the repository configuration.

Verify the installation with:

.. code-block:: bash

   dfactory --help
   dfactory recipes list

The package intentionally pins VeOmni. Runtime API changes are handled only in
``dfactory.runtime.veomni`` and do not leak into recipes.

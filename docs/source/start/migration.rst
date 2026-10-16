====================================
Migrating from the submodule version
====================================

The standalone dFactory package intentionally breaks with the former repository
layout. The ``develop`` line continues the public ``main`` history and imports
the reviewed standalone implementation as a clean snapshot; internal commit
history is not published.

Environment changes
===================

The ``VeOmni/`` git submodule is gone. A recursive clone and manual
``PYTHONPATH`` are no longer required. Create an environment at the repository
root and install the project normally:

.. code-block:: bash

   uv sync --locked
   source .venv/bin/activate

VeOmni is pinned as a package dependency. dFactory recipes and extensions must
not import it directly; runtime integration is owned by
``dfactory.runtime.veomni``.

Command changes
===============

Replace recipe-specific Python entry points with the unified command:

.. code-block:: text

   python tasks/train_llada2_bd.py ...
       -> torchrun ... -m dfactory train --config CONFIG

   python tasks/train.py --recipe sft CONFIG
       -> torchrun ... -m dfactory train --config CONFIG

The recipe name now lives in YAML under ``recipe.name``. Use
``dfactory config validate --config CONFIG`` before launching with ``torchrun``.

Configuration changes
=====================

Configuration is strict and divided into ``run``, ``recipe``, ``model``,
``data``, ``distributed``, ``optimization``, ``checkpoint``, and ``logging``.
Old flat arguments and VeOmni argument dataclasses are not accepted as aliases;
migrate them explicitly so that the resolved run is reviewable.

Checkpoint compatibility
========================

The standalone package resumes checkpoints written by its own checkpoint
layout. It does not claim to resume optimizer or scheduler state from the old
training scripts. Import old model weights through the model conversion tools,
then start a new dFactory run.

Extension changes
=================

Custom code should be packaged with ``dfactory.recipes``,
``dfactory.datasets``, or ``dfactory.models`` entry points. See
:doc:`../extend/plugins` for installable examples. The old pattern of adding
modules below ``tasks/`` or changing a central registry is no longer supported.

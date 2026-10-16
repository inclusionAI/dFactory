============
Architecture
============

dFactory owns training semantics and uses VeOmni only as an installed runtime
dependency. The dependency flow is one-way:

.. code-block:: text

   CLI / Python API
           |
           v
   config + recipe/data/model registries
           |
           v
   recipe spec + data + components + model provider
           |
           v
   BaseTrainer + algorithm trainer
           |
           v
   runtime/veomni gateway -> veomni==0.1.4

``dfactory.trainers.BaseTrainer`` owns the complete training lifecycle. Each
recipe directory is self-contained: its concrete ``trainer.py`` owns the
algorithm step alongside that recipe's configuration, validation, data
construction, and declarative ``RecipeSpec`` metadata.
``components.block_diffusion`` contains only reusable layout, attention, noise,
curriculum, and objective primitives; it is not a diffusion framework.

Only ``dfactory.runtime.veomni`` may import VeOmni. This keeps public extension
contracts independent of one distributed-runtime release and gives dFactory a
single place to adapt runtime API changes.

The stable extension surface is exported from ``dfactory``. Concrete runtime
modules remain implementation details.

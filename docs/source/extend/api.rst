=================
Public Python API
=================

Applications start runs through :class:`dfactory.RunConfig` and
:func:`dfactory.train`:

.. code-block:: python

   from dfactory import RunConfig, train

   config = RunConfig.from_yaml("config.yaml")
   train(config)

Extension packages use these root exports:

* ``RecipeSpec`` and ``RecipeData`` define recipe discovery and data
  construction. ``BaseTrainer``, ``StepContext``, ``StepMetrics``, and
  ``TrainerState`` define the training lifecycle and the operations available
  during one micro-batch.
* ``DatasetSpec``, ``TrainBatch``, and ``AttentionSpec`` define a named dataset
  source and the stable batch contract. ``TrainBatch`` carries ``input_ids``,
  ``labels``, ``loss_mask``, ``position_ids``, and tensor-only attention
  metadata; recipes build device-side attention objects from that metadata.
  Algorithm-specific transforms and collation remain owned by the selected
  recipe.
* ``ModelSpec`` and ``ModelProvider`` define model construction, tokenizer and
  special-token resolution, parallel policy, and checkpoint export.
* ``get_recipe``, ``list_recipes``, ``get_dataset``, ``list_datasets``,
  ``get_model``, and ``list_models`` query built-in and installed providers
  through the same registry.

VeOmni types and operations are excluded from the public contract. Trainer
extensions use the dFactory types exported from the package root and leave
distributed-runtime access to ``BaseTrainer``.

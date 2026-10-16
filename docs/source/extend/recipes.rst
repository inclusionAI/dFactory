=================
Recipe Extensions
=================

A recipe describes algorithm-specific configuration, validation, data
preparation, and the trainer class that executes the algorithm. The shared
:class:`dfactory.BaseTrainer` owns runtime initialization, model and dataloader
construction, gradient accumulation, optimization, checkpointing, metrics, and
model export.

Package an external recipe as a normal Python distribution and publish one
entry point:

.. code-block:: toml

   [project.entry-points."dfactory.recipes"]
   my_recipe = "my_package.recipe:recipe_spec"

The target is a :class:`dfactory.RecipeSpec` with these fields:

``config_type``
   A recipe-owned configuration type providing ``from_mapping(values)``.

``validate_run``
   Validate interactions between recipe parameters and the complete run
   configuration.

``data_factory``
   Build and return :class:`dfactory.RecipeData`. The first release requires a
   map-style dataset; a recipe may also provide a ``collate_fn``. Packing is a
   recipe decision because its layout constraints belong to the algorithm.

``trainer_type``
   A :class:`dfactory.BaseTrainer` subclass. It implements
   ``training_step(batch, context)`` and inherits the common lifecycle.

The entry point value is metadata, not the trainer class itself:

.. code-block:: python

   from dfactory import BaseTrainer, RecipeSpec, StepContext, StepMetrics

   class MyTrainer(BaseTrainer):
       def training_step(self, batch, context: StepContext) -> StepMetrics:
           outputs = context.forward({"input_ids": batch["input_ids"]})
           loss = compute_loss(outputs, batch)
           context.backward(loss)
           return StepMetrics({"loss": float(loss.detach().item())})

   recipe_spec = RecipeSpec(
       name="my_recipe",
       config_type=MyRecipeConfig,
       validate_run=validate_run,
       data_factory=build_data,
       trainer_type=MyTrainer,
       description="My training algorithm.",
   )

The recipe callbacks and trainer constructor are invoked with keyword
arguments:

.. code-block:: text

   validate_run(*, recipe_config, run_config) -> None
   data_factory(*, source, tokenizer, data_config,
                recipe_config, special_tokens) -> RecipeData
   trainer_type(*, run_config, recipe_config, data_factory) -> BaseTrainer

Trainer implementations must not import VeOmni directly. Use
:class:`dfactory.StepContext` inside ``training_step`` for forward, backward,
and reduced-loss operations. Keep the trainer in the recipe package alongside
its config and data modules. Add another recipe trainer only when it has a real
algorithm and runtime contract; dFactory does not publish placeholders for
algorithms it does not yet implement.

==============================
Installable extension packages
==============================

dFactory discovers recipes, dataset sources, and model providers from standard
Python entry points. Extensions are ordinary distributions: they can live in a
separate repository, declare a compatible dFactory release, and be installed
with pip without changing the dFactory source tree.

The repository contains three independently installable examples:

* ``examples/plugins/custom_recipe`` registers ``example_causal_lm`` in
  ``dfactory.recipes``;
* ``examples/plugins/custom_dataset`` registers ``json_array`` in
  ``dfactory.datasets``;
* ``examples/plugins/custom_model`` registers ``hf_causal_lm`` in
  ``dfactory.models``.

Install them into the same environment as dFactory and inspect discovery:

.. code-block:: bash

   pip install -e examples/plugins/custom_recipe
   pip install -e examples/plugins/custom_dataset
   pip install -e examples/plugins/custom_model

   dfactory recipes list
   dfactory datasets list
   dfactory models list

Extension code should import contracts such as :class:`dfactory.RecipeSpec`,
:class:`dfactory.DatasetSpec`, and :class:`dfactory.ModelSpec` from the public
``dfactory`` package. Recipe trainers must extend
:class:`dfactory.BaseTrainer` and use :class:`dfactory.StepContext` for
forward, backward, and loss reduction. They must not import VeOmni directly.

Entry-point names are global within their group. dFactory rejects collisions
between built-in and external providers instead of silently replacing one.

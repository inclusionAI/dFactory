===========================
Multi-stage Teacher Forcing
===========================

The ``mtf`` recipe trains a Block Diffusion model through a confidence-based
curriculum. Every trainable token in the noisy slot is masked before the first
forward pass. Each stage then:

* runs one teacher-forcing forward pass;
* selects tokens independently within each logical block;
* includes the stage's base quota plus any additional tokens whose prediction
  confidence exceeds ``confidence_threshold``;
* records the selected predictions for stage metrics; and
* replaces the accumulated selection history with ground-truth tokens before
  the next stage.

The cumulative base quota is ``(stage + 1) / max_stages``. The final stage
therefore covers every remaining trainable token. Stage losses are averaged
inside ``MTFTrainer``, followed by one call to the public ``StepContext``. The
context applies gradient-accumulation scaling exactly once.

Start from the public example:

.. code-block:: bash

   dfactory config validate --config configs/recipes/mtf/llada2_mini.yaml
   torchrun --nproc-per-node=8 -m dfactory train \
     --config configs/recipes/mtf/llada2_mini.yaml \
     --set distributed.dp=8

Ownership and scope
===================

``MTFTrainer`` owns the complete multi-stage algorithm and depends only on the
public data, component, config, recipe, and trainer contracts. The reusable, stateless
confidence/quota selection lives in ``dfactory.components.block_diffusion`` so
other recipes can use the same primitive without importing MTF source.

The first public slice accepts conversation data and supports the shared Block
Diffusion attention and optional sliding-window configuration. Token-record
inputs remain deferred.

The former ``per_stage_step`` option is intentionally not migrated. Stepping
the optimizer from inside an algorithm step violates the ownership boundary:
optimizer, scheduler, gradient clipping, and accumulation remain
``BaseTrainer`` responsibilities. One MTF micro-batch therefore produces one accumulated
backward contribution and one outer optimizer step.

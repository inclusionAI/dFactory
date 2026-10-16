===================================
Editing Multi-stage Teacher Forcing
===================================

The ``emtf`` recipe extends MTF with a self-correcting phase in every stage.
All trainable noisy-slot tokens start masked. A stage then performs two passes:

#. The teacher-forcing pass selects new tokens with the shared per-block
   confidence curriculum and trains only that selection.
#. The self-correcting pass starts again from the original masked input,
   replaces the accumulated selection with the current teacher-forcing
   predictions, and trains over that complete accumulated selection.

After the self-correcting pass, the accumulated positions are replaced with
ground truth for the next teacher-forcing stage. Prediction history, current
selection, cumulative history, teacher-forcing error, and self-correcting error
remain separate concepts in the implementation and metrics.

Loss weighting
==============

Each teacher-forcing and self-correcting loss receives weight
``1 / (2 * max_stages)``. Their weighted sum is the reported total loss.
Each phase calls ``StepContext.backward`` with that weight, so the context—not
the concrete trainer—applies gradient-accumulation scaling. This preserves the original
two-forward/two-backward stage behavior without giving the algorithm ownership of
the optimizer.

Start from the public example:

.. code-block:: bash

   dfactory config validate --config configs/recipes/emtf/llada2_mini.yaml
   torchrun --nproc-per-node=8 -m dfactory train \
     --config configs/recipes/emtf/llada2_mini.yaml \
     --set distributed.dp=8

Ownership and scope
===================

``EMTFTrainer`` owns its algorithm and imports no MTF trainer source. Both trainers use the
stateless curriculum selection, attention, layout, noise, and objective primitives from
``dfactory.components.block_diffusion`` and the public conversation data
pipeline.

The first public slice accepts conversation data only. Token-record inputs are
deferred. The former inherited configuration fields, recipe-local sequence
parallel reduction, automatic Liger-kernel selection, and ``per_stage_step``
optimizer ownership are intentionally excluded. Distributed loss reduction
and gradient accumulation stay behind ``StepContext``; optimizer and scheduler
steps stay in ``BaseTrainer``.

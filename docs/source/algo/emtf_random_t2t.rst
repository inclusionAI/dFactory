===============
EMTF Random T2T
===============

The ``emtf_random_t2t`` recipe keeps teacher forcing at every curriculum stage
but runs self-correction at exactly one uniformly sampled stage per
micro-batch. The source recipe called this variant ``emtf_randomt2t``; dFactory
uses the clearer public name with a separator between ``random`` and ``t2t``.

The selected stage is sampled with ``torch.randint(0, max_stages)`` on the
training device. Prediction, selection, and ground-truth history are updated
at every stage, including stages where self-correction is skipped. Metrics for
skipped SC stages are explicit zeros, and ``sc_stage`` records the selected
stage.

Loss weighting
==============

The migration preserves the source coefficients exactly:

* every TF loss has weight ``1 / (2 * max_stages)``;
* the single SC loss has weight
  ``(max_stages - 1) / (2 * max_stages)``.

The second coefficient is intentionally not normalized to ``1/2``. It
compensates for running self-correction at only one stage. ``StepContext``
applies gradient accumulation after these algorithm weights. Because the
self-correction coefficient is zero with a single stage, this recipe requires
``max_stages >= 2``.

Start from the public example:

.. code-block:: bash

   dfactory config validate \
     --config configs/recipes/emtf_random_t2t/llada2_mini.yaml
   torchrun --nproc-per-node=8 -m dfactory train \
     --config configs/recipes/emtf_random_t2t/llada2_mini.yaml \
     --set distributed.dp=8

Migration scope
===============

``EMTFRandomT2TTrainer`` owns its algorithm and imports no EMTF or MTF trainer source. It
uses the public conversation pipeline and the shared Block Diffusion layout,
noise, attention, objective, and pure curriculum-selection components.
Token-record inputs, inherited legacy arguments, algorithm-local sequence
parallel reduction, Liger auto-selection, and optimizer ownership are not
migrated.

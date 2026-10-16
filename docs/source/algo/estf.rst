ESTF continued pretraining
==========================

``estf`` trains a Block Diffusion checkpoint with a per-block curriculum and
two dependent forwards in every stage. It consumes pre-tokenized documents
using the same document-preserving packing and paired attention layout as
``ar2bd``.

Algorithm
---------

By default every valid noisy-slot token starts masked. The public RPT profile
enables ``initialize_random_known``: each eligible token draws an initial known
probability from ``initial_known_ratio`` after applying
``initial_known_schedule_power``. The physical terminal position of every
logical block is excluded from this random initialization; in a partial block
that position is already padding. Each stage then performs:

#. An m2t (masked-to-token) forward. Positions already selected in earlier
   stages contain ground truth, while the remaining positions contain the mask
   token. On non-final stages, confidence-threshold selection is applied
   independently per logical block. Forced fill advances each block to
   ``ceil(valid_tokens * (stage + 1) / max_stages)`` selected tokens.
#. A t2t (token-to-token) forward. The cumulative selected positions are
   replaced by detached m2t predictions, while unselected positions remain
   masked. On the final stage, every valid position is replaced and trained.

Ignored labels and padding never count as selected, already known, or forced
fill. The mask token is resolved by the model provider and is not part of the
recipe configuration.

Loss weighting
--------------

The default m2t objective covers all currently masked valid positions. Each
block is scaled by the inverse of its remaining-mask ratio, matching the
source algorithm. With ``selective_loss: true``, non-final m2t stages instead
train only newly selected positions. The t2t objective covers the cumulative
selected positions. After masking and optional inverse-noise weighting, each
phase takes the mean over the complete paired tensor; selection density is
therefore part of the v5 objective scale.

Each m2t and t2t phase calls ``StepContext.backward`` with weight
``1 / (2 * max_stages)``. The context applies gradient-accumulation scaling;
the concrete trainer delegates optimizer ownership and raw ``backward`` to the
shared trainer lifecycle.

Start from the public example:

.. code-block:: bash

   dfactory config validate --config configs/recipes/estf/llada2_mini.yaml
   torchrun --nproc-per-node=8 -m dfactory train \
     --config configs/recipes/estf/llada2_mini.yaml \
     --set distributed.dp=8

Public scope
------------

``ESTFTrainer`` owns the m2t/t2t inputs, per-block deterministic curriculum,
selectable ground-truth or maximum-probability confidence, random-known
initialization, selective-loss option, inverse-noise loss normalization, and
two-phase gradient weighting. Ignored labels are never counted as already
unmasked.

Random schedules, Gumbel confidence, conversation datasets, and recipe-owned
optimizer behavior are not part of this public recipe.

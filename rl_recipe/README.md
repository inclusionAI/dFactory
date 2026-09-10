# dLLM RL Recipe Demo

A small, self-contained demo showing the key deployment boundary and algorithm flow
of a dLLM reinforcement-learning recipe.

It covers:

- sending an agentic search task through the lumen-zoo interface;
- receiving a fixed example rollout and simulated SGLang `step_maps`;
- constructing a discrete perturbed sequence from the rollout;
- scoring the original rollout tokens from the perturbed input;
- computing the block-wise `PackedEBPOLoss` objective.

The example runs locally with the Python standard library and uses fixed values to
keep the algorithm easy to inspect.

```bash
python3 -m rl_recipe.demo
```

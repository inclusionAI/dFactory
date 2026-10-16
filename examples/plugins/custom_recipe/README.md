# Custom recipe plugin

This package demonstrates a recipe that is installed independently from
dFactory and discovered through the `dfactory.recipes` entry-point group.

```bash
pip install -e examples/plugins/custom_recipe
dfactory recipes list
```

The example intentionally uses only dFactory's public API. It owns parameter
validation, token-record conversion, and its causal language-model objective.
Its trainer extends dFactory's public `BaseTrainer` and does not import VeOmni.

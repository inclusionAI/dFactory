# dFactory plugin examples

Each child directory is an independent Python distribution and demonstrates
one public extension point:

- `custom_recipe`: a `RecipeSpec` with its own config, data transform, and trainer;
- `custom_dataset`: a `DatasetSpec` for top-level JSON arrays;
- `custom_model`: a `ModelSpec` backed by Hugging Face Auto classes.

Install any example with `pip install -e PATH`. dFactory discovers it from
package metadata; no registry edit, source copy, or VeOmni import is required.

# dFactory documentation

## Build the docs

```bash
# Run from the repository root on the supported Linux CUDA environment.
uv sync --locked --extra docs
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

## Open the docs with your browser

```bash
python -m http.server -d docs/_build/html/
```

Navigate to <http://localhost:8000> to view the documentation.

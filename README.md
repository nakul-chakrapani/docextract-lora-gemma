# docextract-lora

Scaffold for LoRA-based document extraction experiments with CORD/VRDU data, Gemma-style multimodal prompts, training stubs, and evaluation helpers.

## Project Layout

- `src/data`: dataset loading and prompt template helpers.
- `src/training`: typed dataclass configs and training launcher stub.
- `src/inference`: single prediction, batch evaluation, and API serving stubs.
- `src/evaluation`: baseline metric implementations and run comparison helpers.
- `tests`: placeholder pytest tests.
- `configs`: YAML defaults aligned with dataclass configs.
- `notebooks`: starter experiment notebooks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[all]
```

Optional installs:

- `pip install -e .[dev]` for linting, testing, and notebook tooling.
- `pip install -e .[notebook-cpu]` for data exploration on CPU-only machines.
- `pip install -e .[train]` for full training dependencies (includes Linux-only bitsandbytes).
- `pip install -e .[dev,notebook-cpu]` for local notebook + dev workflows.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,notebook-cpu]
pytest -q
```

## Notes

- `src/data/dataset.py` uses Hugging Face Datasets loaders and expects internet access.
- `src/inference/serve.py` exposes `/health` and `/predict` endpoints via FastAPI.
- Replace stub logic in training/inference modules with your full pipeline implementation.

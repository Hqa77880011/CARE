# CARE

Implementation of certificate-authorized selective safety repair under a finite causal-model ambiguity set.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The command below generates finite coordination incidents with categorical policies and risk tables. For your own finite incidents, add `--input incidents.json` to `prepare`; required fields are `id`, `policy`, `risk_models`, `heldout_risk`, `allowed`, `innocent`, `reward` and `coordination`.

```bash
care prepare --config configs/reproduce.json --data data/incidents
```

## Execution

`configs/reproduce.json` fixes seed `17`. Keep this config for every step; change `seed` there and use separate data/output directories for independent runs.

```bash
care run --config configs/reproduce.json --data data/incidents --output runs
```

## Evaluation and comparisons

`compare` executes the available methods, including component controls, on the same prepared inputs.

```bash
care compare --config configs/reproduce.json --data data/incidents --output runs
care plot --output runs
```

## Outputs

`runs/<method>/` contains `episodes.csv`, `certificates.json` and `metrics.json`. Comparisons write `runs/comparison.csv` and `runs/comparison.json`; plots go to `runs/figures/`.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.

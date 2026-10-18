# EventGrad

Implementation of **Event-time gradient reconstruction under delayed selective labels**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation

The generator writes finite event-time delayed-label scenarios as JSONL. The registered configuration fixes the count and seed; event, arrival and selection records remain available for scoring.

```bash
python -m eventgrad prepare --config configs/reproduce.json --output data/reproduce.jsonl
```

## Execution and evaluation

`run` evaluates the main method; `suite` evaluates the methods in `METHODS` on the same scenarios.

```bash
python -m eventgrad run --data data/reproduce.jsonl --output runs/main
python -m eventgrad suite --data data/reproduce.jsonl --output runs/comparison
```

## Outputs

Each `runs/<run>/<method>/` contains `records.jsonl` and `metrics.json`. Plotting writes `figures/summary.csv` and per-metric figures. The statistics command summarizes the main run records; plot inputs are method run directories.

```bash
python -m eventgrad.statistics --records runs/main/eventgrad/records.jsonl --output results/intervals.json
python -m eventgrad plot --inputs runs/comparison/eventgrad --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training and full evaluation were not run for this review. The GitHub Actions workflow checks packaging and source code.

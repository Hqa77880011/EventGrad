# EventGrad

Implementation of **Event-time gradient reconstruction under delayed selective labels**.

Logs event-time logits and a Jacobian in a fixed update basis, deletes learner input records and reconstructs the coordinate gradient only when selected labels arrive. A frozen warm-start outcome model supplies doubly robust correction.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/EventGrad.git
cd EventGrad
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Input contains warm and chronological feature/label arrays, positive selection propensities, selected flags, delays, class count and fixed rank. A recorded event uses q(r+1)+3 scalars; unresolved selected events determine resident storage.

```bash
python -m eventgrad prepare --count 1 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m eventgrad run --data data/workload.jsonl --output runs/main
python -m eventgrad suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. eventgrad, ipw, plugin_only, uncorrected and current_gradient. current_gradient is an input-retaining arrival-time baseline.

## Evaluation and visualization

Coordinate gradient error/cosine, exact reconstruction error, full-gradient capture, record scalars, peak pending bytes, basis bytes, event-time accuracy/NLL and learned model parameters.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m eventgrad plot --inputs runs/comparison/eventgrad --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The fixed orthonormal update basis makes transport unnecessary. Raw inputs and event parameter snapshots are retained only by the experiment evaluator for oracle comparisons, not by the event-record learner. Outcome fitting uses independent fully observed warm-start events and is frozen during the measured stream. Propensity overlap is required.


## Uncertainty and paired comparisons

```bash
python -m eventgrad.statistics --records runs/main/eventgrad/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Registered compact reproduction setting

Twenty independent categorical streams with eight features, three classes and a fixed rank-six parameter subspace. Event-time records reconstruct the exact projected gradient; rotation transport is unnecessary for this fixed-subspace setting.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
python -m eventgrad prepare --config configs/reproduce.json --output data/reproduce.jsonl
python -m eventgrad suite --data data/reproduce.jsonl --output runs/reproduce
```

This setting needs no external dataset or pretrained weights. Generated inputs and declared fault/shift laws define the finite task; reported quantities retain that provenance.

# CARE

Implementation of certificate-authorized selective safety repair under a finite causal-model ambiguity set.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Incident data and workflow

The public interface uses finite incident models, so certificate feasibility and subset minimality can be checked exactly. `care prepare --input incidents.json` imports a JSON list of incidents. Each incident contains categorical `policy` [agent,action], `risk_models` [model,action_1,...,action_N], a held-out risk tensor, executable action lists `allowed`, evaluation-only `innocent` agent IDs and joint-action `reward`/`coordination` tensors. Risk entries are probabilities in [0,1]. Models can be estimated from incident rollouts from [PettingZoo MPE](https://pettingzoo.farama.org/environments/mpe/) or [RWARE](https://github.com/semitable/robotic-warehouse); estimation and environment policies remain upstream of this finite interface. The supplied `prepare` without input generates a labeled finite coordination workload.

```bash
care prepare --data data/incidents
care run --data data/incidents --method care --output runs
care compare --data data/incidents --output runs
care plot --output runs
```

## Repair authorization

Subsets are searched in increasing cardinality, including the empty subset. Each executable deterministic action witness is evaluated against every declared risk model while all unselected policies stay fixed. The first feasible witness carries an exact cardinality lower bound, checked risks, ambiguity-set IDs and full search trace. If no witness satisfies `alpha`, selectivity abstains and uses the joint safety optimizer. The finite witness policy is the authorized repair; no gradient is allowed to change unselected agents. The gradient optimizer is used by joint and attribution controls. Both certificate methods apply their declared finite witness. Deterministic witness policies are exact categorical policies; KL uses a 1e-12 probability floor for a finite numerical report.

`point` certifies against only the nominal model, `joint` updates every policy, and `attribution` takes the same mask size using one-agent counterfactual risk reduction. These are controlled finite-model comparisons, rather than full implementations of external MCTS/SPIBB planners. Gradient step count and learning rate are explicit settings. `calibrate` sweeps step counts on a separate calibration incident directory and writes candidate worst-case ambiguity-set risks; use the selected setting in the test config. Never use held-out test incidents for this step. The comparator reports `within_safety_band` and does not silently assert safety matching.

## Metrics

`configs/default.json` controls finite workload size, ambiguity perturbation, risk threshold, optimizer and safety band. Exact search grows exponentially with agent count and action cardinality. `episodes.csv` reports held-out violation probability, innocent-agent KL, updated-agent fraction, reward and coordination retention, certificate coverage and abstention. `certificates.json` records before/after policies and authorization evidence. Comparison JSON/CSV and PNG/SVG charts are derived from these outputs. The finite ambiguity set, deterministic witness vocabulary, independently parameterized agent policies and table-based risk evaluation are the implementation's technical scope; there is no learned neural causal model hidden behind the certificate.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
care-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.

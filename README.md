# Structural-Test Geometry as an Inductive Bias for Partially Observable World Models

Todd Y. Zhou · NeurReps 2026 Proceedings Track

This repository contains a **new post-acceptance reference implementation**, prepared on 25 October 2026. It implements the central structural-test geometry objectives and a runnable Hidden-Goal GridWorld example. The original six-task experiment code, checkpoints, and raw records are not included. This release does not reproduce or independently verify the results in the paper.

## Paper

- [Camera-ready paper](paper/zhou26.pdf)
- [BibTeX entry](paper/zhou26.bib)

The manuscript has nine main-text pages, followed by references and appendices. The published reference code does not change its reported results.

## What runs

The package includes a Gaussian RSSM with a default 200-dimensional recurrent state and 56-dimensional stochastic state; QP, MD, TR, RNC, structured belief distillation, privileged-state regression, and a prediction baseline; a discrete-action CEM planner; exact Bayesian filtering and posterior-fork structural responses for the reference GridWorld; IQD and paired nested-bootstrap statistics.

The GridWorld's sensor and continuation details are newly specified where the manuscript leaves choices open. The other five environments, visual particle filters/VAE, original tuning runs, and reported results are not reproduced here. Continuous TR uses hard sorting rather than the paper's regularized SoftSort. Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the exact correspondence and differences.

## Install and check

Python 3.12 is the tested runtime. CPU execution is tested; CUDA and MPS execution have not been validated. Install PyTorch for your platform if the standard package installation does not provide it.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

[requirements-tested.txt](requirements-tested.txt) records the package versions used for verification. The Python files contain no comments or docstrings.

## Run a small example

These commands collect fresh data, train a small model for three updates, plan one episode, and compute held-out IQD. They check that the pipeline runs; they do not establish learning quality or reproduce a paper result. The exact response estimator is useful for this small finite environment.

```sh
python -m stg collect --output data/train.npz --histories 64 --seed 0 --exact
python -m stg collect --output data/heldout.npz --histories 64 --seed 100 --split id --exact
python -m stg train --data data/train.npz --output runs/qp-smoke --config configs/smoke.yaml
python -m stg evaluate --checkpoint runs/qp-smoke/checkpoint.pt --output runs/qp-smoke/id.json --episodes 1 --horizon 3 --population 8 --iterations 2 --elites 2
python -m stg diagnose --checkpoint runs/qp-smoke/checkpoint.pt --data data/heldout.npz --output runs/qp-smoke/iqd.json --pairs 2000
```

Caches, checkpoints, evaluation files, and diagnostics record their reference-code provenance. Existing output files are protected against accidental replacement. Use a new output directory for each run.

## Run a larger reference experiment

```sh
python -m stg collect --output data/train-large.npz --histories 10000 --seed 0 --rollouts 100
python -m stg train --data data/train-large.npz --output runs/qp-10 --config configs/reference.yaml --method QP --seed 10
python -m stg evaluate --checkpoint runs/qp-10/checkpoint.pt --output runs/qp-10/id.json --split id
python -m stg evaluate --checkpoint runs/qp-10/checkpoint.pt --output runs/qp-10/shift.json --split shift
```

The larger configuration uses the paper's stated RSSM dimensions, learning rate, auxiliary batch size, QP scale rule, and CEM settings. Its remaining training choices are documented reference choices. A full 200,000-update run has not been executed for this release. Target collection and training can take substantial time; use the small example first.

Change `--method` to `NONE`, `RNC`, `TR`, `MD`, `BD-S`, or `PH`. MD and QP use the same seeded response-cluster sampler and all unordered pairs in each auxiliary batch. The default coefficients are TR 0.5, MD 1, BD-S 0.5, and QP 1; they are copied from the manuscript, not retuned on this new environment. RNC and PH use a reference coefficient of 1. Configuration files allow explicit overrides.

## Compare measured episode records

Use the same model seeds and evaluation seeds for both methods. Evaluation files contain the task, model seed, episode identifier, and binary success for every episode. Multiple files can be supplied to each side to combine seeds and fixed tasks. Matching is enforced; duplicate or unmatched records are rejected.

```sh
python -m stg bootstrap --first runs/qp-10/id.json --second runs/md-10/id.json --output runs/qp-minus-md.json --replicates 10000
```

The interval resamples matched seeds and then paired episodes within each fixed task and seed. Results are success fractions; multiply by 100 for percentage points. A meaningful empirical comparison requires enough independently trained seeds and episodes. No historical table values are embedded in the implementation.

## Layout

| Path | Contents |
| --- | --- |
| `stg/model.py` | RSSM, posterior encoding, observation/reward/value losses |
| `stg/objectives.py` | QP, MD, TR, RNC, BD-S, PH |
| `stg/environments.py`, `stg/targets.py` | Reference GridWorld, exact posterior, structural probes and continuations |
| `stg/planner.py` | Ordinary-action CEM planning |
| `stg/data.py`, `stg/training.py`, `stg/cli.py` | Data collection, training, evaluation, held-out diagnostics |
| `stg/metrics.py` | TV, W1, structural pseudometric, IQD, matched nested bootstrap |
| `tests/` | Mathematical, gradient, posterior, planner, and pipeline checks |

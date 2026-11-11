# Development provenance

This is a source-only refactor of the already executed Model 1 baseline. It changes the delivery workflow, not the model family, split, hyperparameters, label mapping, or evaluation threshold.

## Actual changes

1. **Standalone official-data preparation.** The previous local `make_split.py` depended on a quality-check report and the starter dataset. The new `common.py` checks the official `train.csv` directly and creates the same ordered stratified 8,000/2,000 split. This removes unrelated local-file requirements without changing the held-out examples.
2. **Fresh train-to-predict workflow.** The old handoff offered an already fitted model, while its prediction helper required that original model's binary hash. The course module now always initializes a fresh pipeline in training. Prediction validates the checkpoint produced by the current run, then generates the official two-column CSV. This permits reproducibility without shipping any model weights.
3. **Explicit paths and verification.** CLI arguments choose data and a new run directory, avoiding absolute personal paths and preserving older runs. `verify_run.py` reloads the generated model, checks ordered split IDs and probabilities, and recomputes AUROC using the independent rank-sum formula. Exact cross-version binary identity is not assumed.

## Earlier source history

- [Original baseline source publication](https://github.com/yx749-lab/cornell-tech-cs5785-model1/commit/2e200ced1073bd94fc24696dd6d6a43ca579b6b8)
- [Original recorded validation results](https://github.com/yx749-lab/cornell-tech-cs5785-model1/commit/908333fe67f8d1640b8c807d11c98ff4a050128a)
- Original local `train_baseline.py` SHA-256: `55f0d3d238844205b120d156267a59c00ff2c7b533177c3e99fcc55641469d52`.
- Original Kaggle submission: `58183934`; public score `0.82964`.

These links identify existing publication commits, not a claim that each represents a tuning experiment. This course refactor was prepared locally; no new PR or remote commit is implied by this document. The team should commit the actual integration changes, preserve genuine development history, and cite relevant commits in Q6.2. Do not manufacture or backdate a history.

## Scope

The baseline, its new submission workflow, and local reproduction checks are included. Tuning, training-size curves, unlabeled-data experiments, subgroup/error analysis, ensemble improvements, the full report and the final team PR are not claimed complete by this module.

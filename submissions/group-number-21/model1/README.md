# Model 1 — Word TF-IDF + Logistic Regression

这是可独立运行的 Model 1 课程代码模块。它从官方 CSV 重新划分数据、训练模型、计算验证指标，并生成 Kaggle 概率文件。无需旧模型、旧结果、starter 文件或个人电脑上的其他项目文件。它不会自动上传 Kaggle、GitHub 或创建 PR。

Model 1 contributor: **Yaoan Xie (NetID: yx749; GitHub: yx749-lab)**.
The parent team README must list all team members and the exact Kaggle team name.
This module is the untuned baseline, not the entire group assignment.

## Files

| File | Purpose |
|---|---|
| `common.py` | Frozen parameters, official input checks, shared split and validation helpers |
| `train.py` | Fresh training, held-out evaluation, saved model, metrics and figures |
| `predict.py` | Load the newly trained model and write `submission.csv` |
| `run_all.py` | Run training/evaluation and prediction in sequence |
| `verify_run.py` | Reload the model, recheck split/predictions and independently recompute AUROC |
| `requirements.txt` | Exact dependency versions used in verification |
| `.gitignore` | Exclude local data, environments, checkpoints and generated predictions |
| `DEVELOPMENT.md` | Actual development changes, original source references and AI disclosure |

## 1. Get the official data

Download the competition files from the [official Kaggle Data page](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data), using an authorized competition account. Put these three **unedited** files in one local directory:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

Do not include the data in the course GitHub PR. The code validates SHA-256 hashes against the original official files. Do not open and resave the CSVs in a spreadsheet application or reorder their rows. `train_starter_2000.csv` and `unlabeled.csv` are not needed by this baseline.

## 2. Install

Use **Python 3.12** (verified with 3.12.6 on Apple Silicon macOS). From this module's directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

On Windows, use `py -3.12 -m venv .venv` and replace `.venv/bin/python` below with `.venv\Scripts\python.exe`. Windows and Linux execution have not been tested. The code has no absolute user-specific paths; numerical differences across platforms should be assessed with tolerance rather than requiring an identical model binary.

## 3. Train from scratch and generate predictions

If the three input files are in `data/`, run:

```bash
.venv/bin/python run_all.py --data-dir data --output-dir outputs/baseline
.venv/bin/python verify_run.py --data-dir data --run-dir outputs/baseline
```

**`outputs/baseline` must not already exist.** This protects previous runs. Use a new output directory name for another run. Data and outputs may also be outside this module; quote paths containing spaces.

The separate entrypoints are equivalent:

```bash
.venv/bin/python train.py --data-dir data --output-dir outputs/separate_run
.venv/bin/python predict.py --data-dir data --run-dir outputs/separate_run
.venv/bin/python verify_run.py --data-dir data --run-dir outputs/separate_run
```

Training reads **only `train.csv`**. Prediction reads `test.csv` and the submission template. The model is freshly initialized and fitted; no pretrained `.joblib` is distributed or loaded during training. No external model service is called.

## Outputs

All generated artifacts stay inside the requested run directory:

- `model.joblib`: the newly fitted TF-IDF + classifier pipeline, for local use.
- `splits/train_ids.csv`, `splits/validation_ids.csv`: ordered IDs and labels for comparison with teammates.
- `metrics.json`, `metrics.csv`: train/validation metrics; JSON includes parameters, environment and source hashes.
- `train_predictions.csv`, `validation_predictions.csv`: per-example probabilities and decisions.
- `validation_summary.png`: ROC curve and confusion matrix.
- `selected_feature_weights.csv`: selected positive/negative logistic coefficients.
- `submission.csv`: 3,000 rows, columns `id,label`; `label` is **P(AI)** in `[0,1]`.
- `prediction_metadata.json`: prediction checks and hashes.
- `verification.json`: produced by `verify_run.py` on success.

These are generated outputs, not files that need to be committed to the course repository. A partial failed run may leave diagnostic artifacts; choose a new directory after fixing the error. Training completion is recorded by `metrics.json`, prediction completion by `prediction_metadata.json`, and verification by `verification.json`.

## Fixed method and reference results

- Human = 0; AI = 1.
- Original `train.csv` order, scikit-learn `train_test_split(test_size=0.2, random_state=42, stratify=y, shuffle=True)`. Preserve returned order.
- Training: 8,000 examples (4,000 per class). Validation: 2,000 (1,000 per class).
- Word TF-IDF, word unigrams and bigrams, `min_df=2`, `max_features=50000`, sublinear TF, L2 normalization, no stopword removal.
- Logistic regression: `C=1`, L2 regularization (`l1_ratio=0` in scikit-learn 1.9.1), `lbfgs`, `max_iter=1000`, `tol=1e-4`, `random_state=42`.
- Vocabulary, IDF and classifier fit only on the 8,000 training examples. Thread limit: 1. Decision threshold for accuracy/F1: 0.5.
- No hyperparameter search, all-10,000-row refit, pseudo-labeling or learning-curve experiment is included.

| Partition | AUROC | Accuracy at 0.5 | AI F1 at 0.5 |
|---|---:|---:|---:|
| Training | 0.9931939375 | 0.95175 | 0.9501034126 |
| Validation | 0.936066 | 0.867 | 0.8597046414 |

The original baseline's official Kaggle submission **58183934** scored **0.82964** on the public leaderboard on November 9, 2026 (America/New_York). This is a historical result, not a new submission by these scripts and not classification accuracy. Private score was unavailable at that check. The public/validation difference is unresolved; no generalization or probability-calibration guarantee is claimed.

## Course integration

The course target is [sekhari/human-or-ai-submissions](https://github.com/sekhari/human-or-ai-submissions), base branch `main`, one PR per team. Place this whole module under `submissions/<actual-kaggle-team-name>/model1/` in the team fork. The team must determine the actual directory name; this module does not assume a group number.

The parent team README should list all names/NetIDs, the Kaggle team name, this module's commands, seeds and dependency setup, and the full group's AI disclosure. Retain meaningful existing commit history when integrating. The team's final prediction command may use another selected model or ensemble; this entrypoint reproduces Model 1's baseline submission only.

Commit the source and documentation files. Do not commit competition CSVs, environment folders, local output folders, model checkpoints, credentials, or temporary archives. Other models and all remaining report experiments are the team's separate integration responsibility.

## AI assistance

OpenAI Codex/ChatGPT assisted with code implementation, debugging, packaging, and explanations. The workflow was actually executed on official data, and split IDs, metrics and predictions were checked. The student discussed and learned the TF-IDF, logistic regression and evaluation workflow; this disclosure does not claim the student manually wrote all code or inspected every line. See `DEVELOPMENT.md` for the concrete changes. No external LLM API was used to label or predict competition articles.

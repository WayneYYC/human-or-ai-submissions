# Group Number 21 — code contributions

This directory is the integration location for the existing `submission/group-number-21` branch. This contribution contains **Model 1** and the **Q2/Q3 experiments using Model 2**; it does not claim to be the complete team submission.

## Included modules

| Module | Contributor / attribution | Purpose |
|---|---|---|
| [model1](model1/README.md) | Yaoan Xie, NetID yx749, GitHub yx749-lab | Untuned word TF-IDF + logistic regression baseline; training, evaluation and Kaggle-format prediction |
| [model2_q2_q3](model2_q2_q3/README.md) | Q2/Q3 extension: Yaoan Xie; original Model 2 attribution retained in the module | Character TF-IDF + calibrated linear SVM learning curve and one-round pseudo-labeling comparison |

Each module README includes the tested Python version, pinned dependencies, exact run/verification commands, fixed seed 42, and AI-assistance disclosure. Use each module's own virtual environment because the recorded numerical dependency versions differ. Download the original competition CSVs from Kaggle; no raw competition text, fitted checkpoints or per-article predictions are included here.

## Reproduction

Follow `model1/README.md` to run `run_all.py` and `verify_run.py`. `run_all.py` trains from the official data and generates `submission.csv` for the Model 1 baseline.

Follow `model2_q2_q3/README.md` to run `run_q2.py`, then `run_q3.py`, then `verify.py`. These reproduce the final Q2/Q3 report experiments. Q2/Q3 do not generate the team's final Kaggle submission.

Reference validation AUROC: Model 1 **0.936066**; supervised Model 2 **0.970440**; Q3 pseudo-labeling **0.970820**. Q2 uses Model 2, not the earlier historical Model 1 learning curve.

## Team integration status

The team still needs to integrate its remaining model/experiment code, complete the full member list and exact Kaggle team name, identify the final prediction entrypoint, and submit one team PR to `sekhari/human-or-ai-submissions`. Existing teammate files are outside this contribution and have not been altered. The newly added files are all under this team directory.

OpenAI Codex/ChatGPT assisted Yaoan Xie with implementation, debugging, verification and report drafting. The included experiments were run on official data and independently checked; no external LLM API was used to label or predict competition passages. Module documentation describes the checks and limitations.

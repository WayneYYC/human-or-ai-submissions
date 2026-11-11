
# Group Number 7 — Human or AI?

Cornell Tech | CS5785 Applied Machine Learning | Fall 2026

## Team
- Yuchen (Wayne) Yang — yy2458
- Yuqi (Yokie) Zhang — yz3657
- Yaoan (Ben) Xie — yx749

Kaggle Team Name: Group Number 7

## Project Overview
We developed three machine learning classifiers to distinguish
human-written from AI-generated text.

### Model 1 — Word TF-IDF + Logistic Regression
Validation AUROC: 0.936066

### Model 2 — Character TF-IDF + Calibrated Linear SVM
Validation AUROC: 0.970440

### Model 3 — ConvoKit Stylometry + Logistic Regression
Validation AUROC: 0.826659

## Selected Final Model
Model 2: Character TF-IDF (2–4 grams) with a
sigmoid-calibrated linear SVM.

Random seed: 42
Training examples: 8,000
Validation examples: 2,000

## Additional Experiments
- Hyperparameter tuning
- Learning curves
- Semi-supervised pseudo-labeling
- Text-length subgroup analysis and error analysis
- Probability-based model ensembles

## Reproducibility
Training, evaluation, and Kaggle prediction commands
will be documented after integration of the model code.

## AI Assistance
ChatGPT and OpenAI Codex were used during development
for coding assistance, debugging, and experiment analysis.
The team will document individual contributions and
verification in the final submission.

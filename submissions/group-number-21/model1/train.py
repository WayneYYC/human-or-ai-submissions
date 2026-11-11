"""Train and evaluate the unchanged word TF-IDF + logistic regression baseline."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from common import (SEED, THRESHOLD, THREADS, TFIDF, CLASSIFIER, OFFICIAL_HASHES,
                    require, sha256, write_json, environment, read_official,
                    split_data, check_probabilities)

def score(frame, probabilities):
    predicted = (probabilities >= THRESHOLD).astype(int)
    return {'rows': len(frame), 'auroc': float(roc_auc_score(frame.label, probabilities)),
            'accuracy_at_0_5': float(accuracy_score(frame.label, predicted)),
            'f1_ai_at_0_5': float(f1_score(frame.label, predicted)),
            'confusion_matrix_true_rows_predicted_columns':
                confusion_matrix(frame.label, predicted, labels=[0, 1]).tolist()}

def train_model(data_dir, output_dir):
    data_dir, out = Path(data_dir).resolve(), Path(output_dir).resolve()
    require(not out.exists(), f'Output directory already exists: {out}. Choose a new directory.')
    # Only train.csv is read here. Validation texts never fit vocabulary, IDF or classifier.
    raw = read_official(data_dir, 'train.csv')
    train, validation = split_data(raw)
    out.mkdir(parents=True)
    (out / 'splits').mkdir()
    for name, frame in [('train', train), ('validation', validation)]:
        frame[['id', 'label']].to_csv(out / 'splits' / f'{name}_ids.csv', index=False)
    model = Pipeline([('tfidf', TfidfVectorizer(**TFIDF)),
                      ('classifier', LogisticRegression(**CLASSIFIER))])
    print('Training a NEW pipeline on 8,000 texts; holding out 2,000 texts...', flush=True)
    with threadpool_limits(limits=THREADS), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        model.fit(train.text, train.label)
        fit_seconds = time.perf_counter() - start
        require(not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                'Training did not converge. No completed run is claimed.')
        require(model.classes_.tolist() == [0, 1], 'Expected Human=0 and AI=1.')
        train_p = model.predict_proba(train.text)[:, 1]
        validation_p = model.predict_proba(validation.text)[:, 1]
    for frame, probability in [(train, train_p), (validation, validation_p)]:
        check_probabilities(probability, len(frame))
    model_path = out / 'model.joblib'
    joblib.dump(model, model_path, compress=3)
    with threadpool_limits(limits=THREADS):
        reloaded_p = joblib.load(model_path).predict_proba(validation.text)[:, 1]
    require(np.allclose(reloaded_p, validation_p, rtol=0, atol=1e-12), 'Model reload mismatch.')
    metrics = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'Word TF-IDF + Logistic Regression; untuned baseline',
        'label_mapping': {'0': 'Human', '1': 'AI'},
        'split': {'method': 'train_test_split on original train.csv order', 'seed': SEED,
                  'test_size': 0.2, 'stratify': 'label', 'shuffle': True,
                  'train_rows': len(train), 'validation_rows': len(validation)},
        'parameters': {'tfidf': TFIDF, 'classifier': CLASSIFIER,
                       'decision_threshold': THRESHOLD, 'thread_limit': THREADS},
        'train': score(train, train_p), 'validation': score(validation, validation_p),
        'feature_count': len(model['tfidf'].vocabulary_),
        'solver_iterations': model['classifier'].n_iter_.tolist(),
        'fit_seconds_including_tfidf': fit_seconds,
        'warnings': [str(w.message) for w in caught], 'environment': environment(),
        'train_csv_sha256': OFFICIAL_HASHES['train.csv'], 'model_sha256': sha256(model_path),
        'training_reads_test_or_unlabeled': False, 'trained_from_scratch': True,
        'source_sha256': {name: sha256(Path(__file__).parent / name)
                          for name in ['common.py', 'train.py', 'requirements.txt']},
    }
    for name, frame, probability in [('train', train, train_p), ('validation', validation, validation_p)]:
        pd.DataFrame({'id': frame.id, 'true_label': frame.label, 'p_ai': probability,
                      'predicted_label_at_0_5': (probability >= THRESHOLD).astype(int)}).to_csv(
                          out / f'{name}_predictions.csv', index=False)
    pd.DataFrame([{'partition': name, **{key: val for key, val in metrics[name].items()
                  if not isinstance(val, list)}} for name in ['train', 'validation']]).to_csv(
                      out / 'metrics.csv', index=False)
    names, coef = model['tfidf'].get_feature_names_out(), model['classifier'].coef_[0]
    indices = np.r_[np.argsort(coef)[:20], np.argsort(coef)[-20:][::-1]]
    pd.DataFrame({'feature': names[indices], 'coefficient': coef[indices]}).to_csv(
        out / 'selected_feature_weights.csv', index=False)
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout='constrained')
    RocCurveDisplay.from_predictions(validation.label, validation_p, ax=axes[0],
                                    name='Validation', plot_chance_level=True)
    axes[0].set_title('Model 1: validation ROC')
    ConfusionMatrixDisplay.from_predictions(validation.label, (validation_p >= THRESHOLD).astype(int),
        labels=[0, 1], display_labels=['Human', 'AI'], ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title('Validation decisions (threshold = 0.5)')
    fig.savefig(out / 'validation_summary.png', dpi=180)
    plt.close(fig)
    # Written last: metrics.json signals completed training and evaluation.
    write_json(out / 'metrics.json', metrics)
    print(f"PASS: validation AUROC={metrics['validation']['auroc']:.6f}, "
          f"accuracy={metrics['validation']['accuracy_at_0_5']:.3%}", flush=True)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='A NEW run directory.')
    args = parser.parse_args()
    train_model(args.data_dir, args.output_dir)

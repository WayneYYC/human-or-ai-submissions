"""Independently check a completed run against its official inputs and saved model."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from common import (THREADS, THRESHOLD, require, sha256, write_json, read_official,
                    split_data, check_probabilities)

def verify(data_dir, run_dir):
    run = Path(run_dir).resolve()
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    require(sha256(run / 'model.joblib') == metrics['model_sha256'], 'Model checksum changed.')
    model = joblib.load(run / 'model.joblib')
    require(model.classes_.tolist() == [0, 1], 'Label mapping changed.')
    train, validation = split_data(read_official(data_dir, 'train.csv'))
    checks = {}
    for name, frame in [('train', train), ('validation', validation)]:
        ids = pd.read_csv(run / 'splits' / f'{name}_ids.csv', dtype={'id': str})
        require(ids.equals(frame[['id', 'label']].reset_index(drop=True)), f'{name}: split differs.')
        saved = pd.read_csv(run / f'{name}_predictions.csv', dtype={'id': str})
        require(saved.id.tolist() == frame.id.tolist(), f'{name}: ordered IDs differ.')
        require(saved.true_label.tolist() == frame.label.tolist(), f'{name}: labels differ.')
        with threadpool_limits(limits=THREADS):
            p = model.predict_proba(frame.text)[:, 1]
        check_probabilities(p, len(frame))
        require(np.allclose(p, saved.p_ai, rtol=0, atol=1e-12), f'{name}: probabilities differ.')
        # Rank-sum AUROC independently checks the sklearn metric used during training.
        positive = frame.label.to_numpy() == 1
        n1, n0 = int(positive.sum()), int((~positive).sum())
        auc = float((rankdata(p)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        decisions = (p >= THRESHOLD).astype(int)
        accuracy = float((decisions == frame.label.to_numpy()).mean())
        require(abs(auc - metrics[name]['auroc']) < 1e-12, f'{name}: AUROC differs.')
        require(abs(accuracy - metrics[name]['accuracy_at_0_5']) < 1e-12, f'{name}: accuracy differs.')
        require(np.array_equal(decisions, saved.predicted_label_at_0_5), f'{name}: decisions differ.')
        checks[name] = {'rows': len(frame), 'independent_auroc': auc, 'accuracy': accuracy}
    test = read_official(data_dir, 'test.csv')
    sample = read_official(data_dir, 'sample_submission.csv')
    submitted = pd.read_csv(run / 'submission.csv', dtype={'id': str})
    require(submitted.columns.tolist() == ['id', 'label'], 'Submission columns differ.')
    require(submitted.id.tolist() == test.id.tolist() == sample.id.tolist(), 'Submission IDs differ.')
    with threadpool_limits(limits=THREADS):
        p = model.predict_proba(test.text)[:, 1]
    check_probabilities(submitted.label.to_numpy(), 3000)
    require(np.allclose(p, submitted.label, rtol=0, atol=1e-12), 'Test predictions differ.')
    metadata = json.loads((run / 'prediction_metadata.json').read_text(encoding='utf-8'))
    require(sha256(run / 'submission.csv') == metadata['submission_sha256'], 'Submission checksum changed.')
    report = {'status': 'PASS', 'checks': checks, 'test_rows': len(test),
              'saved_model_predictions_recomputed': True, 'independent_rank_auroc_verified': True}
    write_json(run / 'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    verify(args.data_dir, args.run_dir)

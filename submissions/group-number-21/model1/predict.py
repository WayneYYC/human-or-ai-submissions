"""Generate Kaggle probabilities from a model produced by train.py."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from common import (THREADS, OFFICIAL_HASHES, require, sha256, write_json,
                    read_official, check_probabilities)

def predict_test(data_dir, run_dir):
    run = Path(run_dir).resolve()
    output = run / 'submission.csv'
    require(not output.exists(), f'Existing prediction file preserved: {output}')
    require((run / 'metrics.json').is_file(), 'Run train.py successfully before prediction.')
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    # Compare with this run's newly saved model, never with an old baseline binary.
    require(sha256(run / 'model.joblib') == metrics['model_sha256'], 'This run model has changed.')
    test = read_official(data_dir, 'test.csv')
    sample = read_official(data_dir, 'sample_submission.csv')
    require(test.id.tolist() == sample.id.tolist(), 'Test/template ID order differs.')
    train_ids = pd.read_csv(run / 'splits/train_ids.csv', dtype={'id': str})
    val_ids = pd.read_csv(run / 'splits/validation_ids.csv', dtype={'id': str})
    require(not set(test.id) & (set(train_ids.id) | set(val_ids.id)), 'Test/labeled ID overlap.')
    model = joblib.load(run / 'model.joblib')
    require(model.classes_.tolist() == [0, 1], 'Expected Human=0 and AI=1.')
    with threadpool_limits(limits=THREADS):
        probability = model.predict_proba(test.text)[:, 1]
    check_probabilities(probability, len(test))
    require(np.unique(probability).size > 1, 'Unexpected constant predictions.')
    pd.DataFrame({'id': sample.id, 'label': probability}).to_csv(output, index=False)
    with output.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['id', 'label'], 'Bad output columns.')
        rows = list(reader)
    require([row['id'] for row in rows] == sample.id.tolist(), 'CSV ID round-trip mismatch.')
    require(np.array_equal(np.array([float(row['label']) for row in rows]), probability),
            'CSV probability round-trip mismatch.')
    write_json(run / 'prediction_metadata.json', {
        'rows': len(rows), 'columns': ['id', 'label'], 'label': 'P(AI), not hard class labels',
        'submission_sha256': sha256(output), 'model_sha256': metrics['model_sha256'],
        'test_sha256': OFFICIAL_HASHES['test.csv'],
        'template_sha256': OFFICIAL_HASHES['sample_submission.csv'],
        'min_probability': float(probability.min()), 'max_probability': float(probability.max()),
        'kaggle_upload_performed': False,
    })
    print(f'PASS: {len(rows)} probabilities written to {output}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    predict_test(args.data_dir, args.run_dir)

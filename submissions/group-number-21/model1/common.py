"""Frozen Model 1 configuration and shared input/output checks."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
THRESHOLD = 0.5
THREADS = 1
TFIDF = dict(analyzer='word', ngram_range=(1, 2), lowercase=True,
             token_pattern=r'(?u)\b\w\w+\b', stop_words=None, min_df=2,
             max_df=1.0, max_features=50000, sublinear_tf=True,
             norm='l2', use_idf=True, smooth_idf=True)
CLASSIFIER = dict(C=1.0, l1_ratio=0.0, solver='lbfgs', max_iter=1000,
                  tol=1e-4, random_state=SEED)
OFFICIAL_HASHES = {
    'train.csv': '325e345faf88fb1c4203278098c282e40319a87b4bf17e094d2f235f51d0c07e',
    'test.csv': '02209fdb74607f6b5f09a43300ed74761edf76f161cf9fd579cf43fb213884cd',
    'sample_submission.csv': '2fdb037826ae4a83136c393fa9d29fd74d899b2dd06d32caadbe3a56e356193d',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def environment():
    names = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib', 'matplotlib', 'threadpoolctl']
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'versions': {name: importlib.metadata.version(name) for name in names}}

def read_official(data_dir, name):
    path = Path(data_dir).resolve() / name
    require(path.is_file(), f'Missing {path}. Download the official competition data first.')
    require(sha256(path) == OFFICIAL_HASHES[name],
            f'{name} differs from the official reference. Use the original unedited CSV.')
    frame = pd.read_csv(path, dtype={'id': str})
    columns = ['id', 'text', 'label'] if name == 'train.csv' else (
        ['id', 'text'] if name == 'test.csv' else ['id', 'label'])
    require(frame.columns.tolist() == columns, f'Unexpected columns in {name}.')
    require(len(frame) == (10000 if name == 'train.csv' else 3000), f'Unexpected row count in {name}.')
    require(frame.id.notna().all() and frame.id.is_unique, f'Invalid IDs in {name}.')
    if 'text' in frame:
        require(frame.text.notna().all() and frame.text.str.strip().str.len().gt(0).all(),
                f'Missing or empty text in {name}.')
    if name == 'train.csv':
        require(frame.label.value_counts().to_dict() == {0: 5000, 1: 5000}, 'Unexpected training labels.')
    return frame

def split_data(raw):
    train, validation = train_test_split(raw, test_size=0.2, random_state=SEED,
                                         stratify=raw.label, shuffle=True)
    require(len(train) == 8000 and len(validation) == 2000, 'Wrong split sizes.')
    require(not set(train.id) & set(validation.id), 'Train/validation ID overlap.')
    require(train.label.value_counts().to_dict() == {0: 4000, 1: 4000}, 'Wrong training balance.')
    require(validation.label.value_counts().to_dict() == {0: 1000, 1: 1000}, 'Wrong validation balance.')
    return train, validation

def check_probabilities(probabilities, rows):
    require(probabilities.shape == (rows,), 'Wrong prediction shape.')
    require(np.isfinite(probabilities).all(), 'Non-finite probabilities.')
    require(((probabilities >= 0) & (probabilities <= 1)).all(), 'Probabilities outside [0, 1].')

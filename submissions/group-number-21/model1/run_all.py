"""Train from official data and generate predictions; does not upload anything."""
import argparse
from pathlib import Path
from train import train_model
from predict import predict_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    train_model(args.data_dir, args.output_dir)
    predict_test(args.data_dir, args.output_dir)

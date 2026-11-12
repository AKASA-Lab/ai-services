"""Aggregate prediction CSVs and write metrics and confusion matrix figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..evaluation import evaluate_predictions, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions_csv)
    results = evaluate_predictions(predictions)
    save_evaluation(predictions, results, args.output_dir)
    print(f"Primary metric macro F1: {results['summary']['macro_f1']}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()

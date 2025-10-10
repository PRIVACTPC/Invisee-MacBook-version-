#!/usr/bin/env python3
"""
Orchestrator for the run_simulation.py pipeline.
This script delegates all work to run_pipeline() without modifying run_simulation.py.
"""

import sys
import argparse
from pathlib import Path
from run_simulation import run_pipeline

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrator for run_simulation.py federated-learning pipeline."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to the pooled CSV file."
    )
    parser.add_argument(
        "--group-col",
        default="client_id",
        help="Column used to split the dataset into logical clients."
    )
    parser.add_argument(
        "--target-col",
        default="y",
        help="Name of the regression/target column."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of clients to simulate."
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    try:
        global_model, pred_path, fig_dir, relationships = run_pipeline(
            args.csv,
            args.group_col,
            args.target_col,
            args.num_clients
        )
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print(f" Predictions written to: {pred_path.resolve()}")
    print(f" Figures stored in:    {fig_dir.resolve()}")
    print("\nTop 10 relationships (corr or R²):")
    for target, preds, metric, value in relationships[:10]:
        preds_str = ", ".join(preds)
        print(f"  {metric:>4} | {target} ~ {preds_str} = {value:.3f}")

if __name__ == "__main__":
    main()

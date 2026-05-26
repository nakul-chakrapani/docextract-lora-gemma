"""CLI script to compute evaluation metrics from results JSON."""

import argparse
import json
from src.evaluation.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser(description="Compute metrics from inference results.")
    parser.add_argument("--results", required=True, help="Path to results JSON file.")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    metrics = compute_metrics(data["results"])
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
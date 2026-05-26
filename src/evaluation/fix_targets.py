"""Reprocess targets in existing results JSON using fixed _extract_target_json."""

import argparse
import json
from dataclasses import asdict
from datasets import load_dataset
from src.data.dataset import _extract_target_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON file.")
    parser.add_argument("--output", required=True, help="Path to save fixed results JSON.")
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    # load existing results
    with open(args.results) as f:
        data = json.load(f)

    # load CORD to get fresh annotations
    cord = load_dataset("naver-clova-ix/cord-v2", split=args.split)

    # reprocess targets
    for result in data["results"]:
        idx = result["sample_id"]
        ground_truth = json.loads(cord[idx]["ground_truth"])
        gt_parse = json.dumps(ground_truth.get("gt_parse", {}))
        receipt = _extract_target_json(gt_parse)
        result["target"] = json.dumps(asdict(receipt))

    # save fixed results
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Fixed targets saved to {args.output}")

if __name__ == "__main__":
    main()
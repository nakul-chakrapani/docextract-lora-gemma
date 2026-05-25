"""Batch inference stubs used during offline evaluation."""

import argparse
import json
from pathlib import Path

from src.inference.predict import load_model, parse_model_output, run_inference
from src.data.dataset import load_cord

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for batch inference.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Run batch inference and save predictions.")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to run.")
    parser.add_argument("--output", required=True, help="JSON output file for predictions.")
    return parser


def main() -> None:
    """CLI entrypoint for batch evaluation."""
    args = build_arg_parser().parse_args()

    # load dataset
    dataset = load_cord(split=args.split, max_samples=args.num_samples)

    # load model and processor
    model, processor = load_model(args.model)

    # Loop over samples one by one to run inference and collect predictions
    predictions = []
    for index, item in enumerate(dataset):
        image = item.get("image", "")
        prompt = item.get("prompt", "")
        raw_output, latency_ms = run_inference(model=model, processor=processor, image=image, prompt=prompt)
        parsed_output = parse_model_output(raw_output)
        final_output = {
            "sample_id": index,
            "prediction": parsed_output,
            "target": item.get("target", ""),
            "latency_ms": latency_ms,
            "valid": parsed_output is not None
        }
        predictions.append(final_output)

        print(f"Sample {index+1}/{len(dataset)} — latency: {latency_ms:.0f}ms — valid: {parsed_output is not None}")


    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # metadata wrapper before writing
    output_data = {
        "model": args.model,
        "split": args.split,
        "num_samples": len(dataset),
        "results": predictions
    }
    output_path.write_text(json.dumps(output_data, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

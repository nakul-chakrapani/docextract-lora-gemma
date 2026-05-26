import json
from src.evaluation.metrics import flatten_sample, compute_field_f1, compute_sample_f1

with open("results/e4b_zeroshot_100_fixed.json") as f:
    data = json.load(f)

# find low scoring samples
for sample in data["results"]:
    prediction = sample["prediction"]
    target = json.loads(sample["target"])
    f1 = compute_sample_f1(prediction, target)
    
    if f1 < 0.5:  # focus on bad ones first
        print(f"\nSample {sample['sample_id']} — F1: {f1:.2f}")
        flattened = flatten_sample(prediction, target)
        for key, pred_val, target_val in flattened:
            field_f1 = compute_field_f1(pred_val, target_val)
            if field_f1 < 1.0:
                print(f"  {key}: pred={pred_val}, target={target_val}, f1={field_f1:.2f}")
"""Metric function stubs for extraction quality and hallucination analysis."""

from collections import Counter
import json


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference strings.

    Args:
        prediction: Model output text.
        reference: Ground-truth text.

    Returns:
        Token-level F1 score in the range [0.0, 1.0].
    """
    pred_tokens: list[str] = prediction.split()
    ref_tokens: list[str] = reference.split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common: int = sum(overlap.values())
    if common == 0:
        return 0.0

    precision: float = common / len(pred_tokens)
    recall: float = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_field_f1(predicted: object, target: object) -> float:
    """Wrapper around compute_f1 that handles None and non-string types."""
    if predicted is None and target is None:
        return 1.0

    if predicted is None or target is None:
        return 0.0
    
    # normalize numeric strings to float for comparison
    try:
        if float(str(predicted)) == float(str(target)):
            return 1.0
    except (ValueError, TypeError):
        pass
    
    return compute_f1(str(predicted).lower(), str(target).lower())


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact-match score for two normalized strings.

    Args:
        prediction: Model output text.
        reference: Ground-truth text.

    Returns:
        1.0 if normalized strings match, else 0.0.
    """
    normalized_pred: str = prediction.strip().lower()
    normalized_ref: str = reference.strip().lower()
    return 1.0 if normalized_pred == normalized_ref else 0.0


def compute_hallucination_rate(predictions: list[str], allowed_tokens: set[str]) -> float:
    """Estimate hallucination rate as fraction of out-of-vocabulary tokens.

    Args:
        predictions: List of model output strings.
        allowed_tokens: Set of valid tokens expected in domain outputs.

    Returns:
        Fraction of generated tokens not present in allowed_tokens.
    """
    total_tokens: int = 0
    hallucinated_tokens: int = 0

    for pred in predictions:
        tokens: list[str] = pred.split()
        total_tokens += len(tokens)
        hallucinated_tokens += sum(1 for token in tokens if token not in allowed_tokens)

    if total_tokens == 0:
        return 0.0
    return hallucinated_tokens / total_tokens


def flatten_sample(prediction: dict | None, target: dict) -> list[tuple[str, object, object]]:
    """Flatten prediction/target receipt fields into comparable tuples.

    Supported input schemas:
    - prediction: menu_items, total_price, tax
    - target: menu, total, tax

    Menu entries are normalized to (name, count, price), sorted, and flattened
    into indexed keys like menu_0_name, menu_0_count, menu_0_price.

    Args:
        prediction: Model output dictionary.
        target: Ground-truth dictionary.

    Returns:
        List of (key_name, predicted_value, target_value) tuples.
    """
    if prediction is None:
        prediction = {}

    def _normalize_menu_item(item: dict | None) -> dict[str, object]:
        if not isinstance(item, dict):
            return {"name": None, "count": None, "price": None}
        return {
            "name": item.get("name", item.get("nm")),
            "count": item.get("count", item.get("cnt")),
            "price": item.get("price"),
        }
    
    def _get_total(pred: dict) -> object:
        return pred.get("total") or pred.get("total_price") or pred.get("grand_total")

    def _get_tax(pred: dict) -> object:
        return pred.get("tax") or pred.get("tax_price") or pred.get("vat")

    pred_menu_raw = prediction.get("menu_items", [])
    target_menu_raw = target.get("menu", [])

    pred_menu_source = pred_menu_raw if isinstance(pred_menu_raw, list) else []
    target_menu_source = target_menu_raw if isinstance(target_menu_raw, list) else []

    pred_menu = [_normalize_menu_item(item) for item in pred_menu_source]
    target_menu = [_normalize_menu_item(item) for item in target_menu_source]

    pred_menu.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("count") or ""), str(item.get("price") or "")))
    target_menu.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("count") or ""), str(item.get("price") or "")))

    flattened: list[tuple[str, object, object]] = []
    max_items = max(len(pred_menu), len(target_menu))
    for idx in range(max_items):
        pred_item = pred_menu[idx] if idx < len(pred_menu) else {"name": None, "count": None, "price": None}
        target_item = target_menu[idx] if idx < len(target_menu) else {"name": None, "count": None, "price": None}

        flattened.append((f"menu_{idx}_name", pred_item.get("name"), target_item.get("name")))
        flattened.append((f"menu_{idx}_count", pred_item.get("count"), target_item.get("count")))
        flattened.append((f"menu_{idx}_price", pred_item.get("price"), target_item.get("price")))

    flattened.append(("tax", _get_tax(prediction), _get_tax(target)))
    flattened.append(("total", _get_total(prediction), _get_total(target)))
    return flattened

def compute_sample_f1(prediction: dict | None, target: dict) -> float:
    """Compute average F1 score across all comparable fields in a sample."""
    flattened = flatten_sample(prediction, target)
    if not flattened:
        return 0.0

    f1_scores = []
    for key, pred_value, target_value in flattened:
        f1 = compute_field_f1(pred_value, target_value)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics across all evaluation results.

    Args:
        results: List of result dicts with keys: prediction, target, valid, latency_ms.

    Returns:
        Dict with json_validity, macro_f1, exact_match, avg_latency_ms.
    """
    num_samples = len(results)
    if num_samples == 0:
        return {}

    valid_count = sum(1 for r in results if r["valid"])
    latencies = [r["latency_ms"] for r in results]

    f1_scores = []
    exact_matches = []

    for r in results:
        prediction = r["prediction"]
        target = json.loads(r["target"]) if isinstance(r["target"], str) else r["target"]

        f1 = compute_sample_f1(prediction, target)
        f1_scores.append(f1)
        exact_matches.append(1.0 if f1 == 1.0 else 0.0)

    return {
        "num_samples": num_samples,
        "json_validity": valid_count / num_samples,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "exact_match": sum(exact_matches) / len(exact_matches),
        "avg_latency_ms": sum(latencies) / len(latencies),
    }


    

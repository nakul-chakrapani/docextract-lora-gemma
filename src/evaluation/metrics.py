"""Metric function stubs for extraction quality and hallucination analysis."""

from collections import Counter


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

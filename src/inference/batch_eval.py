"""Batch inference stubs used during offline evaluation."""

from typing import Iterable

from .predict import predict_single


def batch_predict(prompts: Iterable[str], model_name_or_path: str) -> list[str]:
    """Run batched predictions by iterating over prompts.

    Args:
        prompts: Iterable collection of prompt strings.
        model_name_or_path: Identifier or local path to the model.

    Returns:
        List of prediction strings.
    """
    return [predict_single(prompt=prompt, model_name_or_path=model_name_or_path) for prompt in prompts]

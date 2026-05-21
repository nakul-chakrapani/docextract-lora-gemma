"""Single-example prediction stubs for fine-tuned extraction models."""

from typing import Any


def predict_single(prompt: str, model_name_or_path: str, generation_kwargs: dict[str, Any] | None = None) -> str:
    """Run one prediction for a formatted prompt.

    Args:
        prompt: Model-ready prompt text.
        model_name_or_path: Identifier or local path to the model.
        generation_kwargs: Optional generation settings.

    Returns:
        Placeholder prediction string.
    """
    _ = (prompt, model_name_or_path, generation_kwargs)
    return "prediction_stub"

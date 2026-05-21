"""Prompt template helpers for Gemma 4 multimodal instruction tuning."""


def build_gemma4_multimodal_prompt(instruction: str, image_placeholder: str = "<image>", context: str | None = None) -> str:
    """Build a Gemma 4 chat-formatted multimodal prompt.

    The template uses explicit chat-turn markers and leaves model output empty.

    Args:
        instruction: The extraction task instruction.
        image_placeholder: Placeholder token representing an image input.
        context: Optional extra textual context to include for grounding.

    Returns:
        A formatted prompt string in Gemma 4 chat style.
    """
    context_block: str = f"\nContext:\n{context.strip()}" if context else ""
    user_message: str = (
        f"{image_placeholder}\n"
        f"Task:\n{instruction.strip()}"
        f"{context_block}"
    )
    return (
        "<start_of_turn>user\n"
        f"{user_message}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

"""Prompt template helpers for Gemma 4 multimodal instruction tuning."""


def build_gemma4_multimodal_prompt(context: str | None = None) -> str:
    """Build a Gemma 4 chat-formatted multimodal prompt.

    The template uses explicit chat-turn markers and leaves model output empty.

    Args:
        context: Optional extra textual context to include for grounding.

    Returns:
        A formatted prompt string in Gemma 4 chat style.
    """
    context_block: str = f"\nContext:\n{context.strip()}" if context else ""
    instruction: str = "Extract all fields from this receipt as a JSON object.\nReturn only valid JSON with no explanation.\nFields: menu items (name, count, price), subtotal tax, total price"

    user_message: str = (
        "<|image|>\n"
        f"Task:\n{instruction.strip()}"
        f"{context_block}"
    )
    return (
        "<bos><start_of_turn>user\n"
        f"{user_message}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


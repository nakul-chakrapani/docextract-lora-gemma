"""Placeholder tests for data and prompt utility behavior."""

from src.data.prompt_templates import build_gemma4_multimodal_prompt


class TestDataUtilities:
    """Smoke tests for data-related helper functions."""

    def test_build_gemma4_multimodal_prompt_contains_turn_tokens(self) -> None:
        """Ensure the prompt includes required Gemma turn markers."""
        prompt: str = build_gemma4_multimodal_prompt("Extract invoice number.")
        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt

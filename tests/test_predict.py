"""Unit tests for inference output parsing helpers."""

from src.inference.predict import parse_model_output


class TestParseModelOutput:
    """Tests for parsing model text output into JSON."""

    def test_parse_model_output_with_valid_json(self) -> None:
        """Parses valid model output JSON payload correctly."""
        output = (
            "<start_of_turn>user\nExtract fields\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            '{"invoice_id": "INV-123", "total": 42.5}'
            "<end_of_turn>"
        )

        result = parse_model_output(output)

        assert result == {"invoice_id": "INV-123", "total": 42.5}

    def test_parse_model_output_with_invalid_json_content(self) -> None:
        """Returns None when model turn exists but JSON is malformed."""
        output = (
            "<start_of_turn>model\n"
            '{"invoice_id": "INV-123", "total": }'
            "<end_of_turn>"
        )

        result = parse_model_output(output)

        assert result is None

    def test_parse_model_output_missing_model_turn_marker(self) -> None:
        """Returns None when the model turn marker is missing entirely."""
        output = '{"invoice_id": "INV-123", "total": 42.5}<end_of_turn>'

        result = parse_model_output(output)

        assert result is None

    def test_parse_model_output_with_markdown_code_block(self) -> None:
        """Handles model output wrapped in markdown code fences."""
        output = (
            "<start_of_turn>model\n"
            "```json\n"
            '{"menu": [], "tax": 100.0, "total": 1000.0}\n'
            "```"
            "<end_of_turn>"
        )

        result = parse_model_output(output)

        assert result == {"menu": [], "tax": 100.0, "total": 1000.0}

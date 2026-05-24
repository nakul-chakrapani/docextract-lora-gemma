"""Placeholder tests for data and prompt utility behavior."""

import json

import pytest

from src.data.dataset import _extract_target_json, _parse_price
from src.data.prompt_templates import build_gemma4_multimodal_prompt


class TestDataUtilities:
    """Smoke tests for data-related helper functions."""

    def test_build_gemma4_multimodal_prompt_contains_turn_tokens(self) -> None:
        """Ensure the prompt includes required Gemma turn markers."""
        prompt: str = build_gemma4_multimodal_prompt("Extract invoice number.")
        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt


class TestParsePrice:
    """Unit tests for OCR price normalization."""

    @pytest.mark.parametrize(
        ("raw_price", "expected"),
        [
            ("1,234.56", 1234.56),
            ("2,500.", 2500.0),
            ("$45.00..", 45.0),
            ("0", 0.0),
            ("", 0.0),
            ("abc", 0.0),
            ("40,000.", 40000.0),
            ("75,000", 75000.0),
            ("1,591,600", 1591600.0)
        ],
    )
    def test_parse_price_normalizes_value(self, raw_price: str, expected: float) -> None:
        """Prices should be cleaned and converted to a float safely."""
        value = _parse_price(raw_price)
        assert isinstance(value, float)
        assert value == expected


class TestExtractTargetJson:
    """Unit tests for gt_parse extraction into ReceiptSchema."""

    def test_extract_target_json_with_valid_cord_payload(self) -> None:
        """Valid payload should parse menu, tax, and total fields correctly."""
        gt_parse = json.dumps(
            {
                "menu": [
                    {"nm": "Americano", "cnt": "2 ea", "price": "4,500."},
                    {"nm": "Bagel", "cnt": "1", "price": "2,300"},
                ],
                "sub_total": {"tax_price": "680."},
                "total": {"total_price": "7,480"},
            }
        )

        result = _extract_target_json(gt_parse)

        assert len(result.menu) == 2
        assert result.menu[0].name == "Americano"
        assert result.menu[0].count == "2"
        assert result.menu[0].price == 4500.0
        assert result.menu[1].name == "Bagel"
        assert result.menu[1].count == "1"
        assert result.menu[1].price == 2300.0
        assert result.tax == 680.0
        assert result.total == 7480.0

    def test_extract_target_json_missing_sub_total_sets_tax_zero(self) -> None:
        """Missing sub_total should not crash and tax should fall back to 0.0."""
        gt_parse = json.dumps(
            {
                "menu": [{"nm": "Tea", "cnt": "1", "price": "1,200"}],
                "total": {"total_price": "1,200"},
            }
        )

        result = _extract_target_json(gt_parse)

        assert len(result.menu) == 1
        assert result.tax == 0.0
        assert result.total == 1200.0

    def test_extract_target_json_invalid_json_returns_empty_schema(self) -> None:
        """Invalid JSON should return an empty ReceiptSchema without raising."""
        result = _extract_target_json("{not valid json")

        assert result.menu == []
        assert result.tax is None
        assert result.total is None

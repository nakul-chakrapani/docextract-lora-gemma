"""Placeholder tests for evaluation metrics."""

from src.evaluation.metrics import compute_exact_match


class TestMetrics:
    """Smoke tests for metric helper functions."""

    def test_compute_exact_match_passes_on_case_insensitive_match(self) -> None:
        """Verify exact match is case-insensitive after normalization."""
        assert compute_exact_match("Total", "total") == 1.0

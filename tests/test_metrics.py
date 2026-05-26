"""Placeholder tests for evaluation metrics."""

from src.evaluation.metrics import compute_exact_match, flatten_sample


class TestMetrics:
    """Smoke tests for metric helper functions."""

    def test_compute_exact_match_passes_on_case_insensitive_match(self) -> None:
        """Verify exact match is case-insensitive after normalization."""
        assert compute_exact_match("Total", "total") == 1.0

    def test_flatten_sample_maps_prediction_and_target_keys(self) -> None:
        """Flattened output should align total_price->total and tax fields."""
        prediction = {
            "menu_items": [{"name": "Americano", "count": "1", "price": 4500.0}],
            "tax": 300.0,
            "total_price": 4800.0,
        }
        target = {
            "menu": [{"name": "Americano", "count": "1", "price": 4500.0}],
            "tax": 300.0,
            "total": 4800.0,
        }

        flattened = flatten_sample(prediction, target)

        assert ("menu_0_name", "Americano", "Americano") in flattened
        assert ("menu_0_count", "1", "1") in flattened
        assert ("menu_0_price", 4500.0, 4500.0) in flattened
        assert ("tax", 300.0, 300.0) in flattened
        assert ("total", 4800.0, 4800.0) in flattened

    def test_flatten_sample_sorts_menu_by_name(self) -> None:
        """Menu list should be sorted by item name before flattening indexes."""
        prediction = {
            "menu_items": [
                {"name": "Zebra Cake", "count": "1", "price": 200.0},
                {"name": "Apple Pie", "count": "2", "price": 300.0},
            ],
            "tax": 50.0,
            "total_price": 850.0,
        }
        target = {
            "menu": [
                {"name": "Zebra Cake", "count": "1", "price": 200.0},
                {"name": "Apple Pie", "count": "2", "price": 300.0},
            ],
            "tax": 50.0,
            "total": 850.0,
        }

        flattened = flatten_sample(prediction, target)

        assert flattened[0] == ("menu_0_name", "Apple Pie", "Apple Pie")
        assert flattened[3] == ("menu_1_name", "Zebra Cake", "Zebra Cake")

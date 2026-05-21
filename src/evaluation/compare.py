"""Helpers to compare multiple experiment runs by scalar metrics."""


def compare_runs(run_to_score: dict[str, float]) -> list[tuple[str, float]]:
    """Return runs sorted from best to worst score.

    Args:
        run_to_score: Mapping of run name to a scalar score.

    Returns:
        Sorted list of (run_name, score) tuples in descending score order.
    """
    return sorted(run_to_score.items(), key=lambda item: item[1], reverse=True)

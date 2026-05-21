"""Dataset loading helpers for CORD and VRDU-style document understanding tasks."""

from typing import Any

from datasets import Dataset, load_dataset


def load_cord(split: str = "train", **kwargs: Any) -> Dataset:
    """Load the CORD dataset split from Hugging Face Datasets.

    Args:
        split: Dataset split name, for example "train" or "test".
        **kwargs: Additional keyword arguments forwarded to datasets.load_dataset.

    Returns:
        A Hugging Face Dataset object for the requested split.
    """
    return load_dataset("naver-clova-ix/cord-v2", split=split, **kwargs)


def load_vrdu(split: str = "train", dataset_name: str = "vrdu", **kwargs: Any) -> Dataset:
    """Load a VRDU-style dataset split from Hugging Face Datasets.

    This is a project stub. Replace dataset_name with your concrete VRDU dataset ID.

    Args:
        split: Dataset split name, for example "train" or "validation".
        dataset_name: Hugging Face dataset identifier for VRDU data.
        **kwargs: Additional keyword arguments forwarded to datasets.load_dataset.

    Returns:
        A Hugging Face Dataset object for the requested split.
    """
    return load_dataset(dataset_name, split=split, **kwargs)

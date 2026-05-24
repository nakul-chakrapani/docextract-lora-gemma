"""Dataset loading helpers for CORD and VRDU-style document understanding tasks."""

import re
from typing import Any

from datasets import Dataset, load_dataset
from .schema import MenuItem, ReceiptSchema
import json
from .prompt_templates import build_gemma4_multimodal_prompt
from dataclasses import asdict


def _parse_price(price_str: str) -> float:
    """Parse OCR-extracted price text into a float.

    The parser normalizes common OCR noise by removing commas and non-numeric
    trailing symbols while preserving an optional leading sign and decimal dot.
    """
    if not price_str:
        return 0.0

    cleaned = price_str.strip().replace(",", "")
    if not cleaned:
        return 0.0

    # Keep only characters relevant to numeric parsing; this removes OCR noise.
    cleaned = re.sub(r"[^0-9.]", "", cleaned)

    # Drop trailing punctuation/symbol artifacts while keeping decimal digits.
    cleaned = re.sub(r"[.]+$", "", cleaned)
    if cleaned in {"", "+", "-", ".", "+.", "-."}:
        return 0.0

    try:
        return float(cleaned)
    except ValueError:
        return 0.0
    
def _extract_target_json(gt_parse: str) -> ReceiptSchema:
    """Extract the target JSON object from the gt_parse string.

    The gt_parse field contains a JSON string with a "target" key that holds
    the structured data we want to extract. This function parses the gt_parse
    string and returns the target data as a dictionary.
    """

    final_menu_items: list[MenuItem] = []
    tax: float | None = None
    total: float | None = None

    # parse gt_parse to get menu items, tax, and total
    try:
        parsed = json.loads(gt_parse)
    except json.JSONDecodeError as je:
        print(f"Error parsing gt_parse JSON: {je}")
        return ReceiptSchema(menu=final_menu_items, tax=tax, total=total)

    if not parsed:
        return ReceiptSchema(menu=final_menu_items, tax=tax, total=total)

    try:
        menu = parsed.get("menu", [])

        # get menu items from gt_parse
        for item in menu:
            name = item.get("nm", "")
            count = item.get("cnt", "")
            count = count.split(" ")[0] if count else ""
            price = _parse_price(item.get("price", "0.0"))
            menu_item = MenuItem(name=name, count=count, price=price)
            final_menu_items.append(menu_item)

        # Get tax and total from gt_parse
        tax_raw = parsed.get("sub_total").get("tax_price", "0.0") if parsed.get("sub_total") else "0.0"
        total_raw = parsed.get("total").get("total_price", "0.0") if parsed.get("total") else "0.0"
        tax = _parse_price(tax_raw)
        total = _parse_price(total_raw)
    except Exception as be:
        print(f"Error extracting target data from gt_parse: {be}")

    return ReceiptSchema(menu=final_menu_items, tax=tax, total=total)


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

class CORDDataset:
    """Custom Dataset class for CORD that processes gt_parse into structured schema."""

    def __init__(self, dataset: Dataset, max_samples: int | None = None, use_ocr: bool = False) -> None:
        self.dataset = dataset.select(range(max_samples)) if max_samples is not None else dataset
        self.use_ocr = use_ocr

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item.get("image", "")
        prompt = build_gemma4_multimodal_prompt(context=None) if not self.use_ocr else build_gemma4_multimodal_prompt(context=item.get("ocr_text", ""))

        ground_truth = json.loads(item["ground_truth"])
        gt_parse = json.dumps(ground_truth.get("gt_parse", {}))
        target_json = json.dumps(asdict(_extract_target_json(gt_parse)))

        return {
            "image": image,
            "prompt": prompt,
            "target": target_json
        }
    
def load_cord(split: str = "train", max_samples: int | None = None, use_ocr: bool = False) -> CORDDataset:
    """Load the CORD dataset split from Hugging Face Datasets.

    Args:
        split: Dataset split name, for example "train" or "test".

    Returns:
        A CORDDataset object for the requested split.
    """

    valid_splits = {"train", "validation", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
    
    dataset = load_dataset("naver-clova-ix/cord-v2", split=split)
    return CORDDataset(dataset, max_samples=max_samples, use_ocr=use_ocr)

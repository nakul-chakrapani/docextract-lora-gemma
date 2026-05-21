"""Entry-point stubs for LoRA training orchestration."""

from pathlib import Path
from typing import Any

import yaml

from .config import LoRAConfig, TrainingConfig


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return data


def build_configs(lora_path: Path, training_path: Path) -> tuple[LoRAConfig, TrainingConfig]:
    """Create dataclass config objects from YAML files.

    Args:
        lora_path: Path to LoRA hyperparameter YAML.
        training_path: Path to training hyperparameter YAML.

    Returns:
        Tuple containing LoRAConfig and TrainingConfig instances.
    """
    lora_raw: dict[str, Any] = load_yaml_config(lora_path)
    training_raw: dict[str, Any] = load_yaml_config(training_path)
    return LoRAConfig(**lora_raw), TrainingConfig(**training_raw)


def run_training(lora_cfg: LoRAConfig, train_cfg: TrainingConfig) -> None:
    """Run model training with the provided configuration.

    Args:
        lora_cfg: LoRA adapter configuration.
        train_cfg: General training configuration.
    """
    _ = (lora_cfg, train_cfg)
    print("Training stub: integrate transformers, peft, and accelerate here.")


def main() -> None:
    """Load default configs and execute the training stub."""
    root: Path = Path(__file__).resolve().parents[2]
    lora_cfg, train_cfg = build_configs(
        root / "configs" / "lora_config.yaml",
        root / "configs" / "training_config.yaml",
    )
    run_training(lora_cfg, train_cfg)


if __name__ == "__main__":
    main()

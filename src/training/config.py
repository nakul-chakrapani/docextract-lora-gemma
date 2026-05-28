"""Dataclass-based configuration objects for LoRA and training runs."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class LoRAConfig:
    """Configuration for LoRA adapter hyperparameters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model fine-tuning and evaluation scheduling."""

    model_name_or_path: str = "google/gemma-2-2b"
    output_dir: str = "checkpoints/docextract-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    seed: int = 42
    bf16: bool = True

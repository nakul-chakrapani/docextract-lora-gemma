"""Entry-point stubs for LoRA training orchestration."""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import yaml
from peft import LoraConfig
from peft import prepare_model_for_kbit_training

from src.training.config import LoRAConfig, TrainingConfig
from src.data.dataset import load_cord
from trl import SFTConfig, SFTTrainer
from torch.nn.utils.rnn import pad_sequence

from dataclasses import fields


def create_quantization_config() -> BitsAndBytesConfig:
    """Bitsandbytes for quantization configuration setup."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return quantization_config

def create_lora_config(cfg: LoRAConfig, target_modules: list[str]) -> LoraConfig:
    """Create a LoRAConfig instance with default values."""
    
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
        target_modules=target_modules,
    )
    return lora_config

def get_target_modules(model) -> list[str]:
    target_modules = set()
    for name, module in model.named_modules():
        if type(module).__name__ == "Gemma4TextAttention":
            for child_name, child_module in module.named_children():
                if type(child_module).__name__ == "Linear4bit":
                    if child_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        target_modules.add(f"{name}.{child_name}")
    return list(target_modules)

def load_model_and_processor(model_name: str, quantization_config: BitsAndBytesConfig):
    """Load the pre-trained model and processor with quantization."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    return model, processor



def make_collate_fn(processor):
    def collate_fn(samples: list[dict]) -> dict:
        images = [s["image"] for s in samples]
        prompts = [s["prompt"] for s in samples]
        targets = [s["target"] for s in samples]
    
        texts = [p + t for p, t in zip(prompts, targets)]
        
        all_inputs = [processor(images=img, text=txt, return_tensors="pt") 
                      for img, txt in zip(images, texts)]
        
        all_prompt_inputs = [processor(images=img, text=p, return_tensors="pt")
                             for img, p in zip(images, prompts)]
    
        pad_id = processor.tokenizer.pad_token_id
    
        input_ids = pad_sequence(
            [x["input_ids"].squeeze(0) for x in all_inputs],
            batch_first=True, padding_value=pad_id
        )
        attention_mask = pad_sequence(
            [x["attention_mask"].squeeze(0) for x in all_inputs],
            batch_first=True, padding_value=0
        )
        mm_token_type_ids = pad_sequence(
            [x["mm_token_type_ids"].squeeze(0) for x in all_inputs],
            batch_first=True, padding_value=0
        )
        pixel_values = torch.cat([x["pixel_values"] for x in all_inputs], dim=0)
        image_position_ids = torch.cat([x["image_position_ids"] for x in all_inputs], dim=0)
    
        labels = input_ids.clone()
        for i, prompt_input in enumerate(all_prompt_inputs):
            prompt_length = (prompt_input["input_ids"] != pad_id).sum().item()
            labels[i, :prompt_length] = -100
    
        # mask padding tokens in labels too
        labels[input_ids == pad_id] = -100
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
            "labels": labels,
        }
    return collate_fn

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


def run_training(model: AutoModelForCausalLM, processor: AutoProcessor,
                  lora_cfg: LoRAConfig, train_cfg: TrainingConfig) -> None:
    """Run model training with the provided configuration.

    Args:
        model: The pre-trained model to fine-tune.
        processor: The associated processor for preparing inputs.
        lora_cfg: LoRA adapter configuration.
        train_cfg: General training configuration.
    """

    # Build SFTConfig for trl trainer
    training_args = SFTConfig(
        output_dir=train_cfg.output_dir,
        max_length=train_cfg.max_length,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_train_epochs,
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        bf16=train_cfg.bf16,
        save_strategy="epoch",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}
        )
    
    train_dataset = load_cord(split="train")

    target_modules = get_target_modules(model)
    # print(f"Target modules found: {target_modules}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=create_lora_config(lora_cfg, target_modules),
        data_collator=make_collate_fn(processor),
    )

    trainer.train()
    


def load_config(path: str, cls):
    with open(path) as f:
        data = yaml.safe_load(f)
    # only pass keys that exist in the dataclass
    valid_keys = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid_keys})

def main() -> None:
    # configs 
    train_cfg = load_config("configs/training_config.yaml", TrainingConfig)
    lora_cfg = load_config("configs/lora_config.yaml", LoRAConfig)

    quantization_config = create_quantization_config()
    model, processor = load_model_and_processor(train_cfg.model_name_or_path, quantization_config)

    # print trainable parameters for sanity check
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})")

    run_training(model, processor, lora_cfg, train_cfg)


if __name__ == "__main__":
    main()

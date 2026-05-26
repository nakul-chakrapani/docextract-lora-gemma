"""Single-example prediction stubs for fine-tuned extraction models."""

from typing import Any
import json


def predict_single(prompt: str, model_name_or_path: str, generation_kwargs: dict[str, Any] | None = None) -> str:
    """Run one prediction for a formatted prompt.

    Args:
        prompt: Model-ready prompt text.
        model_name_or_path: Identifier or local path to the model.
        generation_kwargs: Optional generation settings.

    Returns:
        Placeholder prediction string.
    """
    _ = (prompt, model_name_or_path, generation_kwargs)
    return "prediction_stub"

def parse_model_output(output: str) -> dict[str, Any] | None:
    """Parse raw model output into structured data.

    This is a placeholder for post-processing logic that would convert the model's text output
    into a structured format like a dictionary or dataclass instance.

    Args:
        output: Raw string output from the model.

    Returns:
        Structured data parsed from the model output.
    """
    
    if not output:
        return None

    marker = "<start_of_turn>model\n"
    if marker not in output:
        return None
    
    try:
        model_output = output.split(marker)[-1].strip()
        # strip thinking block if present
        if "<channel|>" in model_output:
            model_output = model_output.split("<channel|>")[-1].strip()

        model_output = model_output.replace("```json", "").replace("```", "").strip()
        model_output = model_output.replace("<end_of_turn>", "").replace("<turn|>", "").strip()

        # extract just the JSON object
        start = model_output.find("{")
        end = model_output.rfind("}") + 1
        if start == -1 or end == 0:
            return None
            
        # print(f"DEBUG raw: {model_output[:300]}")
        json_str = model_output[start:end]
        return json.loads(json_str)
    except json.JSONDecodeError as je:
        print(f"Error parsing model output JSON: {je}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing model output: {e}")
        return None

def load_model(model_name: str) -> tuple:
    """Load the fine-tuned model and processor.
    This is a placeholder for model loading logic, which would typically involve
    using Hugging Face Transformers to load a model and its associated processor/tokenizer.
    Args:
        model_name: Identifier or path for the model to load.
    Returns:
        A tuple containing the loaded model and processor.
    """

    import platform
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    quantization_config = None
    if platform.system() == "Linux":
        from transformers import BitsAndBytesConfig  
        import torch
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # then pass quantization_config to model loading
    # it's None on Mac which just loads without quantization

    processor = AutoProcessor.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return model, processor

def run_inference(model: Any, processor: Any, image: Any, prompt: str) -> tuple[str, float]:
    """Run inference on a single image and prompt.

    This is a placeholder for the actual inference logic, which would involve
    processing the image and prompt, running them through the model, and returning
    the raw output along with inference latency.

    Args:
        model: The loaded model to use for inference.
        processor: The associated processor/tokenizer for preparing inputs.
        image: The input image data.
        prompt: The formatted prompt text.

    Returns:
        A tuple containing the raw model output string and the inference latency in seconds.
    """

    import time

    # Process multimodal inputs with the model's processor.
    model_inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Best-effort device placement for eager execution setups.
    try:
        model_device = next(model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
    except Exception:
        # With some sharded/device_map="auto" setups, this step may be unnecessary.
        pass

    start = time.perf_counter()
    generated = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
    )
    latency = (time.perf_counter() - start) * 1000  # Convert to milliseconds

    raw_output = processor.decode(generated[0], skip_special_tokens=False)
    return raw_output, latency

    

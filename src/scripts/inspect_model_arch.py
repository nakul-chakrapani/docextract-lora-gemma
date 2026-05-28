from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForCausalLM.from_config(config)

for name, module in model.named_modules():
    print(name, type(module).__name__)
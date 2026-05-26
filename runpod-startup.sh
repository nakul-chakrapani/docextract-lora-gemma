cat > /workspace/runpod_startup.sh << 'EOF'
#!/bin/bash
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

cd /workspace
git clone https://github.com/nakul-chakrapani/docextract-lora-gemma.git
cd docextract-lora-gemma
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[train]" -q
echo "Setup complete"
EOF
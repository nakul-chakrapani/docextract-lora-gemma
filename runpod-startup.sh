#!/bin/bash
cd /workspace
git clone https://github.com/nakul-chakrapani/docextract-lora.git
cd docextract-lora
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements-train.txt -q
echo "Ready"
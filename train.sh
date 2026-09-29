#!/usr/bin/env bash
# Stop on the first failure: without this, a broken stage 1 still lets training
# start on whatever splits happened to be lying around.
set -euo pipefail

cd "$(dirname "$0")"

export HF_TOKEN="hf_YourTokenHere"

CONFIG=conf/config.json

# Which GPUs to train on comes from the config, nowhere else.
 GPUS=$(python -c "import json;print(','.join(map(str,json.load(open('$CONFIG'))['gpus'])))")
 #NPROC=$(python -c "import json;print(len(json.load(open('$CONFIG'))['gpus']))")

# Data Preparation
python scripts/01_prepare_data.py

# Training. 
CUDA_VISIBLE_DEVICES=$GPUS python scripts/02_train.py


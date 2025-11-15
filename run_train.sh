#!/bin/bash
# Train the Spatial Semantic Identifier model.
set -e

OUTPUT_DIR="output_dir"
DATA_DIR="data"
CONFIG_PATH="config/content_3m.json"

mkdir -p "$OUTPUT_DIR"

torchrun --nnodes=1 --nproc_per_node=2 --master_port=21081 generate_sid/train.py \
  --train_file "$DATA_DIR/train.txt" \
  --val_file "$DATA_DIR/val.txt" \
  --batch_size 256 \
  --num_epochs 15000 \
  --early_stop 2000 \
  --eval_steps 1000 \
  --save_per_epochs 1000 \
  --learning_rate 2e-4 \
  --state_dict_save_path "$OUTPUT_DIR" \
  --use_columns "text_emb,image_emb,cf_emb" \
  --config_path "$CONFIG_PATH"
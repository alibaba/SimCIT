#!/bin/bash
# Script to generate Spatial Semantic IDs (SID) for test data using trained model.
torchrun --nproc_per_node=2 --master_port=21041 generate_sid/infer.py \
  --input_file data/test.txt \
  --output_file results.txt \
  --state_dict_save_path ./output_dir/checkpoint_best.pt \
  --use_columns "text_emb,image_emb,cf" \
  --config_path config/content_3m.json \
  --batch_size 256 

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=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

cd ~/lerobot_workspace/lerobot
rm -rf outputs/train/smolvla_so101

echo "============================================"
echo "SmolVLA Training - batch_size=128"
echo "GPU: RTX 3090 (24GB)"
echo "Start: $(date)"
echo "============================================"

lerobot-train \
  --dataset.repo_id=so101_test_01_25 \
  --dataset.root=/home/ubuntu/so101_test_01_25 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.repo_id=yzyzylll057/smolvla_so101 \
  --policy.push_to_hub=true \
  --output_dir=outputs/train/smolvla_so101 \
  --job_name=smolvla_so101 \
  --batch_size=128 \
  --num_workers=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=true \
  --wandb.project=lerobot-so101

echo "Done: $(date)"

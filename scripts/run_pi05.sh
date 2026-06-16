#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/lerobot_workspace/lerobot
rm -rf outputs/train/pi05_so101

echo "=== PI0.5 bfloat16 Training (gemma_2b default) ==="

lerobot-train \
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.empty_cameras=1 \
  --policy.repo_id=yzyzylll057/pi05_so101_test \
  --policy.push_to_hub=true \
  --dataset.repo_id=so101_test_01_25 \
  --dataset.root=/home/ubuntu/so101_test_01_25 \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --batch_size=1 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=true \
  --wandb.project=lerobot-so101 \
  --output_dir=outputs/train/pi05_so101

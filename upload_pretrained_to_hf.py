#!/usr/bin/env python3
"""Upload pretrained PI0 model to HuggingFace Hub."""

import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Configuration
LOCAL_MODEL_DIR = Path("outputs/pretrained_model_pi0")
REPO_ID = "yzyzylll057/pi0_so101_tissue"  # 与训练输出目录名对应
PRIVATE = False
COMMIT_MESSAGE = "Upload PI0 pretrained model (so101 tissue, 65 episodes)"

def main():
    if not LOCAL_MODEL_DIR.exists():
        print(f"[ERROR] 模型目录不存在: {LOCAL_MODEL_DIR.resolve()}")
        sys.exit(1)

    # 列出文件
    files = list(LOCAL_MODEL_DIR.glob("*"))
    print(f"[INFO] 模型目录: {LOCAL_MODEL_DIR.resolve()}")
    print(f"[INFO] 上传文件列表 ({len(files)} files):")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    # 创建仓库 (exist_ok=True 避免重复报错)
    print(f"\n[INFO] 创建/获取仓库: {REPO_ID}")
    api = HfApi()
    repo_url = api.create_repo(
        repo_id=REPO_ID,
        private=PRIVATE,
        exist_ok=True,
    )
    print(f"[INFO] 仓库 URL: {repo_url}")

    # 上传文件夹
    print(f"\n[INFO] 开始上传到 https://huggingface.co/{REPO_ID} ...")
    commit_info = upload_folder(
        repo_id=REPO_ID,
        folder_path=str(LOCAL_MODEL_DIR),
        commit_message=COMMIT_MESSAGE,
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
    )
    print(f"\n[SUCCESS] 上传完成!")
    print(f"[SUCCESS] 查看: {commit_info}")

if __name__ == "__main__":
    main()

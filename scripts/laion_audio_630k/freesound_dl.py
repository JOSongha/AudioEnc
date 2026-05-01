import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("HF_HOME", "/mnt/tmp/cache/huggingface")
from huggingface_hub import snapshot_download

print("[laion-dl] starting freesound_no_overlap snapshot...", flush=True)
path = snapshot_download(
    repo_id="Meranti/CLAP_freesound",
    repo_type="dataset",
    local_dir="/mnt/tmp/datasets/laion_freesound",
    allow_patterns=[
        "freesound_no_overlap/*",
        "*.csv",
        "README.md",
    ],
    max_workers=16,
)
print(f"[laion-dl] DONE, path={path}", flush=True)

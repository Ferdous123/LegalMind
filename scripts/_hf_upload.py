"""One-shot helper to push the deploy tree to ferdus/LegalMind.

Reads HF_TOKEN from the env. Uses upload_large_folder (resumable,
multi-stream) plus hf-transfer (if installed) for fast uploads.

Not part of the application; safe to delete after the Space is shipped.
"""
import os
from huggingface_hub import upload_folder

DEPLOY = r"C:\Users\Ratul Khan\AppData\Local\Temp\hf_deploy"
REPO = "ferdus/LegalMind"

# hf-transfer (env var HF_HUB_ENABLE_HF_TRANSFER=1) accelerates each
# individual file's upload — Rust-based, multi-stream, much faster than
# the default Python stack. The Space repo already exists, so plain
# upload_folder is fine (no implicit create_repo issues).
url = upload_folder(
    folder_path=DEPLOY,
    repo_id=REPO,
    repo_type="space",
    token=os.environ["HF_TOKEN"],
    commit_message="Initial read-only demo deploy (populated data, GPU endpoints blocked)",
    ignore_patterns=[
        "__pycache__", "*.pyc", ".git", ".venv", "logs",
        "*.bak", "*.expbak",
        "data/chroma*", "data/palace*", "data/queue_state.json",
    ],
)
print("uploaded ->", url)

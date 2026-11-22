"""
Host and run train_qwen25_3b.ipynb interactively on Modal (modal.com).

Spins up an interactive JupyterLab web server inside a GPU-accelerated Modal container
(T4 / A10G) with Unsloth, PyTorch, CUDA, and datasets pre-installed.
All ml-pipeline files (train_qwen25_3b.ipynb, train.jsonl, val.jsonl) are automatically
mounted into /root/ml-pipeline, and checkpoints are stored in a persistent Modal Volume.

Quickstart:
  1. pip install modal
  2. modal setup
  3. modal serve ml-pipeline/serve_notebook_modal.py
     (prints live HTTPS URL with token to open Jupyter in your browser)
"""

import subprocess
from pathlib import Path
import modal

app = modal.App("studyverse-jupyter")

# Persistent volume for checkpoints and GGUF exports
checkpoints_volume = modal.Volume.from_name("studyverse-checkpoints", create_if_missing=True)

ML_PIPELINE_DIR = Path(__file__).resolve().parent

# Container image with PyTorch, CUDA, Unsloth, and JupyterLab
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "torchaudio",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "jupyterlab~=4.2.0",
    )
    .run_commands(
        'pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git"'
    )
    .add_local_dir(str(ML_PIPELINE_DIR), remote_path="/root/ml-pipeline")
)

JUPYTER_PORT = 8888
JUPYTER_TOKEN = "studyverse"


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 4,
    volumes={"/root/checkpoints": checkpoints_volume},
)
@modal.web_server(port=JUPYTER_PORT, startup_timeout=180)
def run_jupyter():
    """Starts JupyterLab bound to 0.0.0.0 and serves it via Modal's public HTTPS proxy."""
    cmd = (
        f"jupyter lab --ip=0.0.0.0 --port={JUPYTER_PORT} --no-browser --allow-root "
        f"--IdentityProvider.token='{JUPYTER_TOKEN}' "
        f"--ServerApp.token='{JUPYTER_TOKEN}' "
        f"--ServerApp.allow_origin='*' "
        f"--ServerApp.disable_check_xsrf=True "
        f"--ServerApp.root_dir='/root/ml-pipeline'"
    )
    print("🚀 Launching JupyterLab on Modal GPU (T4)...")
    print(f"🔑 Pre-configured token: {JUPYTER_TOKEN}")
    print(f"📂 Workspace directory: /root/ml-pipeline (contains train_qwen25_3b.ipynb)")
    subprocess.Popen(cmd, shell=True)

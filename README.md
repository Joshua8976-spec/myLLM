pytho# MyLLM: Fine-Tuning & Modal Deployment Pipeline

End-to-end pipeline for dataset preparation, QLoRA fine-tuning (Qwen 2.5 3B Instruct), and serverless model hosting on **Modal** (modal.com).

## Overview

- **Modal Serverless Hosting**: The model is hosted on Modal using `ml-pipeline/serve_llm_modal.py`, exposing standard OpenAI-compatible endpoints (`/v1/chat/completions` with streaming SSE support, `/v1/models`, `/health`).
- **Persistent Volume Checkpointing**: Checkpoints and fine-tuned LoRA adapters are stored directly in Modal's persistent volume (`studyverse-checkpoints`), connecting training directly to inference.
- **Autoscaling & Cost Efficiency**: Scales to 0 when idle and warms up on demand.

## Repository Structure

- **`ml-pipeline/`**
  - `prepare_dataset.py`: Synthetic & curriculum dataset generation and validation into ChatML JSONL format.
  - `train.jsonl` / `val.jsonl`: Training and validation dataset splits across 4 modes (`tutor`, `quiz`, `diagnostic`, `companion`).
  - `train_modal.py`: Modal cloud GPU training script using Unsloth 4-bit QLoRA.
  - `generate_notebook.py` / `train_qwen25_3b.ipynb`: Interactive Jupyter training notebook (Modal & Colab compatible).
  - `serve_llm_modal.py`: Production OpenAI-compatible FastAPI inference server on Modal.
  - `serve_notebook_modal.py`: Cloud JupyterLab server runner on Modal.
  - `.env.example`: Frontend environment variable configuration template.

For detailed setup instructions, visit [`ml-pipeline/README.md`](./ml-pipeline/README.md).

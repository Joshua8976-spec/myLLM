# MyLLM: Fine-Tuning & Deployment Pipeline

End-to-end pipeline for dataset preparation, QLoRA fine-tuning (Qwen 2.5 3B Instruct), GGUF quantization, and model serving on Modal and Hugging Face Spaces.

## Repository Structure

- **`ml-pipeline/`**
  - `prepare_dataset.py`: Synthetic & curriculum dataset generation and validation into ChatML JSONL format.
  - `train.jsonl` / `val.jsonl`: Training and validation dataset splits.
  - `train_modal.py`: Modal cloud GPU training script using Unsloth.
  - `generate_notebook.py` / `train_qwen25_3b.ipynb`: Interactive Jupyter training notebook (Colab / Modal compatible).
  - `serve_llm_modal.py`: High-throughput vLLM serving on Modal.
  - `serve_notebook_modal.py`: Headless JupyterLab runner on Modal.
  - `hf-space/`: Docker configuration to deploy the quantized GGUF model on Hugging Face Spaces.

For detailed instructions and architecture diagrams, check [`ml-pipeline/README.md`](./ml-pipeline/README.md).

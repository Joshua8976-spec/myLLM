#!/usr/bin/env bash
set -e

MODEL_PATH="${MODEL_DIR}/${MODEL_FILENAME}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Downloading model ${MODEL_FILENAME} from ${MODEL_REPO}..."
    python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='${MODEL_REPO}', filename='${MODEL_FILENAME}', local_dir='${MODEL_DIR}')"
    echo "Model downloaded successfully to ${MODEL_PATH}"
fi

echo "Starting llama-cpp-python OpenAI-compatible server on port ${PORT}..."
exec python3 -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --n_ctx 2048 \
    --n_threads 2

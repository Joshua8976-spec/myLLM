FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV MODEL_DIR=/app/models
ENV MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF
ENV MODEL_FILENAME=qwen2.5-3b-instruct-q4_k_m.gguf

WORKDIR /app

# Install system dependencies required for llama-cpp-python build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install llama-cpp-python with high-performance CPU server and huggingface_hub
RUN pip install --no-cache-dir \
    "llama-cpp-python[server]" \
    huggingface_hub

RUN mkdir -p ${MODEL_DIR}

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]

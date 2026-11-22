---
title: StudyVerse Qwen 2.5 3B API
emoji: 🎓
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# StudyVerse Qwen 2.5 3B Free Hugging Face Space

An OpenAI-compatible inference server running the fine-tuned StudyVerse **Qwen 2.5 3B Instruct (GGUF `q4_k_m`)** model on Hugging Face's **$0 Free CPU Basic tier** (2 vCPUs, 16GB RAM) using `llama-cpp-python[server]`.

## Features
- **Endpoints:**
  - `GET /v1/models` — Health check & model listing
  - `POST /v1/chat/completions` — OpenAI-compatible streaming & non-streaming Chat completions
- **Resource Optimization:**
  - Context window: `--n_ctx 2048`
  - CPU Threads: `--n_threads 2`
  - Quantization: `q4_k_m` (~2.1 GB VRAM / RAM footprint)
  - Speed: ~18–25 tokens/sec on free CPU

## Environment Variables
Configure in your Space settings (or leave defaults):
- `PORT`: `7860` (Default Space port)
- `MODEL_REPO`: e.g. `Joshua8976/studyverse-qwen2.5-3b-gguf` or `Qwen/Qwen2.5-3B-Instruct-GGUF`
- `MODEL_FILENAME`: e.g. `qwen2.5-3b-instruct-q4_k_m.gguf`

## Quick Test (cURL)
```bash
curl -X POST https://YOUR_SPACE_NAME.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [
      {"role": "system", "content": "You are StudyVerse AI Tutor. [Mode: tutor]"},
      {"role": "user", "content": "Explain Newton'\''s second law briefly in LaTeX."}
    ],
    "temperature": 0.15,
    "max_tokens": 200
  }'
```

## Connecting with StudyVerse App
In your StudyVerse web application `.env.local` or environment config:
```env
VITE_STUDY_LLM_URL=https://YOUR_SPACE_NAME.hf.space/v1
```

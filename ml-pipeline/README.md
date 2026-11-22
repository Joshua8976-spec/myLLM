# StudyVerse Qwen 2.5 3B Instruct Fine-Tuning & Modal Deployment Pipeline

This guide outlines the complete end-to-end pipeline for fine-tuning **Qwen 2.5 3B Instruct** on StudyVerse's academic curriculum, formulas, common exam mistakes, and student wellbeing support, and hosting it on **Modal** (modal.com) as a serverless, autoscaling OpenAI-compatible API endpoint.

---

## Architecture Overview

```
                                  +---------------------------+
                                  | StudyVerse Codebase Data  |
                                  | - CISCE Modules (Classes 6-8)
                                  | - Catalog Formulas & Definitions
                                  | - Common Exam Traps & Fixes
                                  | - Spaced Recall Flashcards & MCQs
                                  +-------------+-------------+
                                                |
                                                v
                              +---------------------------------+
                              | ml-pipeline/prepare_dataset.py  |
                              | - 4 Task Modes (ChatML)         |
                              | - Strict json.loads() check     |
                              | - 90/10 Split (seed=42)         |
                              +--------+----------------+-------+
                                       |                |
                         train.jsonl   v                v   val.jsonl
                               +---------------+----------------+
                               | Modal Cloud GPU Fine-Tuning    |
                               | (Tesla T4 / A10G / L4)         |
                               | - train_modal.py (headless)    |
                               | - serve_notebook_modal.py      |
                               | - Unsloth 4-bit QLoRA (r=16)   |
                               | - Auto-resume checkpoints      |
                               +---------------+----------------+
                                               |
                                               v Persistent Volume Mount
                               +---------------+----------------+
                               | Modal Volume:                  |
                               | 'studyverse-checkpoints'       |
                               | /root/checkpoints/adapter      |
                               +---------------+----------------+
                                               |
                                               v Direct Volume Read
                               +---------------+----------------+
                               | Modal Serverless API Server    |
                               | ml-pipeline/serve_llm_modal.py |
                               | - OpenAI-compatible endpoints  |
                               | - POST /v1/chat/completions    |
                               | - Full streaming SSE support   |
                               | - Auto-scaledown to $0 when idle|
                               +---------------+----------------+
                                               |
                                               v HTTPS Endpoint
                               +---------------+----------------+
                               | StudyVerse App Client          |
                               | src/lib/study-llm.ts           |
                               | - VITE_STUDY_LLM_URL           |
                               | - AI Tutor, Quizzes, Calm      |
                               |   Companion, Mistake Journal   |
                               +--------------------------------+
```

---

## Step 1: Prepare the Dataset Locally

Generate the validated ChatML dataset (`train.jsonl` and `val.jsonl`) from StudyVerse's curriculum assets:

```bash
python ml-pipeline/prepare_dataset.py
```

### Outputs:
- `ml-pipeline/train.jsonl` (90% training split)
- `ml-pipeline/val.jsonl` (10% validation split)
- Strict validation guarantees 100% parseable JSON on all `[Mode: quiz]` items.
- Note: `*.jsonl` files are automatically tracked or excluded via `.gitignore`.

---

## Step 2: Fine-Tuning & Hosting on Modal (modal.com)

Modal provides serverless cloud GPU containers with persistent storage volumes. Checkpoints and adapters saved during training are directly accessible by the inference server with zero file copying or manual uploading.

### Prerequisites:
Install the Modal CLI and authenticate your Modal account:
```bash
pip install modal
modal setup
```

### Option A: Run Automated Headless Fine-Tuning on Modal GPU
Run the complete fine-tuning pipeline directly from your terminal:

```bash
modal run ml-pipeline/train_modal.py
```

- Fine-tunes `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` for 120 steps on a Tesla T4 GPU.
- Runs before-and-after evaluations across all 4 modes.
- Saves the LoRA adapter directly to the `studyverse-checkpoints` persistent volume.
- Automatically resumes from any previous checkpoint if interrupted.

---

### Option B: Interactive JupyterLab Training on Modal GPU
Launch a remote JupyterLab server on a Modal GPU container with your dataset mounted:

```bash
modal serve ml-pipeline/serve_notebook_modal.py
```

- Modal creates a secure HTTPS URL (e.g. `https://<user>--studyverse-jupyter-run-jupyter.modal.run`).
- Open the URL in your browser with the default token: `studyverse`.
- Run `train_qwen25_3b.ipynb` cell-by-cell.

For permanent background hosting of the notebook server:
```bash
modal deploy ml-pipeline/serve_notebook_modal.py
```

---

### Option C: Deploy the OpenAI-Compatible Inference Server on Modal
Deploy the serverless model server to host your fine-tuned model:

```bash
modal deploy ml-pipeline/serve_llm_modal.py
```

- Automatically mounts the `studyverse-checkpoints` volume.
- If a fine-tuned adapter exists at `/root/checkpoints/studyverse-qwen-adapter`, it loads it onto the base model. If not, it serves the base model.
- Provides:
  - `GET /health`: Healthcheck.
  - `GET /v1/models`: OpenAI models list endpoint.
  - `POST /v1/chat/completions`: Full ChatML completions with streaming SSE (`text/event-stream`) and non-streaming modes.
- Scales down to 0 when idle to eliminate hosting costs.

Modal will output your live URL, for example:
```
https://<your-org>--studyverse-qwen-serve-modelserver-serve.modal.run
```

---

## Step 3: Configure StudyVerse Front-End

In your StudyVerse frontend project root, update `.env` or set your environment variable:

```env
# Modal Serverless Endpoint (append /v1):
VITE_STUDY_LLM_URL=https://<your-org>--studyverse-qwen-serve-modelserver-serve.modal.run/v1
VITE_STUDY_LLM_MODEL=studyverse-qwen-2.5-3b
```

### Supported Modes in StudyVerse:
- **`[Mode: tutor]`**: Socratic academic tutoring, KaTeX LaTeX formulas ($...$), syllabus grounded.
- **`[Mode: quiz]`**: Strict JSON practice questions formatted for the interactive quiz runner.
- **`[Mode: diagnostic]`**: Mistake Journal root cause diagnosis and step-by-step repair card generation.
- **`[Mode: companion]`**: Grounded emotional support, 1-minute breathing resets, 5-minute micro-tasks, and 24/7 crisis helplines (strictly isolated from academic mastery data).

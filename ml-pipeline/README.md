# StudyVerse $0 Qwen 2.5 3B Instruct Fine-Tuning & Deployment Pipeline

This guide outlines the complete end-to-end $0 pipeline for fine-tuning **Qwen 2.5 3B Instruct** on StudyVerse's academic curriculum, formulas, common exam mistakes, and student wellbeing support, exporting to GGUF `q4_k_m`, and hosting on a free Hugging Face Docker Space or local Ollama instance.

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
                               | Modal GPU / Google Colab T4 |
                               | - serve_notebook_modal.py
                               | - Unsloth 4-bit QLoRA (r=16)   |
                               | - SFTTrainer + Google Drive CV |
                               | - GGUF q4_k_m Quantization     |
                               +---------------+----------------+
                                               |
                                               v GGUF Push
                               +---------------+----------------+
                               | Hugging Face Hub Model Repo    |
                               +---------------+----------------+
                                               |
                     +-------------------------+-------------------------+
                     v                                                   v
        +----------------------------+                     +----------------------------+
        | Free HF Docker Space       |                     | Local Offline Execution    |
        | ml-pipeline/hf-space/      |                     | Ollama / llama-cpp-python  |
        | - CPU Basic (2 vCPU, 16GB) |                     | - Port 11434 or 8080       |
        | - llama-cpp-python[server] |                     +--------------+-------------+
        +--------------+-------------+                                    |
                       |                                                  |
                       +-----------------------+--------------------------+
                                               |
                                               v
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
- Note: `*.jsonl` files are automatically excluded from git tracking via `.gitignore`.

---

## Step 2: Run & Host on Modal (modal.com)

Modal provides serverless cloud GPU containers (Tesla T4, A10G, L4) with persistent storage volumes. You can either host `train_qwen25_3b.ipynb` in an interactive cloud JupyterLab session or run automated headless training.

### Prerequisites:
Install the Modal CLI and authenticate your free Modal account:
```bash
pip install modal
modal setup
```

### Option A: Host `train_qwen25_3b.ipynb` Interactively in JupyterLab on Modal GPU
Launch a remote JupyterLab server on a Modal GPU container (T4) with your dataset mounted and persistent volume storage:

```bash
modal serve ml-pipeline/serve_notebook_modal.py
```

- Modal creates a secure HTTPS URL (e.g. `https://<user>--studyverse-jupyter-run-jupyter.modal.run`).
- Open the URL in your browser with the default token: `studyverse`.
- `train_qwen25_3b.ipynb`, `train.jsonl`, and `val.jsonl` are directly available inside the workspace `/root/ml-pipeline`.
- Checkpoints and exported models are automatically preserved in the persistent `studyverse-checkpoints` Modal Volume at `/root/checkpoints`.

For permanent background hosting, deploy with:
```bash
modal deploy ml-pipeline/serve_notebook_modal.py
```

---

### Option B: Run Automated Headless Training on Modal GPU
If you prefer running the full fine-tuning pipeline from your terminal without opening a browser:

```bash
modal run ml-pipeline/train_modal.py
```

- Fine-tunes `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` for 120 steps on a Tesla T4 GPU.
- Runs before-and-after evaluations across all 4 modes.
- Saves the LoRA adapter and exports `q4_k_m` GGUF to the `studyverse-checkpoints` volume.
- Automatically resumes from any previous checkpoint if interrupted.

---

### Option C: Deploy OpenAI-Compatible API on Modal
Serve your fine-tuned model via an OpenAI-compatible endpoint that connects directly to StudyVerse:

```bash
modal deploy ml-pipeline/serve_llm_modal.py
```

Modal provides a live HTTPS endpoint. Set it in your `.env` or `.env.local`:
```env
VITE_STUDY_LLM_URL=https://<your-org>--studyverse-qwen-serve-modelserver-chat-completions.modal.run/v1
```

---

## Step 3: Fine-Tune on Google Colab (Free T4 GPU Alternative)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `- serve_notebook_modal.py`.
3. In the Colab menu, select **Runtime > Change runtime type > Hardware accelerator > T4 GPU**.
4. In the files pane on the left, upload your local `train.jsonl` and `val.jsonl` files (or place them in your Google Drive).
5. Run the cells step-by-step:
   - **Cell 1–2:** Installs Unsloth and fine-tuning libraries.
   - **Cell 3:** Mounts Google Drive to `/content/drive` for auto-resume checkpointing.
   - **Cell 4–5:** Loads `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` with QLoRA rank $r=16$.
   - **Cell 6:** Formats prompts with the ChatML conversation template.
   - **Cell 7:** Evaluates baseline model responses across the 4 modes.
   - **Cell 8:** Trains with `SFTTrainer` for ~120 steps (~15–20 minutes on T4).
   - **Cell 9:** Verifies post-training responses across all 4 modes.
   - **Cell 10:** Exports to GGUF `q4_k_m` and pushes to your Hugging Face Hub repository with a comprehensive Model Card.

---

## Step 4: Deploy to a Free Hugging Face Docker Space

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose:
   - **Space SDK:** `Docker`
   - **Space hardware:** `CPU basic · 2 vCPU · 16 GB · Free`
3. Upload the files inside `ml-pipeline/hf-space/`:
   - `Dockerfile`
   - `start.sh`
   - `README.md`
4. In the Space settings, configure your model repository:
   - `MODEL_REPO`: e.g. `YourUsername/studyverse-qwen2.5-3b-gguf`
   - `MODEL_FILENAME`: e.g. `studyverse-qwen2.5-3b.q4_k_m.gguf`
5. The Space will build and launch `llama-cpp-python[server]` on port `7860`.

---

## Step 5: Configure StudyVerse Front-End

In your StudyVerse project root, update `.env` or set your environment variable:

```env
# Hugging Face Space endpoint:
VITE_STUDY_LLM_URL=https://your-space-name.hf.space/v1

# Or for local development with Ollama:
# VITE_STUDY_LLM_URL=http://localhost:11434/v1
```

### Supported Modes in StudyVerse:
- **`[Mode: tutor]`**: Socratic academic tutoring, KaTeX LaTeX formulas ($...$), syllabus grounded.
- **`[Mode: quiz]`**: Strict JSON practice questions formatted for the Quiz runner.
- **`[Mode: diagnostic]`**: Mistake Journal root cause diagnosis and step-by-step repair card generation.
- **`[Mode: companion]`**: Grounded emotional support, 1-minute breathing resets, 5-minute micro-tasks, and 24/7 crisis helplines (strictly isolated from academic mastery data).

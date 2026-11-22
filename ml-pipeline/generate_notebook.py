#!/usr/bin/env python3
"""
Generates the ready-to-upload Google Colab / Modal notebook ml-pipeline/train_qwen25_3b.ipynb
Configured for Tesla T4 GPU (16GB VRAM) using Unsloth 4-bit QLoRA,
SFTTrainer with Modal Volume / Google Drive auto-resume checkpointing,
before/after evaluation across 4 modes, and LoRA adapter export for Modal serving.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "train_qwen25_3b.ipynb"


def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook() -> dict:
    cells = [
        make_cell("markdown", """# StudyVerse Qwen 2.5 3B Instruct Fine-Tuning Pipeline
### Ultra-fast, zero-cost QLoRA fine-tuning for Academic Tutoring, Adaptive Quizzes, Mistake Diagnostics & Calm Companion on Cloud GPU
Designed for execution on **Modal GPU** or **Google Colab T4** with direct deployment to **Modal Serverless Inference**.

This notebook trains **Qwen 2.5 3B Instruct** on StudyVerse's curated ChatML dataset (`train.jsonl` and `val.jsonl`) across 4 task modes:
1. `[Mode: tutor]`: Concise, syllabus-grounded academic explanations with LaTeX math.
2. `[Mode: quiz]`: Strict JSON practice questions validated for direct app rendering.
3. `[Mode: diagnostic]`: Root cause mistake analysis and remedial steps.
4. `[Mode: companion]`: Empathetic, grounded emotional support with crisis safety (isolated from academic records).

**Hardware:** Tesla T4 GPU (16GB VRAM)  
**Method:** Unsloth 4-bit QLoRA ($r=16, \\alpha=16$, max_seq_length=1024)  
**Hosting & Inference:** Modal Serverless GPU (`ml-pipeline/serve_llm_modal.py`)"""),

        make_cell("code", """# 1. Install Unsloth and compatible fine-tuning libraries
%%capture
!pip install "transformers>=4.44.0,<4.48.0" "trl<0.13.0" peft accelerate bitsandbytes datasets
!pip install --no-deps "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
!pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git" """),

        make_cell("code", """# 2. Checkpoint Storage: Modal Volume, Google Drive, or Local
import os

# Auto-detect environment: Modal persistent volume, Google Colab Drive, or local storage
if os.path.exists("/root/checkpoints"):
    CHECKPOINT_DIR = "/root/checkpoints"
    print(f"✅ Modal persistent volume detected: {CHECKPOINT_DIR}")
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        CHECKPOINT_DIR = "/content/drive/MyDrive/studyverse-qwen25-3b-checkpoints"
        print(f"✅ Google Colab Drive mounted: {CHECKPOINT_DIR}")
    except (ImportError, Exception):
        CHECKPOINT_DIR = "./checkpoints"
        print(f"ℹ️ Local / fallback checkpoint directory: {CHECKPOINT_DIR}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoint directory ready: {CHECKPOINT_DIR}")"""),

        make_cell("markdown", """### 3. Verify and Load Dataset Files
Locates `train.jsonl` and `val.jsonl` (from `ml-pipeline/prepare_dataset.py`).
Automatically searches in the current directory, `/root/ml-pipeline`, `/content`, or Google Drive."""),

        make_cell("code", """# 3. Verify and load datasets
from datasets import load_dataset

possible_dirs = [".", "/root/ml-pipeline", "/content", CHECKPOINT_DIR]
train_path = next((os.path.join(d, "train.jsonl") for d in possible_dirs if os.path.exists(os.path.join(d, "train.jsonl"))), None)
val_path = next((os.path.join(d, "val.jsonl") for d in possible_dirs if os.path.exists(os.path.join(d, "val.jsonl"))), None)

assert train_path is not None, "train.jsonl not found. Run python ml-pipeline/prepare_dataset.py or upload to workspace."
assert val_path is not None, "val.jsonl not found. Run python ml-pipeline/prepare_dataset.py or upload to workspace."

print(f"Using train dataset: {train_path}")
print(f"Using val dataset:   {val_path}")

dataset = load_dataset("json", data_files={
    "train": train_path,
    "validation": val_path
})

print(f"✅ Training samples:   {len(dataset['train'])}")
print(f"✅ Validation samples: {len(dataset['validation'])}")
print("\\nSample entry:")
print(dataset["train"][0]["messages"][:2])"""),

        make_cell("code", """# 4. Load Qwen 2.5 3B Instruct 4-bit with Unsloth
import torch
from unsloth import FastLanguageModel

max_seq_length = 1024
dtype = None # Auto detection
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=max_seq_length,
)

print("✅ Model & LoRA adapters configured successfully.")"""),

        make_cell("code", """# 5. Format ChatML Conversation Template
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

formatted_train = dataset["train"].map(formatting_prompts_func, batched=True)
formatted_val = dataset["validation"].map(formatting_prompts_func, batched=True)
print("✅ Dataset formatted with ChatML template.")"""),

        make_cell("code", """# 6. Baseline Evaluation (Before Training) across 4 Modes
FastLanguageModel.for_inference(model)

test_eval_prompts = [
    {
        "mode": "tutor",
        "system": "You are StudyVerse AI tutor. [Mode: tutor]",
        "user": "Explain Newton's Second Law and derive F = ma for constant mass."
    },
    {
        "mode": "quiz",
        "system": "You are StudyVerse AI quiz generator. [Mode: quiz]",
        "user": "Generate a medium difficulty MCQ on Coulomb's law."
    },
    {
        "mode": "diagnostic",
        "system": "You are StudyVerse AI diagnostic assistant. [Mode: diagnostic]",
        "user": "Diagnose: I answered F/4 instead of 3F/8 on two touched spheres."
    },
    {
        "mode": "companion",
        "system": "You are StudyVerse AI calm companion. [Mode: companion]",
        "user": "I have my physics exam tomorrow and I'm panicking. My mind feels completely blank."
    }
]

print("=== BASELINE RESPONSES (BEFORE TRAINING) ===")
for test_item in test_eval_prompts:
    prompt_msgs = [
        {"role": "system", "content": test_item["system"]},
        {"role": "user", "content": test_item["user"]}
    ]
    inputs = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    temp = 0.45 if test_item["mode"] == "companion" else 0.15
    outputs = model.generate(input_ids=inputs, max_new_tokens=140, temperature=temp, use_cache=True)
    decoded = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\\n--- [{test_item['mode'].upper()}] ---")
    print(decoded.strip())"""),

        make_cell("code", """# 7. SFTTrainer Fine-Tuning with Auto-Resume
from trl import SFTTrainer, SFTConfig
import glob

training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=120,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    eval_strategy="steps",
    eval_steps=30,
    save_strategy="steps",
    save_steps=30,
    save_total_limit=2,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    eval_dataset=formatted_val,
    args=training_args,
)

existing_checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*")
latest_checkpoint = max(existing_checkpoints, key=os.path.getctime) if existing_checkpoints else None

if latest_checkpoint:
    print(f"🔁 Resuming from existing checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("🚀 Starting fine-tuning from step 0...")
    trainer.train()"""),

        make_cell("code", """# 8. Post-Training Evaluation across 4 Modes
FastLanguageModel.for_inference(model)

print("=== FINE-TUNED RESPONSES (AFTER TRAINING) ===")
for test_item in test_eval_prompts:
    prompt_msgs = [
        {"role": "system", "content": test_item["system"]},
        {"role": "user", "content": test_item["user"]}
    ]
    inputs = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    temp = 0.45 if test_item["mode"] == "companion" else 0.15
    outputs = model.generate(input_ids=inputs, max_new_tokens=220, temperature=temp, use_cache=True)
    decoded = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\\n--- [{test_item['mode'].upper()}] ---")
    print(decoded.strip())"""),

        make_cell("code", """# 9. Save LoRA Adapter for Modal Serving
adapter_dir = os.path.join(CHECKPOINT_DIR, "studyverse-qwen-adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"✅ LoRA adapter saved to: {adapter_dir}")

print(\"\"\"
=============================================================================
🎉 Fine-Tuning Complete!
To serve this model on Modal:
  1. If running inside Modal JupyterLab: The adapter is already saved in the
     persistent volume 'studyverse-checkpoints'.
  2. Deploy the OpenAI-compatible server:
     modal deploy ml-pipeline/serve_llm_modal.py
  3. Set VITE_STUDY_LLM_URL in your StudyVerse app to the deployed Modal URL!
=============================================================================
\"\"\")"""),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    nb = build_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

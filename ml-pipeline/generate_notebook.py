#!/usr/bin/env python3
"""
Generates the ready-to-upload Google Colab notebook ml-pipeline/train_qwen25_3b.ipynb
Configured for free Tesla T4 GPU (16GB VRAM) using Unsloth 4-bit QLoRA,
SFTTrainer with Google Drive checkpointing/auto-resume, before/after evaluation,
and GGUF q4_k_m export with Model Card upload.
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
        make_cell("markdown", """# StudyVerse $0 Qwen 2.5 3B Instruct Fine-Tuning Pipeline
### Ultra-fast, zero-cost QLoRA fine-tuning for Academic Tutoring, Adaptive Quizzes, Mistake Diagnostics & Calm Companion on Free Google Colab T4 GPU

This notebook trains **Qwen 2.5 3B Instruct** on StudyVerse's curated ChatML dataset (`train.jsonl` and `val.jsonl`) across 4 task modes:
1. `[Mode: tutor]`: Concise, syllabus-grounded academic explanations with LaTeX math.
2. `[Mode: quiz]`: Strict JSON practice questions validated for direct app rendering.
3. `[Mode: diagnostic]`: Root cause mistake analysis and remedial steps.
4. `[Mode: companion]`: Empathetic, grounded emotional support with crisis safety (isolated from academic records).

**Hardware:** Free Colab Tesla T4 (16GB VRAM)  
**Method:** Unsloth 4-bit QLoRA ($r=16, \\alpha=16$, max_seq_length=1024)  
**Export:** GGUF `q4_k_m` for $0 CPU deployment on Hugging Face Spaces or local Ollama."""),

        make_cell("code", """# 1. Install Unsloth and modern fine-tuning libraries
%%capture
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install datasets huggingface_hub"""),

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
dtype = None # Auto-detects float16 for T4, bfloat16 for Ampere
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply PEFT / QLoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0, # Optimized = 0 for Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=max_seq_length,
)
print("✅ QLoRA model initialized with 16-rank adapters.")"""),

        make_cell("code", """# 5. Format ChatML Conversation Templates
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
print("✅ Formatted sample ChatML text preview:")
print(formatted_dataset["train"][0]["text"][:350])"""),

        make_cell("code", """# 6. Baseline (Before Training) Qualitative Evaluation
FastLanguageModel.for_inference(model)

test_eval_prompts = [
    {
        "mode": "tutor",
        "system": "You are StudyVerse AI Tutor. [Mode: tutor]",
        "user": "Explain Newton's Second Law and derive F = ma for constant mass."
    },
    {
        "mode": "quiz",
        "system": "You are StudyVerse Quiz Generator. [Mode: quiz] Output strict JSON with stem, options, correct_answer, explanation, difficulty, topic.",
        "user": "Generate a medium difficulty MCQ on Coulomb's law."
    },
    {
        "mode": "diagnostic",
        "system": "You are StudyVerse Mistake Journal Diagnostic Engine. [Mode: diagnostic]",
        "user": "I answered F/4 instead of 3F/8 on two charged spheres touched successively by sphere C. Diagnose my error."
    },
    {
        "mode": "companion",
        "system": "You are StudyVerse Calm Companion. [Mode: companion]",
        "user": "I have my physics exam tomorrow and I haven't finished optics. I'm panicking."
    }
]

print("=== BASELINE RESPONSES (BEFORE TRAINING) ===")
for test_item in test_eval_prompts:
    prompt_msgs = [
        {"role": "system", "content": test_item["system"]},
        {"role": "user", "content": test_item["user"]}
    ]
    inputs = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=180, temperature=0.15, use_cache=True)
    decoded = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\\n--- [{test_item['mode'].upper()}] ---")
    print(decoded.strip())"""),

        make_cell("code", """# 7. Configure SFTConfig and Train with Google Drive Checkpointing
import glob
from trl import SFTTrainer, SFTConfig

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
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["validation"],
    args=training_args,
)

# Auto-resume from Google Drive if a checkpoint exists
existing_checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*")
latest_checkpoint = max(existing_checkpoints, key=os.path.getctime) if existing_checkpoints else None

if latest_checkpoint:
    print(f"🔁 Resuming training from Drive checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("🚀 Starting fine-tuning training...")
    trainer.train()"""),

        make_cell("code", """# 8. Post-Training Qualitative Evaluation (Verify 4 Modes)
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

        make_cell("code", """# 9. Push GGUF (q4_k_m) & Model Card to Hugging Face Hub
from huggingface_hub import login, HfApi

# Step 1: Log in with your Hugging Face write token
# HF_TOKEN = "hf_your_token_here"
# login(token=HF_TOKEN)

HF_REPO_ID = "Joshua8976/studyverse-qwen2.5-3b-gguf" # Change to your repo name

print("Ready to export GGUF q4_k_m. Uncomment the lines below when token is configured:")
# model.push_to_hub_gguf(
#     HF_REPO_ID,
#     tokenizer,
#     quantization_method="q4_k_m",
#     token=HF_TOKEN,
# )

# Step 2: Generate and upload Model Card
model_card = f\"\"\"---
language:
- en
license: apache-2.0
tags:
- studyverse
- qwen2.5
- qlora
- unsloth
- gguf
- education
- stem
base_model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
pipeline_tag: text-generation
---

# StudyVerse Qwen 2.5 3B Instruct (GGUF q4_k_m)

Fine-tuned on StudyVerse's academic curriculum, formulas, common exam mistakes, and student wellbeing support.

### Supported Modes:
1. `[Mode: tutor]`: Concise, syllabus-grounded academic explanations with KaTeX formatting.
2. `[Mode: quiz]`: Strict JSON practice questions formatted for instant interactive quizzes.
3. `[Mode: diagnostic]`: Root cause mistake analysis and remedial action steps.
4. `[Mode: companion]`: Empathetic, grounded emotional support with zero toxic positivity and 24/7 crisis helplines.

### Deployment:
Deployable on free Hugging Face Spaces (CPU Basic) with `llama-cpp-python` or locally with Ollama.
\"\"\"

with open("README.md", "w", encoding="utf-8") as f:
    f.write(model_card)

print("✅ Model card written to README.md.")"""),
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

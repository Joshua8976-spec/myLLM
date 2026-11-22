"""
StudyVerse $0 Qwen 2.5 3B Instruct Fine-Tuning Pipeline on Modal (modal.com).

Executes full 4-bit QLoRA fine-tuning headless on Modal GPU (T4 / A10G)
with persistent checkpoints volume, before/after evaluations, and GGUF export.

Usage:
  modal run ml-pipeline/train_modal.py
"""

import os
import glob
from pathlib import Path
import modal

app = modal.App("studyverse-qwen-train")

volume = modal.Volume.from_name("studyverse-checkpoints", create_if_missing=True)

ML_PIPELINE_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .pip_install(
        "torch==2.4.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.44.0,<4.48.0",
        "datasets",
        "trl<0.13.0",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "hf_transfer",
        "pillow",
        "sentencepiece",
        "protobuf",
        "einops",
        "wheel",
        "packaging",
    )
    .run_commands(
        'pip install --no-deps "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"',
        'pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git"',
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ML_PIPELINE_DIR), remote_path="/root/ml-pipeline")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 2,
    volumes={"/root/checkpoints": volume},
)
def train():
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig

    checkpoint_dir = "/root/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✅ Checkpoint directory ready: {checkpoint_dir}")

    # 1. Load dataset
    train_path = "/root/ml-pipeline/train.jsonl"
    val_path = "/root/ml-pipeline/val.jsonl"
    assert os.path.exists(train_path), f"Missing dataset: {train_path}"
    assert os.path.exists(val_path), f"Missing dataset: {val_path}"

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )
    print(f"✅ Loaded {len(dataset['train'])} train and {len(dataset['validation'])} val samples.")

    # 2. Load model & tokenizer with 4-bit Unsloth
    max_seq_length = 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
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

    # 3. Format ChatML
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in convos
        ]
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Baseline evaluation before training
    FastLanguageModel.for_inference(model)
    print("\n=== BASELINE EVALUATION (BEFORE TRAINING) ===")
    test_eval_prompts = [
        ("tutor", "Explain Newton's Second Law and derive F = ma for constant mass."),
        ("quiz", "Generate a medium difficulty MCQ on Coulomb's law."),
        ("diagnostic", "Diagnose: I answered F/4 instead of 3F/8 on two touched spheres."),
        ("companion", "I have my physics exam tomorrow and I'm panicking."),
    ]
    for mode_name, user_query in test_eval_prompts:
        msgs = [
            {"role": "system", "content": f"You are StudyVerse AI. [Mode: {mode_name}]"},
            {"role": "user", "content": user_query},
        ]
        inp = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        out = model.generate(input_ids=inp, max_new_tokens=140, temperature=0.15, use_cache=True)
        print(f"--- [{mode_name.upper()}] ---")
        print(tokenizer.decode(out[0][inp.shape[1] :], skip_special_tokens=True).strip()[:200])

    # 5. Training configuration
    training_args = SFTConfig(
        output_dir=checkpoint_dir,
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

    # Auto-resume from volume checkpoint if available
    existing_checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint-*")
    latest_checkpoint = max(existing_checkpoints, key=os.path.getctime) if existing_checkpoints else None
    if latest_checkpoint:
        print(f"🔁 Resuming from: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("🚀 Starting fine-tuning...")
        trainer.train()

    # 6. Post-training evaluation
    FastLanguageModel.for_inference(model)
    print("\n=== POST-TRAINING EVALUATION (AFTER FINE-TUNING) ===")
    for mode_name, user_query in test_eval_prompts:
        msgs = [
            {"role": "system", "content": f"You are StudyVerse AI. [Mode: {mode_name}]"},
            {"role": "user", "content": user_query},
        ]
        inp = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        temp = 0.45 if mode_name == "companion" else 0.15
        out = model.generate(input_ids=inp, max_new_tokens=180, temperature=temp, use_cache=True)
        print(f"--- [{mode_name.upper()}] ---")
        print(tokenizer.decode(out[0][inp.shape[1] :], skip_special_tokens=True).strip()[:200])

    # 7. Save adapter and export GGUF
    adapter_dir = f"{checkpoint_dir}/studyverse-qwen-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"✅ LoRA adapter saved to: {adapter_dir}")

    gguf_output_dir = f"{checkpoint_dir}/gguf_q4_k_m"
    os.makedirs(gguf_output_dir, exist_ok=True)
    try:
        model.save_pretrained_gguf(
            gguf_output_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"✅ GGUF exported to: {gguf_output_dir}")
    except Exception as e:
        print(f"ℹ️ GGUF direct export notice: {e}")

    # Commit persistent volume
    volume.commit()
    print("✅ Volume committed! Checkpoints and models safely stored.")


@app.local_entrypoint()
def main():
    print("🌟 Submitting StudyVerse Qwen 2.5 3B fine-tuning job to Modal GPU...")
    train.remote()
    print("🎉 Training completed successfully on Modal!")

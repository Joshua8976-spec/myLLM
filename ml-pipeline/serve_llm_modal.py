"""
OpenAI-Compatible Inference Server for Fine-Tuned StudyVerse Qwen 2.5 3B on Modal (modal.com).
Serves GET /v1/models and POST /v1/chat/completions so StudyVerse can plug it directly
into VITE_STUDY_LLM_URL.

Deploy:
  modal deploy ml-pipeline/serve_llm_modal.py

Connect to StudyVerse:
  Set VITE_STUDY_LLM_URL=https://<your-org>--studyverse-qwen-serve-modelserver-chat-completions.modal.run/v1
"""

import os
from typing import List, Optional
import modal
from pydantic import BaseModel

app = modal.App("studyverse-qwen-serve")
volume = modal.Volume.from_name("studyverse-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi~=0.111.0",
        "uvicorn~=0.30.0",
        "torch>=2.2.0",
        "transformers>=4.44.0",
        "accelerate",
        "bitsandbytes",
        "peft",
    )
)

MODEL_ID = "studyverse-qwen-2.5-3b"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.15
    max_tokens: Optional[int] = 260
    stream: Optional[bool] = False


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={"/root/checkpoints": volume},
)
class ModelServer:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model_name = "Qwen/Qwen2.5-3B-Instruct"
        print("Loading Qwen 2.5 3B tokenizer and model on GPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        adapter_dir = "/root/checkpoints/studyverse-qwen-adapter"
        if os.path.exists(adapter_dir):
            print(f"Loading fine-tuned StudyVerse adapter from {adapter_dir}...")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, adapter_dir)
        else:
            print("Loading base Qwen 2.5 3B Instruct model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        self.model.eval()
        print("✅ Model loaded and ready for inference!")

    @modal.fastapi_endpoint(method="GET", path="/v1/models")
    def list_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "owned_by": "studyverse",
                }
            ],
        }

    @modal.fastapi_endpoint(method="POST", path="/v1/chat/completions")
    def chat_completions(self, request: ChatCompletionRequest):
        import torch

        # Format ChatML messages using the model's tokenizer
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)
        temp = request.temperature if request.temperature and request.temperature > 0 else 0.15
        max_tokens = min(request.max_tokens or 260, 512)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0.05,
                use_cache=True,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "id": "chatcmpl-studyverse",
            "object": "chat.completion",
            "model": request.model or MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

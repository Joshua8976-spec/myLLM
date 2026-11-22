"""
OpenAI-Compatible Inference Server for Fine-Tuned StudyVerse Qwen 2.5 3B on Modal (modal.com).
Serves GET /health, GET /v1/models, and POST /v1/chat/completions (with full streaming SSE support)
so frontend web applications can plug it directly into VITE_STUDY_LLM_URL.

Deploy:
  modal deploy ml-pipeline/serve_llm_modal.py

Connect to StudyVerse:
  Set VITE_STUDY_LLM_URL=https://<your-org>--studyverse-qwen-serve-modelserver-serve.modal.run/v1
"""

import os
import json
import time
from typing import List, Optional
import modal
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

    @modal.asgi_app()
    def serve(self):
        web_app = FastAPI(title="StudyVerse Qwen Inference Server")

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/health")
        def health():
            return {"status": "ok", "model": MODEL_ID}

        @web_app.get("/v1/models")
        def list_models():
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

        @web_app.post("/v1/chat/completions")
        def chat_completions(request: ChatCompletionRequest):
            import torch
            from transformers import TextIteratorStreamer
            from threading import Thread

            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)
            temp = request.temperature if request.temperature and request.temperature > 0 else 0.15
            max_tokens = min(request.max_tokens or 260, 1024)

            # Streaming SSE response
            if request.stream:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=temp > 0.05,
                    use_cache=True,
                )
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                def event_generator():
                    completion_id = f"chatcmpl-{int(time.time() * 1000)}"
                    created_time = int(time.time())
                    for text_chunk in streamer:
                        if text_chunk:
                            data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": request.model or MODEL_ID,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text_chunk},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                    finish_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model or MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(finish_data)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_generator(), media_type="text/event-stream")

            # Non-streaming response
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
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
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

        return web_app

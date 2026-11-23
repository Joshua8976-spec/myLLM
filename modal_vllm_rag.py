"""
modal_vllm_rag.py - Tuned Modal vLLM + RAG Academic Tutor Service

Exposes:
  1. POST /rag/chat - Metadata-scoped retrieval, formula grounding, deterministic Qwen 2.5 3B prompting,
                      citation sanitization (stripping hallucinated tags & tracking verified IDs),
                      and 'Quick Check:' active recall.
  2. POST /v1/chat/completions - OpenAI-compatible chat completion (with full streaming SSE support).
  3. GET /v1/models - OpenAI-compatible model catalog.
  4. GET /health - Server and corpus health check.

Performance & Grounding:
  - vLLM Prefix Caching enabled (--enable-prefix-caching)
  - max-model-len capped cleanly at 4096
  - Deterministic sampling: temperature=0.15, top_p=0.9, repetition_penalty=1.05, max_tokens=650

Deploy:
  modal deploy modal_vllm_rag.py

StudyVerse Frontend Configuration:
  VITE_MODAL_RAG_URL=https://<your-org>--studyverse-rag-vllm-serve.modal.run
"""

import os
import re
import sys
import json
import time
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import modal
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Modal App Configuration
APP_NAME = "studyverse-rag-vllm"
MODEL_ID = "studyverse-qwen-2.5-3b"
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("studyverse-checkpoints", create_if_missing=True)

# Container Image with vLLM, sentence-transformers, FastAPI, and HuggingFace toolkits
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.3",
        "fastapi~=0.111.0",
        "uvicorn~=0.30.0",
        "sentence-transformers>=3.0.0",
        "httpx>=0.27.0",
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "numpy",
        "pydantic>=2.0.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_CACHE_ROOT": "/root/checkpoints/vllm_cache",
        "HF_HOME": "/root/checkpoints/hf_cache",
        "RAG_CACHE_DIR": "/root/checkpoints/rag_cache",
    })
    .add_local_file("data/rag-corpus.json", remote_path="/root/data/rag-corpus.json")
    .add_local_file("rag_retriever.py", remote_path="/root/rag_retriever.py")
)


# Pydantic Schemas
class ProfileInput(BaseModel):
    level: Optional[str] = None
    stream: Optional[str] = None
    targetExams: Optional[List[str]] = None
    enrolledSubjects: Optional[List[str]] = None


class RagChatRequest(BaseModel):
    question: str
    profile: Optional[ProfileInput] = None
    subjectFilter: Optional[str] = None
    subject_filter: Optional[str] = None
    topK: Optional[int] = None
    top_k: Optional[int] = None
    minScore: Optional[float] = None
    min_score: Optional[float] = None


class RagChatResponse(BaseModel):
    answer: str
    citations: List[str]
    retrievedChunks: List[Dict[str, Any]]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.15
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.05
    max_tokens: Optional[int] = 650
    stream: Optional[bool] = False


# Prompt Engineering & Grounding Helpers
def build_grounded_tutor_messages(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Constructs deterministic, strictly grounded tutor prompt for Qwen 2.5 3B.
    Formats context cleanly with ID, Subject / Topic, Title, Formula/Rule, Key Terms, and Explanation.
    Instructs the model to cite [exact-id] inline, refuse when not covered, and end with 'Quick Check:'.
    """
    context_lines = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        chunk_id = chunk.get("id", f"chunk_{idx}")
        title = chunk.get("title", "")
        subject = chunk.get("subject", "")
        stream = chunk.get("stream", "")
        level = chunk.get("level", "")
        topic = chunk.get("topic", title)
        formula = chunk.get("formulaOrRule", "")
        explanation = chunk.get("derivationOrExplanation", "")
        variables = chunk.get("variablesOrTerms", {})

        vars_str = "; ".join([f"{k} = {v}" for k, v in variables.items()]) if variables else "None specified"

        entry_text = (
            f"[Context Entry: {chunk_id}]\n"
            f"Subject / Topic: {subject} ({stream} - {level}) / {topic}\n"
            f"Title: {title}\n"
            f"Formula/Rule: {formula}\n"
            f"Key Terms: {vars_str}\n"
            f"Explanation: {explanation}\n"
        )
        context_lines.append(entry_text)

    context_block = "\n".join(context_lines)

    system_prompt = (
        "You are the StudyVerse Academic AI Tutor, a concise and mathematically precise tutor.\n\n"
        "STRICT GROUNDING & CITATION RULES:\n"
        "1. GROUNDING: Use ONLY the verified Context Entries provided below for academic formulas, definitions, laws, and rules. Do not extrapolate or guess unverified equations.\n"
        "2. INLINE CITATIONS: Every time you state or apply a formula, theorem, or syllabus rule, immediately append its exact citation tag in brackets, e.g. [chunk-id]. Cite ONLY IDs present in the retrieved list.\n"
        "3. REFUSAL WHEN MISSING: If the retrieved Context Entries do not contain the formula or rule needed to answer the question, state clearly: 'The retrieved curriculum context does not contain the verified formula or rule for this topic.' Do not invent hypothetical equations.\n"
        "4. ACTIVE RECALL: Conclude every valid academic explanation with a single line labeled 'Quick Check:' containing one short conceptual active-recall question.\n"
        "5. STYLE: Deterministic, structured, direct, and concise."
    )

    if context_block.strip():
        user_prompt = (
            f"VERIFIED CURRICULUM CONTEXT ENTRIES:\n"
            f"{context_block}\n"
            f"STUDENT QUESTION:\n"
            f"{question}\n\n"
            f"Provide a clear, structured explanation adhering strictly to the citation rules above, citing [id] after each formula/rule, and ending with 'Quick Check: <question>'."
        )
    else:
        user_prompt = (
            f"VERIFIED CURRICULUM CONTEXT ENTRIES:\n"
            f"(No relevant curriculum entries met the relevance threshold for this query)\n\n"
            f"STUDENT QUESTION:\n"
            f"{question}\n\n"
            f"State clearly that the retrieved curriculum context does not cover this request, and do not provide unverified formulas."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def sanitize_citations(
    answer_text: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Sanitizes citation tags in the answer text:
    1. Extracts all [bracketed_ids] cited by the model.
    2. Strips out any hallucinated [id] not present in retrieved_chunks.
    3. Detects if retrieved chunks were used without brackets (matching key formulas or exact titles)
       and ensures verified citations are tracked and appended if omitted.
    Returns (cleaned_answer_text, verified_citations_list).
    """
    if not answer_text:
        return "", []

    retrieved_map = {c["id"]: c for c in retrieved_chunks if c.get("id")}
    retrieved_ids = set(retrieved_map.keys())

    # Find all bracketed tags in the answer text
    bracketed_tags = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", answer_text)

    cleaned_text = answer_text
    verified_citations: List[str] = []
    seen = set()

    for tag in bracketed_tags:
        if tag in retrieved_ids:
            if tag not in seen:
                verified_citations.append(tag)
                seen.add(tag)
        else:
            # Strip hallucinated bracketed tag like [hallucinated-rule-999]
            cleaned_text = re.sub(rf"\[{re.escape(tag)}\]", "", cleaned_text)

    # Secondary check: If the model stated a formula or exact title from a retrieved chunk
    # without its bracketed tag, recognize and verify it
    for c_id, chunk in retrieved_map.items():
        if c_id not in seen:
            title = (chunk.get("title") or "").strip()
            formula = (chunk.get("formulaOrRule") or "").strip()
            # Clean formula for matching (e.g. strip $$, \text, spaces)
            clean_formula = re.sub(r"[\$\\ ]", "", formula)
            clean_answer = re.sub(r"[\$\\ ]", "", cleaned_text)

            matched = False
            if len(clean_formula) > 8 and clean_formula.lower() in clean_answer.lower():
                matched = True
            elif len(title) > 6 and title.lower() in cleaned_text.lower():
                matched = True

            if matched:
                verified_citations.append(c_id)
                seen.add(c_id)

    # Clean up double spaces or floating punctuation left by stripped tags
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r" +([.,;?!])", r"\1", cleaned_text).strip()

    return cleaned_text, verified_citations


def extract_citations(answer_text: str, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Convenience wrapper that returns verified citation IDs using sanitize_citations.
    """
    _, citations = sanitize_citations(answer_text, retrieved_chunks)
    return citations


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={"/root/checkpoints": volume},
)
class RagVllmService:
    @modal.enter()
    def setup(self):
        """Initializes RAG Retriever and starts local vLLM OpenAI-compatible server on GPU."""
        import httpx

        # 1. Initialize RAG Retriever
        print("🔍 Initializing RAG Retriever...")
        from rag_retriever import RAGRetriever

        corpus_file = "/root/data/rag-corpus.json"
        if not os.path.exists(corpus_file):
            corpus_file = "data/rag-corpus.json" if os.path.exists("data/rag-corpus.json") else "rag-corpus.json"

        cache_dir = "/root/checkpoints/rag_cache" if os.path.exists("/root/checkpoints") else "./data/cache"
        self.retriever = RAGRetriever(corpus_path=corpus_file, cache_dir=cache_dir)
        print(f"✅ RAG Retriever initialized with {len(self.retriever.corpus)} chunks.")

        # 2. Start local vLLM server process with prefix caching & 4096 max-model-len
        adapter_path = "/root/checkpoints/studyverse-qwen-adapter"
        has_adapter = os.path.exists(adapter_path)
        print(f"📦 Checking for fine-tuned LoRA adapter: {adapter_path} (exists={has_adapter})")

        vllm_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", BASE_MODEL_NAME,
            "--host", "127.0.0.1",
            "--port", "8000",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "4096",
            "--enable-prefix-caching",
            "--trust-remote-code",
            "--served-model-name", MODEL_ID,
        ]

        if has_adapter:
            vllm_cmd.extend([
                "--enable-lora",
                "--lora-modules", f"{MODEL_ID}={adapter_path}",
            ])

        print(f"🚀 Spawning vLLM OpenAI server: {' '.join(vllm_cmd)}")
        self.vllm_proc = subprocess.Popen(vllm_cmd)

        # Wait for vLLM to become healthy
        print("⏳ Waiting for vLLM server to be ready on http://127.0.0.1:8000/health...")
        healthy = False
        start_time = time.time()
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=5.0) as client:
            while time.time() - start_time < 90:
                if self.vllm_proc.poll() is not None:
                    print(f"❌ vLLM process exited unexpectedly with code {self.vllm_proc.returncode}")
                    break
                try:
                    res = client.get("/health")
                    if res.status_code == 200:
                        healthy = True
                        print(f"✅ Local vLLM server is online and healthy in {time.time() - start_time:.1f}s!")
                        break
                except Exception:
                    time.sleep(1.5)

        if not healthy:
            print("⚠️ vLLM server failed to start within timeout; falling back to direct HuggingFace inference.")
            self._init_hf_fallback()
        else:
            self.hf_model = None

    def _init_hf_fallback(self):
        """Initializes fallback HuggingFace model in case vLLM is not active."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("📦 Loading fallback HuggingFace model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        adapter_path = "/root/checkpoints/studyverse-qwen-adapter"
        if os.path.exists(adapter_path):
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.hf_model = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        self.hf_model.eval()
        print("✅ Fallback HuggingFace model loaded.")

    @modal.exit()
    def terminate(self):
        """Clean shutdown of local vLLM background process."""
        if hasattr(self, "vllm_proc") and self.vllm_proc and self.vllm_proc.poll() is None:
            print("🛑 Terminating vLLM server subprocess...")
            self.vllm_proc.terminate()
            try:
                self.vllm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.vllm_proc.kill()

    @modal.asgi_app()
    def serve(self):
        """Exposes FastAPI ASGI app on Modal."""
        import httpx

        web_app = FastAPI(
            title="StudyVerse Modal RAG + vLLM Academic Tutor Service",
            version="1.1.0",
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "StudyVerse RAG + vLLM Tutor",
                "model": MODEL_ID,
                "corpus_chunks": len(self.retriever.corpus),
                "corpus_hash": self.retriever.corpus_hash[:12],
                "prefix_caching": True,
            }

        @web_app.get("/v1/models")
        async def list_models():
            try:
                async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=5.0) as client:
                    resp = await client.get("/v1/models")
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass

            return {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "studyverse",
                    }
                ],
            }

        @web_app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            is_stream = body.get("stream", False)

            try:
                client = httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=120.0)
                if is_stream:
                    async def stream_generator():
                        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                        await client.aclose()

                    return StreamingResponse(stream_generator(), media_type="text/event-stream")
                else:
                    resp = await client.post("/v1/chat/completions", json=body)
                    await client.aclose()
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
            except Exception as e:
                if getattr(self, "hf_model", None) is not None:
                    return self._generate_hf_completion(body)
                raise HTTPException(status_code=503, detail=f"Inference server unavailable: {str(e)}")

        @web_app.post("/rag/chat", response_model=RagChatResponse)
        async def rag_chat(req: RagChatRequest):
            """
            High-Precision Retrieval-Augmented Academic Tutor Endpoint:
            1. Scoped retrieval with student profile, subject filters, and min_score threshold
            2. Grounded deterministic prompt creation for Qwen 2.5 3B
            3. Local vLLM completion call (temp=0.15, top_p=0.9, repetition_penalty=1.05)
            4. Post-generation citation sanitizer (stripping hallucinated tags, tracking verified IDs)
            """
            subject_filter = req.subjectFilter or req.subject_filter
            top_k = req.topK or req.top_k or 4
            min_score = req.minScore if req.minScore is not None else (req.min_score if req.min_score is not None else 0.35)

            profile_dict = None
            if req.profile:
                profile_dict = {
                    "level": req.profile.level,
                    "stream": req.profile.stream,
                    "targetExams": req.profile.targetExams,
                    "enrolledSubjects": req.profile.enrolledSubjects,
                }

            # 1. Scoped retrieval with prefix instruction, hybrid ranking & deduplication
            retrieved_chunks = self.retriever.retrieve(
                query=req.question,
                profile=profile_dict,
                subject_filter=subject_filter,
                top_k=top_k,
                min_score=min_score,
            )

            # 2. Build deterministic grounded prompt
            messages = build_grounded_tutor_messages(req.question, retrieved_chunks)

            # 3. Call local vLLM completions endpoint with deterministic parameters
            vllm_payload = {
                "model": MODEL_ID,
                "messages": messages,
                "temperature": 0.15,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "max_tokens": 650,
                "stream": False,
            }

            answer_text = ""
            try:
                async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=60.0) as client:
                    resp = await client.post("/v1/chat/completions", json=vllm_payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer_text = data["choices"][0]["message"]["content"]
                    else:
                        raise RuntimeError(f"vLLM call returned status {resp.status_code}: {resp.text}")
            except Exception as vllm_err:
                print(f"⚠️ Local vLLM endpoint call error: {vllm_err}. Using fallback...")
                if getattr(self, "hf_model", None) is not None:
                    answer_text = self._generate_hf_text(messages, temperature=0.15, max_tokens=650)
                else:
                    raise HTTPException(status_code=502, detail=f"vLLM generation error: {str(vllm_err)}")

            # 4. Post-generation citation sanitizer
            cleaned_answer, citations = sanitize_citations(answer_text, retrieved_chunks)

            return RagChatResponse(
                answer=cleaned_answer.strip(),
                citations=citations,
                retrievedChunks=retrieved_chunks,
            )

        return web_app

    def _generate_hf_text(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Helper to generate text using fallback HuggingFace model."""
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)

        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_hf_completion(self, body: Dict[str, Any]):
        """Helper for non-RAG OpenAI chat completions using fallback HuggingFace model."""
        messages = body.get("messages", [])
        temp = body.get("temperature", 0.15)
        max_tokens = body.get("max_tokens", 650)
        text = self._generate_hf_text(messages, temp, max_tokens)

        return JSONResponse(
            content={
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

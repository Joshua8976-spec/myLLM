"""
modal_vllm_rag.py - Fast & High-Throughput Modal vLLM + RAG Academic Tutor Service

Optimized for:
  1. Ultra-Fast TTFT (Time-to-First-Token):
     - vLLM Prefix Caching (--enable-prefix-caching)
     - vLLM Chunked Prefill (--enable-chunked-prefill)
     - Max sequence concurrency (--max-num-seqs 32)
     - GPU memory utilization tuned to 0.90 for Qwen 2.5 3B
     - Compact prompt prefill using trimmed high-signal context entries (~120-160 words)
  2. Container Pre-Warming:
     - SentenceTransformer & vLLM CUDA graph/tokenizer pre-warmed on container startup (@modal.enter())
     - 5-minute warm keep-alive (scaledown_window=300)
  3. Real-Time Streaming & Full JSON Compatibility:
     - POST /rag/chat supports both stream=True (SSE tokens + citations) and stream=False (instant JSON)
  4. Deterministic Grounding & Citation Sanitizer:
     - Exact [id] inline tagging, hallucination stripping, and unbracketed formula recovery
     - Single labeled 'Quick Check:' active-recall question

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
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

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
        "transformers>=4.45.0,<4.49.0",
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
    stream: Optional[bool] = False
    mode: Optional[str] = None


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
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False


# Prompt Engineering & Grounding Helpers
def build_grounded_tutor_messages(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    mode: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Constructs deterministic, strictly grounded tutor prompt for Qwen 2.5 3B.
    Supports dual-mode contract:
      - [MODE: CALM_CORNER]: Warm, deeply empathetic, zero productivity guilt, gentle grounding.
      - [MODE: STRICT_COACH] (default): Firm, accountable, structured time blocks, exact formulas.
    Formats context cleanly with concise, high-signal fields and pre-trimmed explanations
    (~120-160 words) to minimize prompt prefill tokens and TTFT.
    """
    is_calm_corner = (
        (mode and mode.lower() in ("calm_corner", "calm", "companion"))
        or "[MODE: CALM_CORNER]" in question
    )

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

    if is_calm_corner:
        system_prompt = (
            "You are StudyVerse Calm Corner Companion. [MODE: CALM_CORNER]\n\n"
            "GENTLE GUIDELINES:\n"
            "1. EMPATHETIC VALIDATION: Warmly and deeply validate feelings first with patience and non-judgmental acceptance. Never use toxic positivity or dismiss their struggle.\n"
            "2. ZERO PRODUCTIVITY GUILT: Avoid lecturing, pushing study schedules, guilt-tripping about deadlines, or urging immediate work. Slow the pace down.\n"
            "3. GENTLE GROUNDING: Offer optional calming grounding or breathing resets (e.g. 4-4-4 box breathing, unclenching jaw, sipping water, pausing).\n"
            "4. ETHICAL SAFETY: Provide a soothing space without giving clinical or medical diagnoses.\n"
            "5. TONE: Kind, soft, low-pressure, and understanding."
        )
    else:
        system_prompt = (
            "You are StudyVerse Academic AI Coach & Rigorous Tutor. [MODE: STRICT_COACH]\n\n"
            "ACADEMIC INSTRUCTION & GROUNDING GUIDELINES:\n"
            "1. CURRICULUM GROUNDING: Ground your answers firmly in verified academic syllabus standards. If Context Entries are provided below, prioritize their formulas, rules, and definitions.\n"
            "2. SEAMLESS CURRICULUM SYNTHESIS: When retrieved Context Entries do not fully cover the query or are sparse, seamlessly combine verified academic curriculum principles with general academic knowledge to provide an insightful, complete explanation. Never emit rigid refusal boilerplate or decline academic questions due to sparse catalog entries.\n"
            "3. AGE-APPROPRIATE MIDDLE & SECONDARY SCHOOL (CLASSES 6-10): For students in Classes 6, 7, 8, 9, or 10, explain concepts using age-appropriate, crystal-clear language, real-world intuitive analogies, step-by-step worked examples, and simple equations. Avoid collegiate jargon while maintaining complete syllabus accuracy.\n"
            "4. COMMERCE, ACCOUNTANCY & SOCIAL SCIENCES: For Commerce, Accountancy, Economics, or Business Studies questions, provide clear conceptual explanations, structured comparisons, step-by-step calculation methods or journal entries where applicable, and practical real-world scenarios.\n"
            "5. CITATIONS: If drawing directly from a retrieved Context Entry, cite its id in brackets, e.g. [chunk-id]. Cite only valid retrieved IDs. Do NOT invent IDs or append an artificial Sources list.\n"
            "6. ACTIVE RECALL: Conclude every valid academic explanation with a single line labeled 'Quick Check:' containing one short conceptual active-recall question.\n"
            "7. TONE: Firm, structured, clear, and encouraging."
        )

    if context_block.strip():
        user_prompt = (
            f"VERIFIED CURRICULUM CONTEXT ENTRIES:\n"
            f"{context_block}\n\n"
            f"STUDENT QUESTION:\n"
            f"{question}\n\n"
            f"Provide a clear, structured explanation adhering strictly to your tutor guidelines."
        )
    else:
        user_prompt = (
            f"CURRICULUM CONTEXT:\n"
            f"(No specific formula sheet chunk retrieved for this query - synthesize verified academic principles directly)\n\n"
            f"STUDENT QUESTION:\n"
            f"{question}\n\n"
            f"Provide a comprehensive, accurate, step-by-step academic explanation answering the question fully."
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
            cleaned_text = re.sub(rf"\[{re.escape(tag)}\]", "", cleaned_text)

    # Recognize unbracketed formula or exact title mentions
    for c_id, chunk in retrieved_map.items():
        if c_id not in seen:
            title = (chunk.get("title") or "").strip()
            formula = (chunk.get("formulaOrRule") or "").strip()
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

    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r" +([.,;?!])", r"\1", cleaned_text).strip()

    return cleaned_text, verified_citations


def extract_citations(answer_text: str, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
    """Convenience wrapper that returns verified citation IDs."""
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

        # 1. Initialize RAG Retriever with pre-indexed tokens
        print("🔍 Initializing RAG Retriever...")
        from rag_retriever import RAGRetriever

        corpus_file = "/root/data/rag-corpus.json"
        if not os.path.exists(corpus_file):
            corpus_file = "data/rag-corpus.json" if os.path.exists("data/rag-corpus.json") else "rag-corpus.json"

        cache_dir = "/root/checkpoints/rag_cache" if os.path.exists("/root/checkpoints") else "./data/cache"
        self.retriever = RAGRetriever(corpus_path=corpus_file, cache_dir=cache_dir)
        print("🔥 Pre-warming retriever embedding pipeline...")
        self.retriever.warmup()
        print(f"✅ RAG Retriever warm and ready with {len(self.retriever.corpus)} chunks.")

        # 2. Start local vLLM server process with Prefix Caching & Chunked Prefill
        adapter_path = "/root/checkpoints/studyverse-qwen-adapter"
        has_adapter = os.path.exists(adapter_path)
        print(f"📦 Checking for fine-tuned LoRA adapter: {adapter_path} (exists={has_adapter})")

        vllm_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", BASE_MODEL_NAME,
            "--host", "127.0.0.1",
            "--port", "8000",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096",
            "--max-num-seqs", "32",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--dtype", "half",
            "--trust-remote-code",
            "--served-model-name", MODEL_ID,
        ]

        if has_adapter:
            vllm_cmd.extend([
                "--enable-lora",
                "--lora-modules", f"{MODEL_ID}={adapter_path}",
            ])

        print(f"🚀 Spawning high-throughput vLLM OpenAI server: {' '.join(vllm_cmd)}")
        self.vllm_proc = subprocess.Popen(vllm_cmd)

        # Wait for vLLM to become healthy and send a warmup completion request
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
                        # Warm-up request to pre-compile CUDA graphs and initialize KV cache
                        try:
                            client.post(
                                "/v1/chat/completions",
                                json={
                                    "model": MODEL_ID,
                                    "messages": [{"role": "user", "content": "Hello"}],
                                    "max_tokens": 8,
                                },
                                timeout=10.0,
                            )
                            print("🔥 vLLM CUDA graph and tokenization pre-warmed successfully.")
                        except Exception as warm_err:
                            print(f"⚠️ Note on vLLM warm-up completion: {warm_err}")
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

    @modal.asgi_app(label="studyverse-rag-vllm-serve")
    def serve(self):
        """Exposes FastAPI ASGI app on Modal."""
        import httpx

        web_app = FastAPI(
            title="StudyVerse Fast Modal RAG + vLLM Academic Tutor Service",
            version="1.2.0",
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "StudyVerse Fast RAG + vLLM Tutor",
                "model": MODEL_ID,
                "corpus_chunks": len(self.retriever.corpus),
                "corpus_hash": self.retriever.corpus_hash[:12],
                "prefix_caching": True,
                "chunked_prefill": True,
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

        @web_app.post("/rag/chat")
        async def rag_chat(req: RagChatRequest):
            """
            High-Speed Retrieval-Augmented Academic Tutor Endpoint:
            1. Scoped sub-15ms retrieval with pre-indexed tokens & trimmed explanations
            2. Grounded prompt with short prefill to minimize TTFT
            3. Local vLLM call (max_tokens=500, temp=0.15, prefix-caching enabled)
            4. Citation sanitizer and structured response (or SSE streaming if stream=True)
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

            # 1. Fast scoped retrieval (< 15ms)
            retrieved_chunks = self.retriever.retrieve(
                query=req.question,
                profile=profile_dict,
                subject_filter=subject_filter,
                top_k=top_k,
                min_score=min_score,
            )

            # 2. Build lean grounded prompt
            messages = build_grounded_tutor_messages(req.question, retrieved_chunks, mode=req.mode)
            is_calm = (req.mode and req.mode.lower() in ("calm_corner", "calm", "companion")) or "[MODE: CALM_CORNER]" in req.question
            gen_temp = 0.45 if is_calm else 0.15

            # 3. Handle Streaming if requested by client
            if req.stream:
                async def stream_rag_tokens() -> AsyncGenerator[str, None]:
                    accumulated_text = ""
                    client = httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=120.0)
                    try:
                        vllm_payload = {
                            "model": MODEL_ID,
                            "messages": messages,
                            "temperature": gen_temp,
                            "top_p": 0.9,
                            "repetition_penalty": 1.05,
                            "max_tokens": 500,
                            "stream": True,
                        }
                        async with client.stream("POST", "/v1/chat/completions", json=vllm_payload) as resp:
                            async for raw_line in resp.aiter_lines():
                                if not raw_line or not raw_line.startswith("data: "):
                                    continue
                                data_str = raw_line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_str)
                                    delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                                    content_chunk = delta.get("content", "")
                                    if content_chunk:
                                        accumulated_text += content_chunk
                                        yield f"data: {json.dumps({'text': content_chunk, 'done': False})}\n\n"
                                except Exception:
                                    continue

                        # Send final citations and retrieved chunks payload
                        cleaned_answer, citations = sanitize_citations(accumulated_text, retrieved_chunks)
                        final_payload = {
                            "answer": cleaned_answer,
                            "citations": citations,
                            "retrievedChunks": retrieved_chunks,
                            "done": True,
                        }
                        yield f"data: {json.dumps(final_payload)}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        await client.aclose()

                return StreamingResponse(stream_rag_tokens(), media_type="text/event-stream")

            # 4. Standard Non-Streaming JSON Response (Fastest for StudyVerse Frontend)
            vllm_payload = {
                "model": MODEL_ID,
                "messages": messages,
                "temperature": gen_temp,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "max_tokens": 500,
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
                    answer_text = self._generate_hf_text(messages, temperature=0.15, max_tokens=500)
                else:
                    raise HTTPException(status_code=502, detail=f"vLLM generation error: {str(vllm_err)}")

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
        max_tokens = body.get("max_tokens", 500)
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

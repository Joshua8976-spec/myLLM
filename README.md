# MyLLM: Fine-Tuning, Modal Deployment & Gemini Web Application

End-to-end pipeline for dataset preparation, QLoRA fine-tuning (Qwen 2.5 3B Instruct), serverless hosting on **Modal** (modal.com), and an interactive **Google Gemini-styled web client**.

---

## 🌟 Gemini-Styled Frontend Web Application (`frontend/`)

A modern web application modeled after the Google Gemini chat experience, built with React 19, TypeScript, Vite, Tailwind CSS, Lucide icons, and KaTeX math rendering.

### Features:
- **Google Gemini Aesthetics**: Deep charcoal dark theme (`#131314`), gradient shimmer sparkles, greeting cards with animated typography, and a floating pill input bar.
- **Direct Live Modal Integration**: Pre-configured to stream tokens from your deployed Modal endpoint (`https://rockefashy--studyverse-qwen-serve-serve.modal.run/v1`).
- **Real-Time Token Streaming**: Server-Sent Events (SSE) streaming with stop generation controls and typing cursor animations.
- **Task Mode Switcher**: Quick-toggle between the 4 fine-tuned modes:
  - 🎓 **Academic Tutor**: Socratic guidance with rendered LaTeX formulas ($...$).
  - 📝 **Adaptive Quiz**: Practice problems and conceptual quizzes.
  - 🔍 **Mistake Diagnostic**: Error root-cause diagnosis and repair steps.
  - 🌿 **Calm Companion**: Grounding breathing resets and exam anxiety support.
  - ⚡ **General**: Standard Qwen 2.5 3B assistance.
- **Collapsible Sidebar & Local History**: Save, switch, rename, and delete conversation threads stored persistently in your browser.
- **Settings Drawer**: Built-in latency tester (`/health`), temperature slider, max tokens, and custom endpoint URL inputs.

### Quickstart Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🚀 Backend & Pipeline Structure

- **`ml-pipeline/`**
  - `prepare_dataset.py`: Synthetic & curriculum dataset generation and validation into ChatML JSONL format.
  - `train.jsonl` / `val.jsonl`: Training and validation dataset splits across 4 modes.
  - `train_modal.py`: Modal cloud GPU training script using Unsloth 4-bit QLoRA.
  - `serve_llm_modal.py`: Production OpenAI-compatible FastAPI inference server on Modal.
  - `serve_notebook_modal.py`: Cloud JupyterLab server runner on Modal.
  - `.env.example`: Frontend environment variable configuration template.
- **`frontend/`**: Google Gemini-style web frontend.

For detailed pipeline setup instructions, visit [`ml-pipeline/README.md`](./ml-pipeline/README.md).

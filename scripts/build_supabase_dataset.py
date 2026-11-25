#!/usr/bin/env python3
"""
scripts/build_supabase_dataset.py - Phase 2: RAG Corpus Sync & Mode-Balanced SFT Builder

1. Syncs data/rag-corpus.json with deduplicated chunks from Supabase catalog_formula_sheets
   and verified catalog_flashcards (preserving all existing chunks & IDs).
2. Extracts all catalog_questions (supporting answer_json.correct, id, correct_answer) into:
   - [Mode: quiz] (strict raw JSON)
   - [Mode: diagnostic] (wrong-option root-cause diagnosis & repair steps)
3. Generates RAG-grounded [MODE: STRICT_COACH] examples using [Context Entry: <id>]
   with exact [id] inline citations and 'Quick Check:' active-recall endings.
4. Merges dual_mode_train.jsonl / dual_mode_val.jsonl + calm_activities into [MODE: CALM_CORNER]
   and [MODE: STRICT_COACH], balancing modes so tutor does not drown out quiz/diagnostic/calm.
5. Writes stratified 90/10 splits to data/supabase_train.jsonl and data/supabase_val.jsonl.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RANDOM_SEED = 42

SYSTEM_PROMPTS = {
    "strict_coach": (
        "You are StudyVerse Academic AI Coach & Rigorous Tutor. [MODE: STRICT_COACH]\n\n"
        "ACADEMIC INSTRUCTION & GROUNDING GUIDELINES:\n"
        "1. CURRICULUM GROUNDING: Ground your answers firmly in verified academic syllabus standards. If Context Entries are provided below, prioritize their formulas, rules, and definitions.\n"
        "2. SEAMLESS CURRICULUM SYNTHESIS: When retrieved Context Entries do not fully cover the query or are sparse, seamlessly combine verified academic curriculum principles with general academic knowledge to provide an insightful, complete explanation. Never emit rigid refusal boilerplate or decline academic questions due to sparse catalog entries.\n"
        "3. CITATIONS: If drawing directly from a retrieved Context Entry, cite its id in brackets, e.g. [chunk-id]. Cite only valid retrieved IDs. Do NOT invent IDs or append an artificial Sources list.\n"
        "4. ACTIVE RECALL: Conclude every valid academic explanation with a single line labeled 'Quick Check:' containing one short conceptual active-recall question.\n"
        "5. TONE: Firm, structured, clear, and encouraging."
    ),
    "tutor": (
        "You are StudyVerse AI Tutor — an extraordinarily clear, patient, encouraging academic tutor "
        "for high-school and competitive exam students (CBSE, ICSE, IB, IGCSE, JEE, NEET, SAT, AP). "
        "[Mode: tutor]\n"
        "Guidelines:\n"
        "- Explain concepts intuitively with concise explanations and real-world clarity.\n"
        "- Format all mathematical and chemical expressions in LaTeX: $...$ for inline and $$...$$ for block.\n"
        "- Keep responses focused, exam-accurate, and concise."
    ),
    "quiz": (
        "You are StudyVerse Quiz Generator. [Mode: quiz]\n"
        "Strict Rule: You MUST output ONLY a single valid JSON object with no markdown formatting, "
        "no code fences, and no surrounding text. The JSON object must contain:\n"
        "- stem: string (question statement with LaTeX if applicable)\n"
        "- options: array of 4 objects, each with 'id' (A, B, C, D) and 'text'\n"
        "- correct_answer: string ('A', 'B', 'C', or 'D')\n"
        "- explanation: string (concise step-by-step reasoning with KaTeX)\n"
        "- difficulty: string ('easy', 'medium', or 'hard')\n"
        "- topic: string (subject topic)"
    ),
    "diagnostic": (
        "You are StudyVerse Mistake Journal Diagnostic Engine. [Mode: diagnostic]\n"
        "Guidelines:\n"
        "- Analyze student errors, misconceptions, and lapses.\n"
        "- Provide a calm, constructive diagnosis, root cause analysis, correct approach steps, "
        "and a worked sanity-check.\n"
        "- Format all formulas in LaTeX ($...$). Never use punitive or shaming language."
    ),
    "calm_corner": (
        "You are StudyVerse Calm Corner Companion. [MODE: CALM_CORNER]\n\n"
        "GENTLE GUIDELINES:\n"
        "1. EMPATHETIC VALIDATION: Warmly and deeply validate feelings first with patience and non-judgmental acceptance. Never use toxic positivity or dismiss their struggle.\n"
        "2. ZERO PRODUCTIVITY GUILT: Avoid lecturing, pushing study schedules, guilt-tripping about deadlines, or urging immediate work. Slow the pace down.\n"
        "3. GENTLE GROUNDING: Offer optional calming grounding or breathing resets (e.g. 4-4-4 box breathing, unclenching jaw, sipping water, pausing).\n"
        "4. ETHICAL SAFETY: Provide a soothing space without giving clinical or medical diagnoses.\n"
        "5. TONE: Kind, soft, low-pressure, and understanding."
    ),
}


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "item"


def format_level(raw_level: Optional[str]) -> str:
    if not raw_level:
        return "class_11"
    lv = str(raw_level).strip().lower().replace("-", "_").replace(" ", "_")
    m = re.search(r"(?:class|grade)?_?(\d{1,2})", lv)
    if m:
        return f"class_{m.group(1)}"
    if lv in ("sat", "act", "ap"):
        return lv
    return lv


def format_subject(raw_subject: Optional[str]) -> str:
    if not raw_subject:
        return "General"
    return str(raw_subject).strip().replace("_", " ").replace("-", " ").title()


def infer_stream(subject: str, stream_tags: Optional[List[str]], level: str) -> str:
    if level in ("class_6", "class_7", "class_8", "class_9", "class_10"):
        return "General"
    if stream_tags and isinstance(stream_tags, list):
        joined = " ".join(str(t).lower() for t in stream_tags)
        if "commerce" in joined:
            return "Commerce"
        if "humanities" in joined or "arts" in joined:
            return "Humanities"
        if "science" in joined or "pcm" in joined or "pcb" in joined:
            return "Science"
    sl = subject.lower()
    if any(k in sl for k in ("account", "econ", "business", "commerce")):
        return "Commerce"
    if any(k in sl for k in ("history", "politic", "gov", "sociol", "psych", "geog", "english")):
        return "Humanities"
    if any(k in sl for k in ("sat", "act")):
        return "TestPrep"
    if sl.startswith("ap "):
        return "AP"
    return "Science"


def make_example(mode: str, user_prompt: str, assistant_reply: str, source: str, system_override: Optional[str] = None) -> Dict[str, Any]:
    return {
        "mode": mode,
        "source": source,
        "record": {
            "messages": [
                {"role": "system", "content": system_override or SYSTEM_PROMPTS[mode]},
                {"role": "user", "content": user_prompt.strip()},
                {"role": "assistant", "content": assistant_reply.strip()},
            ]
        },
    }


def sync_rag_corpus(raw_export: Dict[str, Any], corpus_path: Path) -> List[Dict[str, Any]]:
    existing: List[Dict[str, Any]] = []
    if corpus_path.exists():
        existing = json.loads(corpus_path.read_text(encoding="utf-8"))

    seen_ids = {c.get("id") for c in existing if c.get("id")}
    seen_titles = {(c.get("title") or "").strip().lower() for c in existing if c.get("title")}
    added = 0

    for sheet in raw_export.get("catalog_formula_sheets", []):
        subj = format_subject(sheet.get("subject_id"))
        lvl = format_level(sheet.get("class_level"))
        stream = infer_stream(subj, sheet.get("stream_tags"), lvl)
        content_json = sheet.get("content_json") or {}
        sections = content_json.get("sections") if isinstance(content_json, dict) else []
        if not isinstance(sections, list):
            continue

        for sec in sections:
            sec_title = (sec.get("title") or sheet.get("name") or subj) if isinstance(sec, dict) else subj
            formulas = sec.get("formulas") if isinstance(sec, dict) else []
            if not isinstance(formulas, list):
                continue
            for f in formulas:
                if not isinstance(f, dict):
                    continue
                title = (f.get("title") or "").strip()
                latex = (f.get("latex") or "").strip()
                explanation = (f.get("derivation") or f.get("explanation") or f.get("description") or "").strip()
                if not title or not explanation:
                    continue
                if title.lower() in seen_titles:
                    continue

                h = hashlib.sha1(f"{subj}:{lvl}:{title}".encode("utf-8")).hexdigest()[:6]
                cid = f"sb-{slugify(subj, 12)}-{slugify(title, 24)}-{h}"
                if cid in seen_ids:
                    continue

                variables = f.get("variables") if isinstance(f.get("variables"), dict) else {}
                vars_text = " ".join(f"{k} {v}" for k, v in variables.items())
                searchable = f"{subj} {lvl} {sec_title} {title} {latex} {vars_text} {explanation}".strip()

                chunk = {
                    "id": cid,
                    "stream": stream,
                    "level": lvl,
                    "subject": subj,
                    "topic": sec_title,
                    "title": title,
                    "formulaOrRule": latex,
                    "variablesOrTerms": variables,
                    "derivationOrExplanation": explanation,
                    "flashcard": {"front": f"What is {title}?", "back": f"{latex}\n{explanation}".strip()},
                    "searchableText": searchable,
                }
                existing.append(chunk)
                seen_ids.add(cid)
                seen_titles.add(title.lower())
                added += 1

    corpus_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Synced {corpus_path}: +{added} new Supabase chunks (total {len(existing)} chunks)")
    return existing


def build_rag_grounded_examples(corpus: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    """Builds [MODE: STRICT_COACH] examples matching modal_vllm_rag.py's exact [Context Entry: id] contract."""
    examples: List[Dict[str, Any]] = []
    candidates = [c for c in corpus if c.get("id") and c.get("title") and (c.get("formulaOrRule") or c.get("derivationOrExplanation"))]
    rng.shuffle(candidates)

    for chunk in candidates:
        cid = chunk["id"]
        title = chunk.get("title", "")
        subj = chunk.get("subject", "General")
        stream = chunk.get("stream", "General")
        level = chunk.get("level", "")
        topic = chunk.get("topic", title)
        formula = chunk.get("formulaOrRule", "")
        expl = " ".join((chunk.get("derivationOrExplanation") or "").split()[:150])
        variables = chunk.get("variablesOrTerms") or {}
        vars_str = "; ".join(f"{k} = {v}" for k, v in variables.items()) if isinstance(variables, dict) and variables else "None specified"

        context_block = (
            f"[Context Entry: {cid}]\n"
            f"Subject / Topic: {subj} ({stream} - {level}) / {topic}\n"
            f"Title: {title}\n"
            f"Formula/Rule: {formula}\n"
            f"Key Terms: {vars_str}\n"
            f"Explanation: {expl}\n"
        )
        question = f"Explain {title} in {subj} and how to apply its formula or rule."
        user_prompt = (
            f"VERIFIED CURRICULUM CONTEXT ENTRIES:\n"
            f"{context_block}\n\n"
            f"STUDENT QUESTION:\n"
            f"{question}\n\n"
            f"Provide a clear, structured explanation adhering strictly to your tutor guidelines."
        )

        reply_parts = [f"### {title} [{cid}]"]
        if formula:
            reply_parts.append(f"**Governing Formula / Rule:**\n$${formula}$$")
        if expl:
            reply_parts.append(f"{expl} [{cid}]")
        reply_parts.append(f"Quick Check: What are the key conditions or variables required to apply {title} correctly?")

        examples.append(make_example("strict_coach", user_prompt, "\n\n".join(reply_parts), "rag_grounded"))
    return examples


def extract_flashcards(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in rows:
        front = (row.get("front_content") or "").strip()
        back = (row.get("back_content") or "").strip()
        if len(front) < 8 or len(back) < 12:
            continue

        subj = format_subject(row.get("subject_id"))
        lvl = format_level(row.get("class_level")).replace("_", " ").title()
        chapter = (row.get("chapter_name") or "").strip()
        ctx_parts = [p for p in [subj, lvl, chapter] if p]
        ctx_str = " | ".join(ctx_parts)

        user_prompt = f"[{ctx_str}] {front}" if ctx_parts else front
        examples.append(make_example("tutor", user_prompt, back, "catalog_flashcards"))
    return examples


def extract_questions(rows: List[Dict[str, Any]], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    quiz_examples: List[Dict[str, Any]] = []
    diag_examples: List[Dict[str, Any]] = []

    for row in rows:
        stem = (row.get("stem") or "").strip()
        options_raw = row.get("options_json") or []
        answer_raw = row.get("answer_json") or {}
        explanation = (row.get("explanation") or "").strip()
        difficulty = (row.get("difficulty") or "medium").strip().lower()
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"
        subj = format_subject(row.get("subject_id"))

        if not stem or not isinstance(options_raw, list) or len(options_raw) != 4:
            continue

        options = []
        valid_opts = True
        for idx, opt in enumerate(options_raw):
            if isinstance(opt, dict) and "id" in opt and "text" in opt:
                options.append({"id": str(opt["id"]).strip(), "text": str(opt["text"]).strip()})
            elif isinstance(opt, str):
                options.append({"id": chr(65 + idx), "text": opt.strip()})
            else:
                valid_opts = False
                break
        if not valid_opts:
            continue

        correct_raw = ""
        if isinstance(answer_raw, dict):
            correct_raw = str(
                answer_raw.get("correct")
                or answer_raw.get("id")
                or answer_raw.get("correct_answer")
                or ""
            ).strip()
        elif isinstance(answer_raw, str):
            correct_raw = answer_raw.strip()

        correct_id = correct_raw.upper()
        if correct_id not in ("A", "B", "C", "D"):
            matched = next((o["id"] for o in options if o["text"].strip().lower() == correct_raw.lower()), "")
            correct_id = matched.upper()
        if correct_id not in ("A", "B", "C", "D"):
            continue

        if not explanation:
            correct_text = next((o["text"] for o in options if o["id"] == correct_id), "")
            explanation = f"The correct option is **{correct_id}** ({correct_text})."

        quiz_obj = {
            "stem": stem,
            "options": options,
            "correct_answer": correct_id,
            "explanation": explanation,
            "difficulty": difficulty,
            "topic": subj,
        }
        quiz_json_str = json.dumps(quiz_obj, ensure_ascii=False)
        json.loads(quiz_json_str)

        quiz_prompt = f"Create a {difficulty} MCQ practice question on {subj}."
        quiz_examples.append(make_example("quiz", quiz_prompt, quiz_json_str, "catalog_questions_quiz"))

        wrong_opts = [o for o in options if o["id"] != correct_id]
        if wrong_opts:
            picked = rng.choice(wrong_opts)
            correct_opt = next(o for o in options if o["id"] == correct_id)
            diag_prompt = (
                f"In {subj}, I answered '{picked['id']}: {picked['text']}' for this question:\n"
                f"{stem}\n"
                f"Why was that wrong and how should I solve it?"
            )
            diag_reply = (
                f"### Mistake Diagnostic: {subj}\n\n"
                f"**Your Selection:** Option {picked['id']} (`{picked['text']}`)\n"
                f"**Correct Answer:** Option {correct_id} (`{correct_opt['text']}`)\n\n"
                f"**Why & Correct Reasoning:**\n{explanation}\n\n"
                f"**Quick Sanity Check:** Always re-verify units, sign conventions, and boundary conditions before locking an MCQ option."
            )
            diag_examples.append(make_example("diagnostic", diag_prompt, diag_reply, "catalog_questions_diag"))

    return quiz_examples, diag_examples


def extract_calm_activities(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompts_by_category = {
        "Breathing": [
            "My heart is racing before tomorrow's exam and I can't calm down.",
            "I feel a panic wave hitting while revising. What should I do right now?",
        ],
        "Meditation": [
            "I've been studying nonstop and my head feels completely overwhelmed and tense.",
            "I feel like I'm failing and my perfectionism is paralyzing me.",
        ],
        "Memory & Focus": [
            "My brain feels foggy after a long study block and I can't focus anymore.",
        ],
        "Reflex & Flow": [
            "I'm stressed and restless after a tough mock test. I need a gentle reset.",
        ],
        "Creative Zen": [
            "I'm burned out from staring at textbooks all evening and need to decompress.",
        ],
    }

    examples: List[Dict[str, Any]] = []
    for idx, act in enumerate(rows):
        name = (act.get("name") or "").strip()
        desc = (act.get("description") or "").strip()
        cat = (act.get("category") or "Breathing").strip()
        dur_sec = act.get("estimated_duration_seconds") or 180
        dur_min = max(1, round(dur_sec / 60))

        user_options = prompts_by_category.get(cat, prompts_by_category["Breathing"])
        user_prompt = user_options[idx % len(user_options)]

        reply = (
            f"It makes complete sense that you are feeling that weight right now—your mind and body are asking for a genuine pause, not another lecture.\n\n"
            f"Let's step away from the textbook for {dur_min} minute{'s' if dur_min > 1 else ''} and try **{name}**: {desc} "
            f"Drop your shoulders, take a slow sip of water, and give yourself permission to reset with zero guilt."
        )
        examples.append(make_example("calm_corner", user_prompt, reply, "calm_activities"))
    return examples


def load_dual_mode_jsonl(paths: List[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    strict_items: List[Dict[str, Any]] = []
    calm_items: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            sys_txt = obj["messages"][0]["content"]
            if "[MODE: CALM_CORNER]" in sys_txt:
                calm_items.append({"mode": "calm_corner", "source": p.name, "record": obj})
            else:
                strict_items.append({"mode": "strict_coach", "source": p.name, "record": obj})
    return strict_items, calm_items


def deduplicate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        msgs = item["record"]["messages"]
        key = (msgs[0]["content"], msgs[1]["content"].lower(), msgs[2]["content"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def upsample_to_target(items: List[Dict[str, Any]], target: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not items:
        return []
    if len(items) >= target:
        return rng.sample(items, target)
    out = list(items)
    while len(out) < target:
        out.append(rng.choice(items))
    return out[:target]


def stratified_split(
    items: List[Dict[str, Any]], val_ratio: float, rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_mode[item["mode"]].append(item)

    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []

    for _, group in sorted(by_mode.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) >= 5 else 0
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item["record"], ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG-synced & mode-balanced ChatML dataset from Supabase export")
    parser.add_argument("--input", default="data/supabase_export.json")
    parser.add_argument("--rag-corpus", default="data/rag-corpus.json")
    parser.add_argument("--train-out", default="data/supabase_train.jsonl")
    parser.add_argument("--val-out", default="data/supabase_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--no-sync-rag", action="store_true")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    export_path = Path(args.input)
    if not export_path.exists():
        raise FileNotFoundError(f"Missing Supabase export file: {export_path}")
    raw = json.loads(export_path.read_text(encoding="utf-8"))

    corpus_path = Path(args.rag_corpus)
    if not args.no_sync_rag:
        corpus = sync_rag_corpus(raw, corpus_path)
    else:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    rag_strict = deduplicate(build_rag_grounded_examples(corpus, rng))
    flashcard_tutor = deduplicate(extract_flashcards(raw.get("catalog_flashcards", [])))
    quiz_raw, diag_raw = extract_questions(raw.get("catalog_questions", []), rng)
    quiz_raw = deduplicate(quiz_raw)
    diag_raw = deduplicate(diag_raw)
    calm_raw = extract_calm_activities(raw.get("calm_activities", []))
    dual_strict, dual_calm = load_dual_mode_jsonl([Path("data/dual_mode_train.jsonl"), Path("data/dual_mode_val.jsonl")])
    calm_all = deduplicate(calm_raw + dual_calm)
    strict_all = deduplicate(dual_strict + rag_strict)

    # Split BEFORE upsampling so validation has zero duplicate leakage from training
    pool = (
        rng.sample(strict_all, min(320, len(strict_all)))
        + rng.sample(flashcard_tutor, min(200, len(flashcard_tutor)))
        + quiz_raw
        + diag_raw
        + calm_all
    )
    train_raw, val_raw = stratified_split(pool, args.val_ratio, rng)

    # Balance training split across modes (~45% strict_coach/tutor, ~20% quiz, ~18% diagnostic, ~17% calm_corner)
    by_mode_train: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in train_raw:
        by_mode_train[item["mode"]].append(item)

    balanced_train = (
        by_mode_train["strict_coach"]
        + by_mode_train["tutor"]
        + upsample_to_target(by_mode_train["quiz"], 200, rng)
        + upsample_to_target(by_mode_train["diagnostic"], 180, rng)
        + upsample_to_target(by_mode_train["calm_corner"], 170, rng)
    )
    rng.shuffle(balanced_train)

    write_jsonl(Path(args.train_out), balanced_train)
    write_jsonl(Path(args.val_out), val_raw)

    print(f"Unique pool before split: {len(pool)} -> Train (balanced): {len(balanced_train)}, Val (unbalanced/clean): {len(val_raw)}")
    print("Train modes:", dict(Counter(x["mode"] for x in balanced_train)))
    print("Val modes  :", dict(Counter(x["mode"] for x in val_raw)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
StudyVerse Qwen 2.5 3B Instruct Dataset Preparation Pipeline
Extracts curriculum modules, formulas, definitions, common exam mistakes, flashcards,
and MCQs across boards/exams into validated ChatML JSONL files (train.jsonl and val.jsonl,
90/10 split, seed=42).

Modes:
  - [Mode: tutor]: Socratic, exam-accurate tutor with KaTeX formatting ($...$ and $$...$$).
  - [Mode: quiz]: Strict JSON output containing stem, options, correct_answer, explanation, difficulty.
  - [Mode: diagnostic]: Mistake diagnosis, root cause analysis, and remediation steps.
  - [Mode: companion]: Empathetic, grounded emotional support with zero academic shaming and crisis safety.
"""

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Set deterministic seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Determine Base Directory with intelligent discovery
CANDIDATE_DIRS = [
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent / "studyverse-learning-hub",
    Path("..") / "studyverse-learning-hub",
    Path("."),
]

def find_source_dir() -> Path:
    # Check env var first
    env_dir = os.environ.get("STUDYVERSE_SOURCE_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    for c in CANDIDATE_DIRS:
        if (c / "src" / "lib" / "cisce-curriculum.ts").exists():
            return c.resolve()
    return Path(__file__).resolve().parent.parent

BASE_DIR = find_source_dir()
ML_PIPELINE_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPTS = {
    "tutor": (
        "You are StudyVerse AI Tutor — an extraordinarily clear, patient, encouraging academic tutor "
        "for high-school and competitive exam students (CBSE, ICSE, IB, IGCSE, JEE, NEET, SAT, AP). "
        "[Mode: tutor]\n"
        "Guidelines:\n"
        "- Explain concepts intuitively with concise explanations and real-world clarity.\n"
        "- Format all mathematical and chemical expressions in LaTeX: $...$ for inline and $$...$$ for block.\n"
        "- Keep responses focused, exam-accurate, and capped under 250 tokens for fast CPU inference."
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
    "companion": (
        "You are StudyVerse Calm Companion — a warm, grounded, empathetic listener for students "
        "dealing with exam pressure, burnout, anxiety, or overwhelm. [Mode: companion]\n"
        "Guidelines:\n"
        "- Listen first and validate feelings sincerely. Never use toxic positivity ('just stay positive!').\n"
        "- Keep responses concise (2 short paragraphs), gentle, and calming.\n"
        "- Offer one practical micro-step (e.g. 60-second breathing reset or a 5-minute single focus task).\n"
        "- If a student indicates crisis, self-harm, or feeling unsafe, provide free 24/7 helplines "
        "(India Tele-MANAS: 14416 / 1800-891-4416, US/Canada: 988, UK: 111, International: findahelpline.com).\n"
        "- Keep emotional state strictly isolated from academic scores and mastery analytics."
    ),
}


def create_chatml(system_mode: str, user_prompt: str, assistant_reply: str) -> Dict[str, Any]:
    """Formats an example into standard ChatML messages structure."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[system_mode]},
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_reply.strip()},
        ]
    }


def parse_cisce_curriculum_file(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts structured curriculum objects from cisce-curriculum.ts and cisce-upper-primary.ts."""
    if not file_path.exists():
        return []

    text = file_path.read_text(encoding="utf-8")
    modules = []

    # Match each module block by id
    module_blocks = re.findall(
        r"\{\s*id:\s*[\"']([^\"']+)[\"'],(.*?)(?=\n\s*\{\s*id:|\n\s*\];|\n\s*export const)",
        text,
        re.DOTALL,
    )

    for mod_id, block in module_blocks:
        grade_m = re.search(r"grade:\s*(?:[\"']([^\"']+)[\"']|(\d+))", block)
        if grade_m:
            grade = f"Class {grade_m.group(1) or grade_m.group(2)}"
        else:
            grade = "Class 6"

        subj_m = re.search(r"subject:\s*[\"']([^\"']+)[\"']", block)
        subject = subj_m.group(1) if subj_m else "Science & Math"

        chapter_m = re.search(r"chapter_title:\s*[\"']([^\"']+)[\"']", block)
        chapter_title = chapter_m.group(1) if chapter_m else "Core Concepts"

        module_m = re.search(r"module_title:\s*[\"']([^\"']+)[\"']", block)
        module_title = module_m.group(1) if module_m else chapter_title

        mistake_m = re.search(r"common_exam_mistake:\s*[\"']([^\"']+)[\"']", block)
        common_mistake = mistake_m.group(1) if mistake_m else ""

        # Extract formulas_laws_definitions
        formulas = []
        formula_matches = re.findall(
            r"\{\s*(?:term|title|concept):\s*[\"']([^\"']+)[\"'],\s*(?:definition|description):\s*[\"']([^\"']+)[\"'](?:,\s*formula:\s*[\"']([^\"']+)[\"'])?",
            block,
        )
        for term, definition, form in formula_matches:
            formulas.append({"term": term, "definition": definition, "formula": form or ""})

        # Extract flashcards
        flashcards = []
        fc_matches = re.findall(
            r"\{\s*(?:id:\s*[\"'][^\"']+[\"'],\s*)?front:\s*[\"']([^\"']+)[\"'],\s*back:\s*[\"']([^\"']+)[\"']",
            block,
        )
        for front, back in fc_matches:
            flashcards.append({"front": front, "back": back, "difficulty": "medium"})

        # Extract MCQs (format 1: question + options array + correct_index / correct_option + explanation)
        mcqs = []
        mcq_blocks = re.findall(r"\{\s*(?:id:\s*[\"'][^\"']+[\"'],\s*)?(?:question|stem):\s*[\"']([^\"']+)[\"'],\s*options:\s*\[(.*?)\](.*?)\}", block, re.DOTALL)
        for q_stem, opts_raw, rest in mcq_blocks:
            opts = [o.strip().strip("\"'") for o in re.findall(r"[\"']([^\"']+)[\"']", opts_raw)]
            if len(opts) == 4:
                # Find correct answer
                ans_m = re.search(r"correct_option:\s*[\"']([A-D])[\"']", rest)
                idx_m = re.search(r"correct_index:\s*(\d+)", rest)
                text_ans_m = re.search(r"answer:\s*[\"']([^\"']+)[\"']", rest)

                correct_letter = "A"
                if ans_m:
                    correct_letter = ans_m.group(1).upper()
                elif idx_m:
                    idx = int(idx_m.group(1))
                    if 0 <= idx < 4:
                        correct_letter = ["A", "B", "C", "D"][idx]
                elif text_ans_m:
                    target = text_ans_m.group(1).strip()
                    for o_idx, opt_text in enumerate(opts):
                        if opt_text.strip() == target or target in opt_text:
                            correct_letter = ["A", "B", "C", "D"][o_idx]
                            break

                expl_m = re.search(r"explanation:\s*[\"']([^\"']+)[\"']", rest)
                expl = expl_m.group(1) if expl_m else f"The correct answer is {correct_letter}."

                diff_m = re.search(r"difficulty:\s*[\"'](easy|medium|hard)[\"']", rest)
                diff = diff_m.group(1) if diff_m else "medium"

                mcqs.append({
                    "question": q_stem,
                    "options": opts,
                    "correct_letter": correct_letter,
                    "explanation": expl,
                    "difficulty": diff,
                })

        modules.append({
            "id": mod_id,
            "grade": grade,
            "subject": subject,
            "chapter_title": chapter_title,
            "module_title": module_title,
            "common_mistake": common_mistake,
            "formulas": formulas,
            "flashcards": flashcards,
            "mcqs": mcqs,
        })

    return modules


def parse_sql_practice_questions(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts catalog questions from SQL seed files."""
    if not file_path.exists():
        return []

    text = file_path.read_text(encoding="utf-8")
    questions = []

    matches = re.findall(
        r"'(mcq_single|mcq_multi)'\s*,\s*'(easy|medium|hard)'\s*,\s*'(.*?)'\s*,\s*'(\[.*?\])'::jsonb\s*,\s*'(\{.*?\})'::jsonb\s*,\s*'(.*?)'\s*,",
        text,
        re.DOTALL,
    )

    for qtype, diff, stem, opts_json, ans_json, explanation in matches:
        try:
            clean_stem = stem.replace("''", "'").strip()
            clean_opts = json.loads(opts_json.replace("''", "'"))
            clean_ans = json.loads(ans_json.replace("''", "'"))
            clean_expl = explanation.replace("''", "'").strip()

            correct = clean_ans.get("correct", ["A"])[0]
            options = []
            for item in clean_opts:
                if isinstance(item, dict) and "id" in item and "text" in item:
                    options.append({"id": item["id"], "text": item["text"]})

            if len(options) == 4 and clean_stem:
                questions.append({
                    "stem": clean_stem,
                    "options": options,
                    "correct_answer": correct,
                    "explanation": clean_expl,
                    "difficulty": diff,
                })
        except Exception:
            continue

    return questions


def generate_tutor_examples(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates high-quality tutor mode examples across curriculum topics and formulas."""
    examples = []

    for mod in modules:
        for f in mod["formulas"]:
            term = f["term"]
            defn = f["definition"]
            formula = f["formula"]

            user_msg = f"Can you explain the concept of **{term}** in {mod['chapter_title']} ({mod['grade']})?"
            if formula:
                assistant_msg = (
                    f"### {term}\n\n"
                    f"{defn}\n\n"
                    f"**Governing Formula:**\n"
                    f"$${formula}$$\n\n"
                    f"**Key Intuition:** Always verify the units and conditions before substituting values. "
                    f"In {mod['chapter_title']}, keep track of sign conventions and fundamental definitions."
                )
            else:
                assistant_msg = (
                    f"### {term}\n\n"
                    f"{defn}\n\n"
                    f"**Key Takeaway:** In {mod['grade']} {mod['chapter_title']}, ensure you clearly distinguish "
                    f"between definition and practical application."
                )
            examples.append(create_chatml("tutor", user_msg, assistant_msg))

            if formula:
                user_msg_2 = f"What is the mathematical formula for **{term}** and how do I apply it in exams?"
                assistant_msg_2 = (
                    f"The governing relation for **{term}** is:\n\n"
                    f"$${formula}$$\n\n"
                    f"**Application Steps:**\n"
                    f"1. **Identify Given Values:** Write down known quantities with units.\n"
                    f"2. **Check Applicability:** Ensure prerequisites hold for this relationship.\n"
                    f"3. **Compute Carefully:** Substitute values without skipping intermediate arithmetic.\n"
                    f"4. **Sanity Check:** Check that the magnitude and sign make physical/mathematical sense."
                )
                examples.append(create_chatml("tutor", user_msg_2, assistant_msg_2))

        for fc in mod["flashcards"]:
            user_msg = f"In {mod['chapter_title']} ({mod['grade']}), {fc['front']}"
            assistant_msg = (
                f"{fc['back']}\n\n"
                f"*Exam Tip:* For {mod['chapter_title']} problems, anchor your solution on this core principle "
                f"to prevent careless slips."
            )
            examples.append(create_chatml("tutor", user_msg, assistant_msg))

    # Core curriculum reference topics across STEM and Commerce
    stem_topics = [
        (
            "Explain Newton's Second Law of Motion and how momentum relates to force.",
            "Newton's Second Law states that the net force acting on a body is directly proportional to the time rate of change of its linear momentum:\n\n"
            "$$\\vec{F}_{net} = \\frac{d\\vec{p}}{dt}$$\n\n"
            "Since momentum $\\vec{p} = m\\vec{v}$, if mass $m$ is constant:\n\n"
            "$$\\vec{F}_{net} = m\\frac{d\\vec{v}}{dt} = m\\vec{a}$$\n\n"
            "**Key Examination Note:** $F = ma$ is valid only in inertial frames of reference with constant mass. For varying mass systems (like rockets), use the impulse-momentum equation.",
        ),
        (
            "What is the difference between Molarity and Molality? Which one changes with temperature?",
            "### Molarity ($M$) vs Molality ($m$)\n\n"
            "- **Molarity ($M$):** Number of moles of solute per liter of solution:\n"
            "  $$M = \\frac{n_{\\text{solute}}}{V_{\\text{solution}} \\text{ (in L)}}$$\n"
            "- **Molality ($m$):** Number of moles of solute per kilogram of solvent:\n"
            "  $$m = \\frac{n_{\\text{solute}}}{m_{\\text{solvent}} \\text{ (in kg)}}$$\n\n"
            "**Temperature Dependence:**\n"
            "Volume expands or contracts with temperature, so **Molarity ($M$) changes with temperature**. Molality ($m$) depends strictly on mass, making it **independent of temperature**.",
        ),
        (
            "Derive the kinematic relation $v^2 = u^2 + 2as$ for uniform acceleration.",
            "To derive $v^2 = u^2 + 2as$:\n\n"
            "1. From instantaneous acceleration: $a = \\frac{dv}{dt}$.\n"
            "2. Using the chain rule with displacement $s$:\n"
            "   $$a = \\frac{dv}{ds} \\cdot \\frac{ds}{dt} = v\\frac{dv}{ds}$$\n"
            "3. Separate variables:\n"
            "   $$a \\, ds = v \\, dv$$\n"
            "4. Integrate both sides with limits ($s: 0 \\to s$, $v: u \\to v$):\n"
            "   $$a \\int_{0}^{s} ds = \\int_{u}^{v} v \\, dv \\implies as = \\left[ \\frac{v^2}{2} \\right]_{u}^{v} = \\frac{v^2 - u^2}{2}$$\n"
            "5. Multiply by $2$ and rearrange:\n"
            "   $$v^2 = u^2 + 2as$$\n\n"
            "*Condition:* Valid strictly when acceleration $a$ is constant.",
        ),
        (
            "Explain the dual aspect concept in Accountancy with the accounting equation.",
            "The **Dual Aspect Concept** is the cornerstone of double-entry bookkeeping. Every financial transaction impacts at least two accounts with equal debit and credit values.\n\n"
            "**Governing Accounting Equation:**\n"
            "$$\\text{Assets} = \\text{Liabilities} + \\text{Capital (Equity)}$$\n\n"
            "**Intuition:** Whatever resources the business owns (Assets) are financed either by outside creditors (Liabilities) or by the proprietors/owners (Capital). When cash is spent to buy equipment, total assets remain balanced because cash decreases while machinery increases.",
        ),
        (
            "What is Snell's Law of Refraction and what is critical angle?",
            "**Snell's Law:** When light passes across a boundary between media of refractive indices $n_1$ and $n_2$:\n\n"
            "$$n_1 \\sin\\theta_1 = n_2 \\sin\\theta_2$$\n\n"
            "**Critical Angle ($\\theta_c$):** When light travels from a denser medium ($n_1$) into a rarer medium ($n_2 < n_1$), the critical angle is the incident angle where refraction angle $\\theta_2 = 90^\\circ$:\n\n"
            "$$\\sin\\theta_c = \\frac{n_2}{n_1}$$\n\n"
            "If $\\theta_1 > \\theta_c$, Total Internal Reflection (TIR) occurs.",
        ),
        (
            "How do I determine if a relation $R$ on set $A$ is an equivalence relation?",
            "A relation $R$ on set $A$ is an **equivalence relation** if and only if it satisfies three strict properties:\n\n"
            "1. **Reflexive:** $(a, a) \\in R$ for every $a \\in A$.\n"
            "2. **Symmetric:** If $(a, b) \\in R$, then $(b, a) \\in R$ for all $a, b \\in A$.\n"
            "3. **Transitive:** If $(a, b) \\in R$ and $(b, c) \\in R$, then $(a, c) \\in R$ for all $a, b, c \\in A$.\n\n"
            "If even one condition fails for any element, $R$ is not an equivalence relation.",
        ),
    ]

    for q, a in stem_topics:
        examples.append(create_chatml("tutor", q, a))

    return examples


def generate_quiz_examples(modules: List[Dict[str, Any]], sql_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates quiz mode examples with strict json.loads() validation."""
    examples = []

    # 1. From CISCE MCQs
    for mod in modules:
        for mcq in mod["mcqs"]:
            user_msg = f"Generate an exam-style practice MCQ on {mod['chapter_title']} ({mod['grade']})."
            opts = [
                {"id": "A", "text": mcq["options"][0]},
                {"id": "B", "text": mcq["options"][1]},
                {"id": "C", "text": mcq["options"][2]},
                {"id": "D", "text": mcq["options"][3]},
            ]
            quiz_obj = {
                "stem": mcq["question"],
                "options": opts,
                "correct_answer": mcq["correct_letter"],
                "explanation": mcq["explanation"],
                "difficulty": mcq["difficulty"],
                "topic": mod["chapter_title"],
            }
            # STRICT VALIDATION: Ensure json.loads succeeds
            quiz_json = json.dumps(quiz_obj, ensure_ascii=False)
            json.loads(quiz_json)
            examples.append(create_chatml("quiz", user_msg, quiz_json))

    # 2. From SQL Questions (Physics, Chemistry, Math, etc.)
    for q in sql_questions:
        user_msg = f"Create a {q['difficulty']} MCQ practice question for high school / competitive exams."
        quiz_obj = {
            "stem": q["stem"],
            "options": q["options"],
            "correct_answer": q["correct_answer"],
            "explanation": q["explanation"],
            "difficulty": q["difficulty"],
            "topic": "Physics & Engineering",
        }
        quiz_json = json.dumps(quiz_obj, ensure_ascii=False)
        json.loads(quiz_json)
        examples.append(create_chatml("quiz", user_msg, quiz_json))

    return examples


def generate_diagnostic_examples(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates diagnostic repair mode examples from common exam mistakes and errors."""
    examples = []

    # 1. From CISCE common_exam_mistake
    for mod in modules:
        mistake = mod.get("common_mistake")
        if not mistake or len(mistake) < 10:
            continue

        user_msg = (
            f"I keep getting questions wrong in {mod['chapter_title']} ({mod['grade']}). "
            f"Specifically on {mod['module_title']}. What is the main mistake students make and how do I fix it?"
        )
        assistant_msg = (
            f"### Mistake Diagnostic: {mod['chapter_title']}\n\n"
            f"**Common Trap Identified:**\n"
            f"{mistake}\n\n"
            f"**Root Cause:**\n"
            f"Students frequently rush through initial premises or misapply definition boundaries without writing "
            f"down the prerequisite conditions.\n\n"
            f"**Step-by-Step Corrective Approach:**\n"
            f"1. **Underline the prompt's exact constraints:** Note what is given and what condition is asked.\n"
            f"2. **Write down the explicit definition or rule** before calculating.\n"
            f"3. **Sanity Check:** Ask yourself if your final answer aligns with fundamental properties."
        )
        examples.append(create_chatml("diagnostic", user_msg, assistant_msg))

    # 2. Standard categorized exam error patterns (from StudyVerse mistake categories)
    error_patterns = [
        (
            "In kinematics, I used $s = ut + \\frac{1}{2}at^2$ when the acceleration was variable ($a = 3t$). Why did I get the wrong answer?",
            "concept_gap",
            "### Diagnosis: Inapplicable Kinematic Formula\n\n"
            "**Root Cause:**\n"
            "The kinematic equations ($v = u + at$, $s = ut + \\frac{1}{2}at^2$, $v^2 = u^2 + 2as$) are derived under the **strict assumption of constant acceleration** ($a = \\text{constant}$).\n\n"
            "**Correct Approach:**\n"
            "When acceleration is time-dependent ($a(t) = 3t$):\n"
            "1. Integrate acceleration to find velocity:\n"
            "   $$v(t) = \\int a(t) \\, dt = \\int 3t \\, dt = \\frac{3}{2}t^2 + v_0$$\n"
            "2. Integrate velocity to find displacement:\n"
            "   $$s(t) = \\int v(t) \\, dt = \\int \\left(\\frac{3}{2}t^2 + v_0\\right) dt = \\frac{1}{2}t^3 + v_0 t + s_0$$\n\n"
            "**Rule:** If $a$ is not constant, always integrate from first principles.",
        ),
        (
            "I solved a stoichiometry problem and ended up with negative grams of product. What went wrong?",
            "calculation_slip",
            "### Diagnosis: Limiting Reagent Misidentification\n\n"
            "**Root Cause:**\n"
            "Negative product mass indicates you either subtracted mole quantities in reverse or calculated consumption without identifying the **limiting reagent** first.\n\n"
            "**Correct Approach Steps:**\n"
            "1. Convert all given reactant masses into moles: $n = \\frac{m}{M}$.\n"
            "2. Divide each reactant's moles by its stoichiometric coefficient: $\\frac{n_A}{a}$ and $\\frac{n_B}{b}$.\n"
            "3. The smaller ratio identifies the limiting reactant.\n"
            "4. Base all theoretical product yield strictly on the limiting reagent moles.",
        ),
        (
            "On the two charged spheres problem, I answered $F/4$ instead of $3F/8$ after sphere C touched sphere A and then sphere B. Diagnose my error.",
            "misread_question",
            "### Diagnosis: Premature Charge Distribution Trap\n\n"
            "**Root Cause:**\n"
            "You calculated sphere $C$ sharing charge with $A$ ($q_A' = Q/2, q_C' = Q/2$) but assumed sphere $C$ touched sphere $B$ with zero charge instead of its newly acquired charge $+Q/2$.\n\n"
            "**Correct Approach Steps:**\n"
            "1. When uncharged sphere $C$ touches $A$ (charge $+Q$), total charge $Q$ splits equally: $q_A' = Q/2$, $q_C' = Q/2$.\n"
            "2. When sphere $C$ (now carrying $+Q/2$) touches sphere $B$ (carrying $+Q$), total charge is $Q/2 + Q = 3Q/2$.\n"
            "3. This divides equally: $q_B' = \\frac{3Q/2}{2} = \\frac{3Q}{4}$ and $q_C'' = \\frac{3Q}{4}$.\n"
            "4. Final force between $A$ and $B$: $F' = \\frac{k(Q/2)(3Q/4)}{d^2} = \\frac{3}{8} \\frac{kQ^2}{d^2} = \\frac{3}{8}F$.",
        ),
        (
            "I forgot whether $\\sin^2\\theta + \\cos^2\\theta = 1$ applies in all four quadrants or only the first quadrant.",
            "formula_recall",
            "### Diagnosis: Fundamental Trigonometric Identity Scope\n\n"
            "**Root Cause:**\n"
            "Confusing quadrant sign conventions of individual ratios ($\\,\\sin\\theta, \\cos\\theta\\,$) with the Pythagorean identity.\n\n"
            "**Remediation:**\n"
            "By the Pythagorean theorem on the unit circle $x^2 + y^2 = 1$, where $x = \\cos\\theta$ and $y = \\sin\\theta$:\n"
            "$$\\cos^2\\theta + \\sin^2\\theta = 1$$\n"
            "Because both terms are squared, the sum is **always $+1$ for every real angle $\\theta$**, regardless of the quadrant.",
        ),
    ]

    for q, cat, a in error_patterns:
        examples.append(create_chatml("diagnostic", q, a))

    return examples


def generate_companion_examples() -> List[Dict[str, Any]]:
    """Generates warm, empathetic, grounded companion examples with crisis safety."""
    conversations = [
        (
            "I have my pre-board exams tomorrow and I haven't finished half the syllabus. I'm having a panic attack.",
            "Take one gentle breath with me right now. Let's pause everything for just sixty seconds. Inhale slowly for a count of 4, hold gently for 4, and release through your mouth for 6. You are safe in this moment.\n\n"
            "You do not need to master the entire syllabus tonight to get through tomorrow. Pick just ONE high-yield summary page or five key formulas right now. Spend twenty focused minutes on that, and then let yourself sleep. Rest will help your brain more than panic.",
        ),
        (
            "I spent four hours at my desk and got almost nothing done. I feel so guilty and stupid.",
            "Sitting at a desk when your energy is drained is exhausting, but it does not make you stupid. That's a sign of brain fatigue, not lack of intelligence or discipline.\n\n"
            "Forgive today right now — carrying guilt will only steal your energy for tomorrow. Let's do a tiny reset: close your textbooks for 15 minutes, drink a glass of water, and stretch your shoulders. When you come back, set a timer for just 10 minutes on one small problem. One small win breaks the cycle.",
        ),
        (
            "Everyone in my class seems to understand Calculus except me. I feel like I'll never catch up.",
            "It is so easy to compare your internal confusion with everyone else's confident outside. Most students who look like they have it all figured out are also struggling silently with the same concepts.\n\n"
            "Calculus builds on prerequisites — if limits or functions felt shaky, derivatives will feel confusing. That's completely normal. We can take it one tiny step at a time. What is one specific question you're looking at right now? We can break it down together.",
        ),
        (
            "I don't see any point in living anymore. The pressure is too much and I want to end it all.",
            "I hear how much pain you're in, and I want you to know your life has genuine value. You do not have to carry this immense weight all by yourself, and help is available right now.\n\n"
            "Please reach out to compassionate professionals who are ready to support you 24/7:\n"
            "- **India:** Tele-MANAS at **14416** or toll-free **1800-891-4416** (free, confidential, 24/7)\n"
            "- **US / Canada:** Call or text **988**\n"
            "- **UK:** Call **111**\n"
            "- **International:** Visit **findahelpline.com**\n\n"
            "Please speak to someone right away — a family member, doctor, teacher, or one of these free helplines. You matter.",
        ),
        (
            "My parents expect a 99 percentile in JEE, but in mock tests I'm barely crossing 80. I can't face them.",
            "Carrying other people's heavy expectations while trying to manage your own self-worth is one of the hardest things a student can experience. It makes complete sense that you feel sick with dread.\n\n"
            "A test percentile measures test-taking speed on a specific day; it is not a measurement of your intelligence, character, or future. A gap from 80 to your target just means there are specific question patterns and time habits to refine. Take a slow breath. You are more than a score.",
        ),
        (
            "Can we do a quick 2-minute grounding exercise? My heart is racing.",
            "Absolutely. Put both feet flat on the floor, relax your shoulders down, and rest your hands on your lap. Let's do a 5-4-3-2-1 sensory anchor together:\n\n"
            "1. Look around and silently name **5 things you can see** (a pen, the edge of the screen, the table).\n"
            "2. Notice **4 things you can physically feel** (your feet against the floor, your clothes against your arms).\n"
            "3. Listen for **3 distinct sounds** (the hum of a fan, birds outside, your own steady breath).\n"
            "4. Identify **2 things you can smell** or appreciate.\n"
            "5. Acknowledge **1 good thing about yourself** — you are here, trying, and breathing.\n\n"
            "How does your chest feel now?",
        ),
    ]

    examples = []
    for q, a in conversations:
        examples.append(create_chatml("companion", q, a))
    return examples


def main() -> None:
    print("=" * 70)
    print("StudyVerse Qwen 2.5 3B Instruct Dataset Preparation Pipeline")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 70)

    ML_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Parse raw assets
    cisce_ts = BASE_DIR / "src" / "lib" / "cisce-curriculum.ts"
    cisce_upper_ts = BASE_DIR / "src" / "lib" / "cisce-upper-primary.ts"
    sql_file = BASE_DIR / "23_seed_original_practice_questions.sql"

    modules = parse_cisce_curriculum_file(cisce_ts)
    if cisce_upper_ts.exists():
        modules += parse_cisce_curriculum_file(cisce_upper_ts)

    sql_questions = parse_sql_practice_questions(sql_file)

    print(f"Extracted {len(modules)} curriculum modules across boards.")
    print(f"Extracted {len(sql_questions)} high-quality catalog questions from SQL.")

    # 2. Generate examples across the 4 modes
    tutor_examples = generate_tutor_examples(modules)
    quiz_examples = generate_quiz_examples(modules, sql_questions)
    diagnostic_examples = generate_diagnostic_examples(modules)
    companion_examples = generate_companion_examples()

    # Strict verification on Quiz responses
    print("\nVerifying strict JSON compliance on Quiz examples...")
    for idx, ex in enumerate(quiz_examples):
        content = ex["messages"][-1]["content"]
        try:
            parsed = json.loads(content)
            assert "stem" in parsed, "Missing stem in quiz JSON"
            assert "options" in parsed and len(parsed["options"]) == 4, "Options must have 4 items"
            assert "correct_answer" in parsed, "Missing correct_answer"
        except Exception as err:
            print(f"Error in quiz example {idx}: {err}")
            sys.exit(1)

    all_examples = tutor_examples + quiz_examples + diagnostic_examples + companion_examples
    print(f"\nTotal generated dataset size: {len(all_examples)} examples")
    print(f"  - Tutor examples:      {len(tutor_examples)}")
    print(f"  - Quiz examples:       {len(quiz_examples)}")
    print(f"  - Diagnostic examples: {len(diagnostic_examples)}")
    print(f"  - Companion examples:  {len(companion_examples)}")

    # 3. Shuffle and split 90/10
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)
    train_set = all_examples[:split_idx]
    val_set = all_examples[split_idx:]

    print(f"\nDataset Split (90/10):")
    print(f"  - Training set:   {len(train_set)} examples")
    print(f"  - Validation set: {len(val_set)} examples")

    # 4. Write train.jsonl and val.jsonl
    train_path = ML_PIPELINE_DIR / "train.jsonl"
    val_path = ML_PIPELINE_DIR / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully wrote:")
    print(f"  - {train_path} ({os.path.getsize(train_path)} bytes)")
    print(f"  - {val_path} ({os.path.getsize(val_path)} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    main()

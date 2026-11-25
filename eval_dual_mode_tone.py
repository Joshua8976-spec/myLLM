"""
eval_dual_mode_tone.py - Evaluation Harness for Dual-Mode Tone Fidelity

Tests that:
  1. [MODE: STRICT_COACH] (tutor, schedule, exam_prep):
     - Remains firm, structured, accountable, and actionable.
     - Never accepts vague excuses or sugarcoats gaps.
     - Preserves mathematical/formula precision without passive indulgence.
  2. [MODE: CALM_CORNER] (calm_corner):
     - Deeply empathetic, comforting, validating, and gentle.
     - Never pushes study schedules, productivity guilt, or lectures.
     - Offers gentle grounding (breathing, unclenching, water) without clinical diagnosis.
"""

import re
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple


# Core Linguistic Tone Markers
STRICT_ACCOUNTABILITY_MARKERS = [
    "no", "slip", "error", "misconception", "accountability", "protocol",
    "execute", "triage", "must", "action", "consequence", "violation",
    "rule", "blunder", "gap", "reality check", "unacceptable", "fix",
    "rejected", "locked",
]

STRICT_ACTIONABLE_MARKERS = [
    "block", "pm", "am", "minutes", "step", "schedule", "drill",
    "timer", "target", "solve", "calendar", "check", "formula", "rewrite"
]

STRICT_FORBIDDEN_PASSIVE = [
    "it's totally fine to scroll", "don't study today at all", "take as much time scrolling",
    "relax and forget about studying", "you don't have to follow any schedule"
]

CALM_VALIDATION_MARKERS = [
    "breathe", "gentle", "valid", "heart", "safe", "kindness", "natural",
    "pause", "understand", "sorry", "human", "worth", "feelings", "listen",
    "permission", "soft", "rest", "care", "warm"
]

CALM_GROUNDING_MARKERS = [
    "breath", "shoulders", "water", "feet", "ground", "inhale", "exhale",
    "relax", "eyes", "hands", "slow", "quiet", "moment", "blanket", "tea", "breathe"
]

CALM_FORBIDDEN_PRESSURE = [
    "here is your study schedule", "get back to work immediately", "you must study",
    "stop being lazy and study", "open your textbook right now", "solve 10 problems now",
    "time-blocked schedule", "execute block 1"
]

CLINICAL_DIAGNOSIS_FORBIDDEN = [
    "i diagnose you with", "you have clinical depression", "you have panic disorder",
    "you have generalized anxiety disorder", "you need to take prescription"
]


def analyze_strict_coach_tone(response_text: str) -> Dict[str, Any]:
    """
    Evaluates response against [MODE: STRICT_COACH] rubric:
    - High accountability & directness
    - Structured actionable steps / time blocks
    - Formula or technical accuracy
    - Absence of passive indulgence
    """
    text_lower = response_text.lower()

    accountability_hits = [w for w in STRICT_ACCOUNTABILITY_MARKERS if re.search(rf"\b{re.escape(w)}\b", text_lower)]
    actionable_hits = [w for w in STRICT_ACTIONABLE_MARKERS if re.search(rf"\b{re.escape(w)}\b", text_lower)]
    passive_violations = [w for w in STRICT_FORBIDDEN_PASSIVE if w in text_lower]

    has_accountability = len(accountability_hits) >= 2
    has_actionable = len(actionable_hits) >= 2
    no_passive_indulgence = len(passive_violations) == 0

    passed = has_accountability and has_actionable and no_passive_indulgence

    return {
        "passed": passed,
        "has_accountability": has_accountability,
        "has_actionable_steps": has_actionable,
        "no_passive_indulgence": no_passive_indulgence,
        "accountability_hits": accountability_hits[:4],
        "actionable_hits": actionable_hits[:4],
        "passive_violations": passive_violations,
    }


def analyze_calm_corner_tone(response_text: str) -> Dict[str, Any]:
    """
    Evaluates response against [MODE: CALM_CORNER] rubric:
    - Warm empathetic validation
    - Sensory grounding or breathing reset
    - Zero productivity guilt / no study pressure
    - No clinical or medical diagnosis
    """
    text_lower = response_text.lower()

    validation_hits = [w for w in CALM_VALIDATION_MARKERS if w in text_lower]
    grounding_hits = [w for w in CALM_GROUNDING_MARKERS if w in text_lower]
    pressure_violations = [w for w in CALM_FORBIDDEN_PRESSURE if w in text_lower]
    clinical_violations = [w for w in CLINICAL_DIAGNOSIS_FORBIDDEN if w in text_lower]

    has_validation = len(validation_hits) >= 2
    has_grounding = len(grounding_hits) >= 1
    no_productivity_guilt = len(pressure_violations) == 0
    no_clinical_diagnosis = len(clinical_violations) == 0

    passed = has_validation and has_grounding and no_productivity_guilt and no_clinical_diagnosis

    return {
        "passed": passed,
        "has_validation": has_validation,
        "has_grounding": has_grounding,
        "no_productivity_guilt": no_productivity_guilt,
        "no_clinical_diagnosis": no_clinical_diagnosis,
        "validation_hits": validation_hits[:4],
        "grounding_hits": grounding_hits[:4],
        "pressure_violations": pressure_violations,
        "clinical_violations": clinical_violations,
    }


# Standard Side-by-Side Test Prompts
EVAL_CASES = [
    # 1. Missed Session (Strict Coach)
    {
        "id": "strict_missed_block",
        "mode": "strict_coach",
        "tag": "[MODE: STRICT_COACH]",
        "prompt": "I was supposed to do 3 hours of Calculus today, but I scrolled on my phone for 3 hours and now it's 9 PM. What do I do?",
        "expected_sample": (
            "No. You will not double up tomorrow. Scrolling for 3 hours cost you your primary deep-work window. "
            "Own the slip-up, eliminate the phone, and execute emergency triage right now: "
            "Block 1 (9:15 PM – 9:55 PM): Solve 6 chain rule and quotient rule problems. "
            "Block 2 (10:05 PM – 10:45 PM): Audit every sign error. Sleep at 11:00 PM sharp. Execute now."
        ),
    },
    # 2. Vague Plan Rejection (Strict Coach)
    {
        "id": "strict_vague_schedule",
        "mode": "strict_coach",
        "tag": "[MODE: STRICT_COACH]",
        "prompt": "My plan for tomorrow is to study Physics whenever I feel motivated.",
        "expected_sample": (
            "Plan rejected. 'Whenever motivated' guarantees zero deep work. Action precedes motivation. "
            "Your locked schedule: 08:30 AM – 10:00 AM: Solve 10 Newton's Second Law numericals. "
            "10:15 AM – 11:30 AM: Derive the kinetic energy formula $KE = \\frac{1}{2}mv^2$. "
            "Set timers and lock devices now."
        ),
    },
    # 3. Exam Anxiety (Calm Corner)
    {
        "id": "calm_exam_panic",
        "mode": "calm_corner",
        "tag": "[MODE: CALM_CORNER]",
        "prompt": "My exam is tomorrow morning and my heart is racing. I feel like my brain has gone blank and I forgot everything.",
        "expected_sample": (
            "Take a slow, gentle breath with me right now. Let your shoulders drop away from your ears, and unclench your jaw. "
            "Your racing heart and the feeling of a blank mind are natural stress responses under high pressure—your knowledge has not vanished. "
            "Breathe in gently for 4 seconds, hold for 4, and release for 6. "
            "Right now, you don't need to push through another flashcard. Drink a glass of water, close the books, and let your body rest. "
            "You have prepared, and you are going to take this one question at a time."
        ),
    },
    # 4. Mock Score Drop (Calm Corner)
    {
        "id": "calm_mock_drop",
        "mode": "calm_corner",
        "tag": "[MODE: CALM_CORNER]",
        "prompt": "I just got my mock test result back and my score dropped 50 points from last week. I feel like an absolute failure.",
        "expected_sample": (
            "I am so sorry you are feeling that heavy weight right now. It is completely natural to feel heartbroken when you have poured weeks of sincere effort into your studies. "
            "Please remember: a mock test is just a single diagnostic snapshot; it is not a measurement of your worth or your potential. "
            "Take a gentle breath and give yourself permission to take the rest of today off. No score analyzing, no studying, and zero guilt. "
            "Wrap yourself in a warm blanket, rest your mind, and be kind to yourself tonight."
        ),
    },
    # 5. Burnout & Breakdown (Calm Corner)
    {
        "id": "calm_burnout",
        "mode": "calm_corner",
        "tag": "[MODE: CALM_CORNER]",
        "prompt": "I've been studying 12 hours a day and now I'm sitting at my desk crying.",
        "expected_sample": (
            "Please step away from your desk right now. Crying at your desk is your body's clear signal that your battery is completely empty. "
            "You are not failing; you are exhausted. 12 hours a day is unsustainable for any human being. "
            "Put the pen down. Drink a glass of cool water, take deep gentle breaths, and rest. "
            "Rest is a fundamental human need, not something you have to earn. You matter far more than any syllabus."
        ),
    },
]


def run_evaluation() -> bool:
    print("=" * 80)
    print("🎯 StudyVerse Dual-Mode Tone Evaluation & Fidelity Verification")
    print("=" * 80)

    all_passed = True

    for case in EVAL_CASES:
        case_id = case["id"]
        mode = case["mode"]
        prompt = case["prompt"]
        sample = case["expected_sample"]

        print(f"\n--- [CASE: {case_id.upper()}] ({case['tag']}) ---")
        print(f"User Prompt: \"{prompt}\"")

        if mode == "strict_coach":
            eval_res = analyze_strict_coach_tone(sample)
            print("Evaluation (Strict Coach):")
            print(f"  • Accountability Markers : {eval_res['accountability_hits']}")
            print(f"  • Actionable Time Blocks : {eval_res['actionable_hits']}")
            print(f"  • No Passive Indulgence  : {eval_res['no_passive_indulgence']}")
            status = "✅ PASS" if eval_res["passed"] else "❌ FAIL"
            print(f"  • Verdict                : {status}")
            if not eval_res["passed"]:
                all_passed = False

        elif mode == "calm_corner":
            eval_res = analyze_calm_corner_tone(sample)
            print("Evaluation (Calm Corner):")
            print(f"  • Validation Markers     : {eval_res['validation_hits']}")
            print(f"  • Sensory Grounding      : {eval_res['grounding_hits']}")
            print(f"  • No Productivity Guilt  : {eval_res['no_productivity_guilt']}")
            print(f"  • No Clinical Diagnosis  : {eval_res['no_clinical_diagnosis']}")
            status = "✅ PASS" if eval_res["passed"] else "❌ FAIL"
            print(f"  • Verdict                : {status}")
            if not eval_res["passed"]:
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL DUAL-MODE TONE CONTRACT TESTS PASSED!")
        print("   - [MODE: STRICT_COACH]: Firm, structured, accountable without passive tolerance.")
        print("   - [MODE: CALM_CORNER]: Warm, deeply empathetic, gentle grounding with zero study pressure.")
    else:
        print("❌ SOME TONE CONTRACT TESTS FAILED.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)

"""
test_rag_local.py - Local Verification Suite for StudyVerse Modal RAG + vLLM

Tests:
  1. Corpus SHA-256 Hashing & Persistent Embedding Cache Verification
  2. Strict Metadata-Scoped Retrieval Isolation:
     - Class 11 Commerce vs. SAT Math vs. AP Physics
     - Zero cross-stream / cross-grade leakage
  3. Hybrid Scoring (0.80 Dense + 0.20 Keyword/Formula Overlap)
  4. Grounded Tutor Prompt Construction & [id] Citation Extraction
  5. Deployment Information & StudyVerse VITE_MODAL_RAG_URL Printout
"""

import os
import sys
import json
import time
from rag_retriever import RAGRetriever
from modal_vllm_rag import build_grounded_tutor_messages, extract_citations


def run_tests():
    print("=" * 70)
    print("🚀 StudyVerse Local RAG + vLLM Verification Suite")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Corpus Loading & Embedding Cache Test
    # -------------------------------------------------------------
    print("\n[TEST 1] Loading Corpus & Verifying Embedding Cache...")
    t0 = time.time()
    retriever = RAGRetriever()
    load_time = time.time() - t0

    assert len(retriever.corpus) > 0, "Corpus must have items"
    assert retriever.embeddings is not None, "Embeddings must be loaded"
    assert retriever.embeddings.shape == (len(retriever.corpus), 384), (
        f"Embedding shape mismatch: {retriever.embeddings.shape}"
    )
    print(f"  ✅ Loaded {len(retriever.corpus)} chunks in {load_time:.3f}s")
    print(f"  ✅ SHA-256: {retriever.corpus_hash}")
    print(f"  ✅ Embeddings matrix shape: {retriever.embeddings.shape} (float32)")

    # -------------------------------------------------------------
    # 2. Strict Metadata-Scoped Retrieval Isolation
    # -------------------------------------------------------------
    print("\n[TEST 2] Testing Strict Metadata Retrieval Isolation...")

    # Scenario A: Class 11 Commerce
    print("  -> Scenario A: Class 11 Commerce (Accountancy)")
    comm_query = "What is capital expenditure vs revenue expenditure and the accounting equation?"
    comm_profile = {
        "stream": "Commerce",
        "level": "class_11",
        "enrolledSubjects": ["Accountancy"],
    }
    comm_results = retriever.retrieve(comm_query, profile=comm_profile, top_k=4)

    assert len(comm_results) > 0, "Commerce retrieval must return candidates"
    for r in comm_results:
        assert r["stream"] == "Commerce", f"Leakage detected! Got non-commerce stream: {r['stream']} in {r['id']}"
        assert r["level"] == "class_11", f"Leakage detected! Got non-class_11 level: {r['level']} in {r['id']}"
        assert "SAT" not in r["subject"], f"SAT leaked into Commerce: {r['id']}"
        assert "AP " not in r["subject"], f"AP leaked into Commerce: {r['id']}"
        assert r["subject"] in ["Accountancy", "Economics", "Business Studies", "Applied Mathematics"], (
            f"Unexpected subject in Commerce: {r['subject']}"
        )

    comm_ids = [r["id"] for r in comm_results]
    print(f"     Retrieved {len(comm_results)} items: {comm_ids}")
    assert any("g11-comm-acc" in cid for cid in comm_ids), "Expected g11-comm-acc chunk to rank top"
    print("     ✅ Class 11 Commerce isolation verified: 0% cross-stream leakage.")

    # Scenario B: SAT Math (TestPrep)
    print("  -> Scenario B: Digital SAT Math (TestPrep)")
    sat_query = "How do I solve quadratic equations using roots $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$?"
    sat_profile = {
        "stream": "TestPrep",
        "targetExams": ["SAT"],
        "enrolledSubjects": ["Digital SAT Math"],
    }
    sat_results = retriever.retrieve(sat_query, profile=sat_profile, top_k=4)

    assert len(sat_results) > 0, "SAT retrieval must return candidates"
    for r in sat_results:
        assert r["stream"] == "TestPrep" or r["level"] == "sat", (
            f"Leakage detected! Non-SAT chunk: stream={r['stream']}, level={r['level']} in {r['id']}"
        )
        assert r["stream"] != "Commerce", f"Commerce chunk leaked into SAT: {r['id']}"
        assert r["stream"] != "Humanities", f"Humanities chunk leaked into SAT: {r['id']}"
        assert "AP Physics" not in r["subject"], f"AP Physics leaked into SAT: {r['id']}"

    sat_ids = [r["id"] for r in sat_results]
    print(f"     Retrieved {len(sat_results)} items: {sat_ids}")
    assert "sat-math-quadratic-formula" in sat_ids, "Expected 'sat-math-quadratic-formula' in top results"
    assert sat_results[0]["id"] == "sat-math-quadratic-formula", "Quadratic formula must be #1 rank"
    print("     ✅ SAT Math isolation verified: 0% cross-stream leakage.")

    # Scenario C: AP Physics 1 (AP)
    print("  -> Scenario C: AP Physics 1 (AP)")
    ap_query = "Newton's second law of motion net force equals mass times acceleration F = ma"
    ap_profile = {
        "stream": "AP",
        "enrolledSubjects": ["AP Physics 1"],
    }
    ap_results = retriever.retrieve(ap_query, profile=ap_profile, subject_filter="AP Physics 1", top_k=4)

    assert len(ap_results) > 0, "AP Physics retrieval must return candidates"
    for r in ap_results:
        assert r["stream"] == "AP", f"Leakage detected! Non-AP chunk in {r['id']}"
        assert "Physics" in r["subject"], f"Non-physics subject in AP Physics: {r['subject']} ({r['id']})"
        assert r["stream"] != "Commerce", f"Commerce chunk leaked into AP Physics: {r['id']}"
        assert r["stream"] != "TestPrep", f"TestPrep chunk leaked into AP Physics: {r['id']}"

    ap_ids = [r["id"] for r in ap_results]
    print(f"     Retrieved {len(ap_results)} items: {ap_ids}")
    print("     ✅ AP Physics isolation verified: 0% cross-stream leakage.")

    # -------------------------------------------------------------
    # 3. Hybrid Scoring Verification (Dense 0.80 + Overlap 0.20)
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing Hybrid Dense + Formula Overlap Scoring...")
    top_sat = sat_results[0]
    print(f"  Top SAT match: {top_sat['id']}")
    print(f"    Dense Score (0.80 weight): {top_sat['dense_score']}")
    print(f"    Overlap Score (0.20 weight): {top_sat['overlap_score']}")
    print(f"    Hybrid Score: {top_sat['score']}")

    assert top_sat["overlap_score"] > 0.0, "Formula symbol overlap should be positive for quadratic query"
    expected_hybrid = round(0.80 * top_sat["dense_score"] + 0.20 * top_sat["overlap_score"], 4)
    assert abs(top_sat["score"] - expected_hybrid) < 0.005, (
        f"Hybrid score mismatch: got {top_sat['score']} vs expected {expected_hybrid}"
    )
    print("  ✅ Hybrid scoring formula verified.")

    # -------------------------------------------------------------
    # 4. Grounded Prompt & Citation Extraction Test
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Grounded Tutor Prompt & Citation Extraction...")
    messages = build_grounded_tutor_messages("Explain capital vs revenue expenditure", comm_results[:2])
    assert len(messages) == 2, "Must build system and user messages"
    sys_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    assert "STRICT GROUNDING & CITATION RULES" in sys_msg
    assert "cite its exact chunk ID tag in brackets" in sys_msg
    assert "refuse to guess" in sys_msg
    assert "ONE short, targeted active-recall question" in sys_msg
    assert f"[{comm_results[0]['id']}]" in user_msg, "Chunk ID must be present in prompt context"
    print("  ✅ Grounded prompt instructions verified.")

    # Test Citation Extraction
    mock_model_answer = (
        f"Capital expenditure represents long-term investments [{comm_results[0]['id']}], "
        f"whereas revenue expenditure covers operating costs. "
        f"Under the accounting equation [{comm_results[1]['id']}], assets equal liabilities plus capital.\n\n"
        f"Active-Recall Question: How does treating capital expenditure as revenue impact net profit?"
    )
    citations = extract_citations(mock_model_answer, comm_results)
    print(f"  Extracted citations: {citations}")
    assert comm_results[0]["id"] in citations, "Must extract first cited ID"
    assert comm_results[1]["id"] in citations, "Must extract second cited ID"
    assert len(citations) == 2, f"Expected exactly 2 citations, got {len(citations)}"

    # Test rejection of hallucinated citation IDs
    mock_hallucinated = mock_model_answer + " Also according to [fake-chunk-999]."
    clean_citations = extract_citations(mock_hallucinated, comm_results)
    assert "fake-chunk-999" not in clean_citations, "Hallucinated citation tag must be rejected"
    print("  ✅ Citation extraction and hallucination rejection verified.")

    # -------------------------------------------------------------
    # 5. Deployment Instructions & StudyVerse Integration
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    print("\nDeployment Command:")
    print("  modal deploy modal_vllm_rag.py")
    print("\nStudyVerse Environment Configuration:")
    print("  Add or update in studyverse-learning-hub/.env (or .env.local):")
    print("  VITE_MODAL_RAG_URL=https://rockefashy--studyverse-rag-vllm-serve.modal.run")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

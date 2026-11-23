"""
rag_retriever.py - Hybrid Metadata-Scoped Retriever for StudyVerse

Loads data/rag-corpus.json and encodes `searchableText` using BAAI/bge-small-en-v1.5.
Caches normalized float32 embeddings matrix keyed by corpus SHA-256 hash.
Pre-filters candidates strictly using student profile (level, stream, targetExams, enrolledSubjects)
and optional subject_filter to prevent off-stream/off-grade leakage into context.
Ranks candidates via hybrid scoring:
  score = 0.80 * dense_cosine_sim + 0.20 * keyword_symbol_overlap
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Common English stopwords to ignore in keyword extraction
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don",
    "should", "now", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "how", "why", "i", "me", "my", "myself",
    "we", "our", "ours", "you", "your", "yours", "yourself", "he", "him", "his",
    "she", "her", "hers", "it", "its", "they", "them", "their", "give", "tell", "explain",
    "find", "solve", "calculate", "write", "state", "show"
}

# Regex to capture math/formula tokens (LaTeX commands, operators, symbols)
MATH_SYMBOL_PATTERN = re.compile(
    r"\\[a-zA-Z]+|[+\-*/=^<>_()\[\]{}|\\!]+|[a-zA-Z0-9]+"
)


def extract_keywords_and_symbols(text: str) -> List[str]:
    """
    Extracts significant alphanumeric keywords and math/LaTeX formula symbols from text.
    """
    if not text:
        return []
    tokens = MATH_SYMBOL_PATTERN.findall(text)
    meaningful = []
    for t in tokens:
        tl = t.lower()
        # Keep LaTeX commands like \frac, \sqrt, operators like =, +, -, and non-stopwords
        if t.startswith("\\") or len(t) == 1 and not t.isalnum():
            meaningful.append(tl)
        elif tl not in STOPWORDS and len(tl) > 1:
            meaningful.append(tl)
    return meaningful


class RAGRetriever:
    """
    Hybrid Metadata-Scoped Retriever.
    - Dense embeddings: BAAI/bge-small-en-v1.5
    - Persistent caching keyed by SHA-256 hash of data/rag-corpus.json
    - Metadata-scoped pre-filtering
    - Hybrid ranking: 0.80 dense + 0.20 keyword/formula-symbol overlap
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        # Resolve corpus path
        if corpus_path is None:
            # Try data/rag-corpus.json, then rag-corpus.json
            if os.path.exists("data/rag-corpus.json"):
                corpus_path = "data/rag-corpus.json"
            elif os.path.exists("rag-corpus.json"):
                corpus_path = "rag-corpus.json"
            else:
                corpus_path = os.path.join(os.path.dirname(__file__), "data", "rag-corpus.json")

        self.corpus_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"RAG corpus file not found at: {self.corpus_path}")

        # Resolve cache dir
        if cache_dir is None:
            modal_vol_cache = "/root/checkpoints/rag_cache"
            if os.path.exists("/root/checkpoints"):
                cache_dir = modal_vol_cache
            else:
                cache_dir = os.environ.get(
                    "RAG_CACHE_DIR",
                    os.path.join(os.path.dirname(self.corpus_path), "cache"),
                )
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model_name = model_name
        self.model = None

        # Load corpus & compute SHA-256 hash
        self.corpus: List[Dict[str, Any]] = []
        self.corpus_hash: str = ""
        self.embeddings: Optional[np.ndarray] = None
        self.corpus_by_id: Dict[str, Dict[str, Any]] = {}

        self._load_corpus()
        self._load_or_compute_embeddings()

    def _load_corpus(self):
        """Loads corpus JSON and computes SHA-256 hash."""
        with open(self.corpus_path, "rb") as f:
            raw_bytes = f.read()
        self.corpus_hash = hashlib.sha256(raw_bytes).hexdigest()
        self.corpus = json.loads(raw_bytes.decode("utf-8"))
        self.corpus_by_id = {item["id"]: item for item in self.corpus}
        print(f"📖 Loaded {len(self.corpus)} corpus chunks from {self.corpus_path} (SHA-256: {self.corpus_hash[:12]}...)")

    def _get_model(self):
        """Lazy loader for SentenceTransformer to allow fast startup if embeddings exist."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"⚡ Loading SentenceTransformer model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _load_or_compute_embeddings(self):
        """
        Loads cached embeddings if SHA-256 hash matches, otherwise encodes searchableText
        and saves normalized embeddings matrix.
        """
        cache_file = os.path.join(self.cache_dir, f"corpus_embeddings_{self.corpus_hash}.npy")

        if os.path.exists(cache_file):
            try:
                emb = np.load(cache_file)
                if emb.shape[0] == len(self.corpus):
                    self.embeddings = emb.astype(np.float32)
                    print(f"✅ Loaded cached embeddings from {cache_file} (shape: {self.embeddings.shape})")
                    return
                else:
                    print(f"⚠️ Cache count mismatch ({emb.shape[0]} vs {len(self.corpus)}). Recomputing...")
            except Exception as e:
                print(f"⚠️ Failed to load cached embeddings: {e}. Recomputing...")

        # Compute embeddings
        model = self._get_model()
        texts = [item.get("searchableText", item.get("title", "")) for item in self.corpus]
        print(f"🔄 Encoding {len(texts)} chunks with {self.model_name}...")
        emb = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(emb, dtype=np.float32)

        # Save to persistent cache
        try:
            np.save(cache_file, self.embeddings)
            print(f"💾 Saved normalized embeddings to {cache_file}")
        except Exception as e:
            print(f"⚠️ Could not write cache file to {cache_file}: {e}")

    def matches_profile(
        self,
        item: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        subject_filter: Optional[str] = None,
    ) -> bool:
        """
        Strictly checks whether a corpus item matches the student's profile & subject filter.
        Prevents off-stream/off-grade leakage into context.
        """
        item_stream = (item.get("stream") or "").strip().lower()
        item_level = (item.get("level") or "").strip().lower()
        item_subject = (item.get("subject") or "").strip().lower()

        # 1. Subject filter (highest specificity if provided)
        if subject_filter and subject_filter.strip():
            sf = subject_filter.strip().lower()
            if sf not in item_subject and item_subject not in sf:
                return False

        if not profile:
            return True

        # 2. Stream filter
        prof_stream = str(profile.get("stream") or "").strip().lower()
        if prof_stream:
            # Map common stream aliases
            if prof_stream in ("commerce", "commercial"):
                if item_stream != "commerce":
                    return False
            elif prof_stream in ("humanities", "arts"):
                if item_stream != "humanities":
                    return False
            elif prof_stream in ("science", "stem"):
                if item_stream != "science":
                    return False
            elif prof_stream in ("ap", "advanced placement"):
                if item_stream != "ap":
                    return False
            elif prof_stream in ("testprep", "test prep", "sat", "act"):
                if item_stream != "testprep":
                    return False
            else:
                if prof_stream != item_stream:
                    return False

        # 3. Level / Grade filter
        prof_level = str(profile.get("level") or "").strip().lower()
        if prof_level:
            # Check 11 / 12
            if "11" in prof_level or "eleven" in prof_level:
                if item_level not in ("class_11", "cbse", "icse"):
                    return False
            elif "12" in prof_level or "twelve" in prof_level:
                if item_level != "class_12":
                    return False
            elif "sat" in prof_level:
                if item_level != "sat":
                    return False
            elif "act" in prof_level:
                if item_level != "act":
                    return False
            elif "ap" in prof_level:
                if item_level != "ap":
                    return False
            else:
                if prof_level != item_level:
                    return False

        # 4. Target exams filter (e.g. SAT, ACT, AP, JEE, NEET, CBSE)
        target_exams = profile.get("targetExams")
        if target_exams and isinstance(target_exams, list) and len(target_exams) > 0:
            exams_lower = [str(e).strip().lower() for e in target_exams if str(e).strip()]
            if exams_lower:
                exam_matched = False
                for ex in exams_lower:
                    if "sat" in ex and (item_level == "sat" or "sat" in item_subject or item_stream == "testprep"):
                        exam_matched = True
                        break
                    if "act" in ex and (item_level == "act" or "act" in item_subject or item_stream == "testprep"):
                        exam_matched = True
                        break
                    if "ap" in ex and (item_level == "ap" or item_stream == "ap" or "ap" in item_subject):
                        exam_matched = True
                        break
                    if "cbse" in ex and (item_level in ("cbse", "class_11", "class_12")):
                        exam_matched = True
                        break
                    if "icse" in ex and (item_level in ("icse", "class_11", "class_12")):
                        exam_matched = True
                        break
                    if ("jee" in ex or "neet" in ex) and item_stream == "science":
                        exam_matched = True
                        break
                if not exam_matched:
                    return False

        # 5. Enrolled subjects filter
        enrolled = profile.get("enrolledSubjects")
        if enrolled and isinstance(enrolled, list) and len(enrolled) > 0:
            enrolled_lower = [str(s).strip().lower() for s in enrolled if str(s).strip()]
            if enrolled_lower:
                subj_matched = any(
                    es in item_subject or item_subject in es
                    for es in enrolled_lower
                )
                if not subj_matched:
                    return False

        return True

    def calculate_keyword_overlap(self, query: str, candidate: Dict[str, Any]) -> float:
        """
        Calculates normalized keyword and formula-symbol overlap [0.0, 1.0].
        Boosts exact matches in formulaOrRule and title.
        """
        query_tokens = set(extract_keywords_and_symbols(query))
        if not query_tokens:
            return 0.0

        searchable_tokens = set(extract_keywords_and_symbols(candidate.get("searchableText", "")))
        formula_tokens = set(extract_keywords_and_symbols(candidate.get("formulaOrRule", "")))
        title_tokens = set(extract_keywords_and_symbols(candidate.get("title", "")))

        overlap_searchable = len(query_tokens & searchable_tokens) / len(query_tokens)
        overlap_formula = len(query_tokens & formula_tokens) / len(query_tokens)
        overlap_title = len(query_tokens & title_tokens) / len(query_tokens)

        # Weighted combination of overlaps
        overlap_score = 0.50 * overlap_searchable + 0.30 * overlap_formula + 0.20 * overlap_title
        return float(min(1.0, max(0.0, overlap_score)))

    def retrieve(
        self,
        query: str,
        profile: Optional[Dict[str, Any]] = None,
        subject_filter: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching query with strict metadata pre-filtering
        and hybrid dense (0.80) + keyword/symbol overlap (0.20) ranking.
        """
        if not query or not query.strip():
            return []

        # 1. Pre-filter candidates strictly
        candidate_indices = [
            i for i, item in enumerate(self.corpus)
            if self.matches_profile(item, profile, subject_filter)
        ]

        # Graceful fallback: if strict filter left 0 candidates, try relaxing targetExams/enrolledSubjects
        # while keeping stream/level isolation
        if not candidate_indices and profile:
            relaxed_profile = {"stream": profile.get("stream"), "level": profile.get("level")}
            candidate_indices = [
                i for i, item in enumerate(self.corpus)
                if self.matches_profile(item, relaxed_profile, subject_filter)
            ]

        # If still 0 and subject_filter was used, try subject_filter alone
        if not candidate_indices and subject_filter:
            candidate_indices = [
                i for i, item in enumerate(self.corpus)
                if self.matches_profile(item, None, subject_filter)
            ]

        if not candidate_indices:
            # No eligible candidates found
            return []

        # 2. Dense retrieval scoring
        model = self._get_model()
        query_emb = model.encode(query, normalize_embeddings=True)
        query_vec = np.array(query_emb, dtype=np.float32)

        # Dot product with candidate embeddings (already normalized)
        candidate_embeddings = self.embeddings[candidate_indices]
        dense_scores = np.dot(candidate_embeddings, query_vec)

        # 3. Hybrid scoring: 0.80 dense + 0.20 keyword/formula-symbol overlap
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        for idx_in_subset, global_idx in enumerate(candidate_indices):
            candidate = self.corpus[global_idx]
            dense_sim = float(dense_scores[idx_in_subset])
            dense_clamped = max(0.0, min(1.0, dense_sim))

            overlap_score = self.calculate_keyword_overlap(query, candidate)
            hybrid_score = (0.80 * dense_clamped) + (0.20 * overlap_score)

            chunk_result = {
                "id": candidate["id"],
                "score": round(hybrid_score, 4),
                "dense_score": round(dense_clamped, 4),
                "overlap_score": round(overlap_score, 4),
                "title": candidate.get("title", ""),
                "subject": candidate.get("subject", ""),
                "stream": candidate.get("stream", ""),
                "level": candidate.get("level", ""),
                "topic": candidate.get("topic", candidate.get("title", "")),
                "formulaOrRule": candidate.get("formulaOrRule", ""),
                "derivationOrExplanation": candidate.get("derivationOrExplanation", ""),
                "variablesOrTerms": candidate.get("variablesOrTerms", {}),
                "flashcard": candidate.get("flashcard", {}),
                "text": candidate.get("searchableText", ""),
            }
            scored_candidates.append((hybrid_score, chunk_result))

        # Sort descending by hybrid score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored_candidates[:top_k]]

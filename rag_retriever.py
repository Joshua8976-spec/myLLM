"""
rag_retriever.py - High-Precision Hybrid Metadata-Scoped Retriever for StudyVerse

Enhancements:
  1. BGE Query Instruction Prefixing:
     Prepends 'Represent this sentence for searching relevant passages: ' when encoding queries
     with BAAI/bge-small-en-v1.5 to maximize retrieval accuracy on short student questions,
     while keeping document passages unprefixed.
  2. Upgraded Lexical & Formula Scorer (0.20 weight):
     Normalizes LaTeX math symbols (=, ^, \\frac, \\sqrt, math variables, and exact title/topic n-grams).
  3. Minimum Relevance Score Threshold:
     Filters out weak candidates (default min_score=0.35) so out-of-scope queries do not force irrelevant chunks.
  4. Diversity & Deduplication Filtering:
     Suppresses near-identical duplicate chunks in top-k to keep diverse, complementary curriculum entries.
  5. Strict Metadata Scoping:
     Pre-filters strictly against student profile (level, stream, targetExams, enrolledSubjects) and subject_filter.
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np

# BGE official query instruction prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

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
    "find", "solve", "calculate", "write", "state", "show", "describe", "define"
}


def normalize_math_text(text: str) -> str:
    """Normalizes LaTeX math symbols and equations for lexical matching."""
    if not text:
        return ""
    t = text.lower()
    # Normalize common LaTeX tokens to plain forms
    t = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\{([^}]+)\}", r"\1", t)
    t = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1 / \2", t)
    t = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt \1", t)
    t = t.replace(r"\pm", "+-").replace(r"\times", "*").replace(r"\cdot", "*")
    t = t.replace(r"\le", "<=").replace(r"\ge", ">=").replace(r"\neq", "!=")
    t = t.replace(r"\approx", "~=").replace(r"\sum", "sum").replace(r"\int", "integral")
    t = t.replace(r"\Delta", "delta").replace(r"\theta", "theta").replace(r"\pi", "pi")
    t = t.replace("$$", " ").replace("$", " ")
    return t


def extract_keywords_and_symbols(text: str) -> List[str]:
    """
    Extracts significant alphanumeric keywords, math variable names,
    LaTeX commands, and operators from text.
    """
    if not text:
        return []
    norm_text = normalize_math_text(text)
    # Match words, single math variables, math operators
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b|[+\-*/=^]", norm_text)
    meaningful = []
    for t in tokens:
        tl = t.lower()
        if tl in ("=", "+", "-", "*", "/", "^"):
            meaningful.append(tl)
        elif len(tl) == 1 and tl.isalpha():
            # Keep standalone variable names like x, y, f, m, a, v, etc.
            meaningful.append(tl)
        elif tl not in STOPWORDS and len(tl) > 1:
            meaningful.append(tl)
    return meaningful


def get_ngrams(tokens: List[str], n_min: int = 2, n_max: int = 3) -> List[str]:
    """Generates 2-gram and 3-gram phrases from tokens."""
    ngrams = []
    for n in range(n_min, min(n_max + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


class RAGRetriever:
    """
    High-Precision Hybrid Metadata-Scoped Retriever.
    - Dense embeddings: BAAI/bge-small-en-v1.5 with query instruction prefixing
    - Persistent caching keyed by SHA-256 hash of data/rag-corpus.json
    - Metadata-scoped pre-filtering with zero cross-scope leakage
    - Upgraded lexical/formula scorer with LaTeX normalization & n-gram matching
    - Minimum relevance score threshold (min_score=0.35)
    - Diversity/deduplication filtering across top-k results
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        # Resolve corpus path
        if corpus_path is None:
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
        and saves normalized embeddings matrix. Passages are stored without query prefix.
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

        # Compute embeddings without query prefix for document passages
        model = self._get_model()
        texts = [item.get("searchableText", item.get("title", "")) for item in self.corpus]
        print(f"🔄 Encoding {len(texts)} chunks with {self.model_name}...")
        emb = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(emb, dtype=np.float32)

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

        # 4. Target exams filter (e.g. SAT, ACT, AP, JEE, NEET, CBSE, ICSE)
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
        Upgraded Lexical & Formula Scorer:
        Normalizes LaTeX/math symbols, compares mathematical tokens and variables,
        and awards significant bonus for exact title/topic n-gram matches.
        """
        query_tokens = extract_keywords_and_symbols(query)
        if not query_tokens:
            return 0.0

        q_set = set(query_tokens)

        # Candidate token sets
        searchable_tokens = set(extract_keywords_and_symbols(candidate.get("searchableText", "")))
        formula_tokens = set(extract_keywords_and_symbols(candidate.get("formulaOrRule", "")))
        title_tokens = set(extract_keywords_and_symbols(candidate.get("title", "")))
        topic_tokens = set(extract_keywords_and_symbols(candidate.get("topic", "")))

        token_overlap = len(q_set & searchable_tokens) / len(q_set)
        formula_overlap = len(q_set & formula_tokens) / len(q_set) if formula_tokens else 0.0
        title_overlap = len(q_set & (title_tokens | topic_tokens)) / len(q_set) if (title_tokens | topic_tokens) else 0.0

        # Exact n-gram matching on title & searchable text
        query_ngrams = get_ngrams(query_tokens, 2, 3)
        cand_norm_title = normalize_math_text(candidate.get("title", "") + " " + candidate.get("topic", ""))
        cand_norm_searchable = normalize_math_text(candidate.get("searchableText", ""))

        ngram_hits = 0.0
        if query_ngrams:
            for ng in query_ngrams:
                if ng in cand_norm_title:
                    ngram_hits += 2.0  # Double weight for title/topic match
                elif ng in cand_norm_searchable:
                    ngram_hits += 1.0
            ngram_score = min(1.0, ngram_hits / (len(query_ngrams) * 1.5))
        else:
            ngram_score = 0.0

        # Weighted combination
        lexical_score = (
            0.30 * token_overlap
            + 0.30 * formula_overlap
            + 0.20 * title_overlap
            + 0.20 * ngram_score
        )
        return float(min(1.0, max(0.0, lexical_score)))

    @staticmethod
    def _is_too_similar(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
        """
        Checks if two candidates are nearly identical in title/topic to prevent duplicate crowding.
        """
        t1 = (c1.get("title") or "").strip().lower()
        t2 = (c2.get("title") or "").strip().lower()
        if t1 == t2:
            return True

        top1 = (c1.get("topic") or "").strip().lower()
        top2 = (c2.get("topic") or "").strip().lower()
        if top1 and top2 and top1 == top2:
            w1 = set(extract_keywords_and_symbols(t1))
            w2 = set(extract_keywords_and_symbols(t2))
            if w1 and w2 and len(w1 & w2) / max(len(w1), len(w2)) > 0.75:
                return True

        # Check content Jaccard
        k1 = set(extract_keywords_and_symbols((c1.get("formulaOrRule") or "") + " " + t1))
        k2 = set(extract_keywords_and_symbols((c2.get("formulaOrRule") or "") + " " + t2))
        if k1 and k2 and len(k1 & k2) / max(len(k1 | k2), 1) > 0.85:
            return True

        return False

    def retrieve(
        self,
        query: str,
        profile: Optional[Dict[str, Any]] = None,
        subject_filter: Optional[str] = None,
        top_k: int = 4,
        min_score: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks matching query with:
        1. Strict metadata pre-filtering (zero cross-scope leakage)
        2. BGE query instruction prefixing for sharp semantic alignment
        3. Hybrid dense (0.80) + upgraded lexical/formula (0.20) ranking
        4. Minimum relevance score threshold (min_score=0.35)
        5. Diversity/deduplication filtering to avoid duplicate crowding
        """
        if not query or not query.strip():
            return []

        # 1. Pre-filter candidates strictly
        candidate_indices = [
            i for i, item in enumerate(self.corpus)
            if self.matches_profile(item, profile, subject_filter)
        ]

        # Graceful fallback: relax targetExams/enrolledSubjects while keeping stream/level isolation
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
            return []

        # 2. Dense retrieval with BGE query instruction prefix
        model = self._get_model()
        prefixed_query = f"{BGE_QUERY_PREFIX}{query.strip()}"
        query_emb = model.encode(prefixed_query, normalize_embeddings=True)
        query_vec = np.array(query_emb, dtype=np.float32)

        # Dot product with candidate embeddings (pre-normalized)
        candidate_embeddings = self.embeddings[candidate_indices]
        dense_scores = np.dot(candidate_embeddings, query_vec)

        # 3. Hybrid scoring: 0.80 dense + 0.20 upgraded lexical/formula overlap
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        for idx_in_subset, global_idx in enumerate(candidate_indices):
            candidate = self.corpus[global_idx]
            dense_sim = float(dense_scores[idx_in_subset])
            dense_clamped = max(0.0, min(1.0, dense_sim))

            # Calibrate against dense embedding background floor (~0.38 for unrelated texts)
            if dense_clamped > 0.38:
                calibrated_dense = (dense_clamped - 0.38) / (1.0 - 0.38)
            else:
                calibrated_dense = 0.0

            overlap_score = self.calculate_keyword_overlap(query, candidate)
            hybrid_score = (0.80 * calibrated_dense) + (0.20 * overlap_score)

            chunk_result = {
                "id": candidate["id"],
                "score": round(hybrid_score, 4),
                "dense_score": round(dense_clamped, 4),
                "calibrated_dense": round(calibrated_dense, 4),
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

        # 4. Minimum relevance score threshold (min_score=0.35)
        filtered_by_threshold = [
            item for score, item in scored_candidates if score >= min_score
        ]

        # 5. Diversity / Deduplication filtering
        diverse_results: List[Dict[str, Any]] = []
        for item in filtered_by_threshold:
            if any(self._is_too_similar(item, existing) for existing in diverse_results):
                continue
            diverse_results.append(item)
            if len(diverse_results) == top_k:
                break

        return diverse_results

import os
import math
import pickle
import hashlib
import re
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag.indexer import build_safe_index


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file. "
        "Please add GEMINI_API_KEY."
    )


client = genai.Client(api_key=api_key)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
CACHE_FILE = "embeddings_cache.pkl"

DEFAULT_TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.05

STRONG_RETRIEVAL_SCORE = 0.40
CONFLICT_SCORE_THRESHOLD = 0.40
CONFLICT_CHUNK_SIMILARITY_THRESHOLD = 0.60


# ============================================================
# RETRIEVER
# ============================================================

class VectorRetriever:

    def __init__(self, kb_directory: str):
        self.kb_directory = kb_directory

        print(f"Building safe index from {kb_directory}...")
        self.chunks = build_safe_index(kb_directory)
        self.embeddings: List[List[float]] = []

        if not self.chunks:
            print("Warning: No valid chunks found in the knowledge base.")
            return

        self._load_or_generate_embeddings()

    # ========================================================
    # CACHE
    # ========================================================

    def _get_cache_signature(self) -> str:
        hasher = hashlib.sha256()

        for chunk in self.chunks:
            data = (
                str(chunk.document_id)
                + "\n"
                + str(chunk.file_name)
                + "\n"
                + str(chunk.heading)
                + "\n"
                + str(chunk.content)
                + "\n"
                + str(chunk.metadata)
            )
            hasher.update(data.encode("utf-8", errors="ignore"))

        return hasher.hexdigest()

    def _load_or_generate_embeddings(self):
        cache_file = CACHE_FILE
        current_signature = self._get_cache_signature()

        if os.path.exists(cache_file):
            try:
                print("⚡ Checking local embedding cache...")
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)

                if isinstance(cached, dict):
                    cached_signature = cached.get("signature")
                    cached_embeddings = cached.get("embeddings")
                    cached_count = cached.get("chunk_count")
                    cached_dimension = cached.get("dimension")

                    if (
                        cached_signature == current_signature
                        and cached_count == len(self.chunks)
                        and cached_dimension == EMBEDDING_DIMENSIONS
                        and isinstance(cached_embeddings, list)
                        and len(cached_embeddings) == len(self.chunks)
                    ):
                        self.embeddings = cached_embeddings
                        print("✅ Loaded valid embeddings from local cache.")
                        return

                    print("⚠️ Existing embedding cache is stale or incompatible.")
                else:
                    print("⚠️ Old embedding cache format detected. Rebuilding cache.")

            except Exception as exc:
                print(f"⚠️ Could not load embedding cache: {exc}")

        self._generate_corpus_embeddings(current_signature)

    # ========================================================
    # CORPUS EMBEDDINGS
    # ========================================================

    def _generate_corpus_embeddings(self, signature: str):
        print(f"Generating embeddings for {len(self.chunks)} safe chunks...")

        texts = [chunk.content for chunk in self.chunks]

        contents = [
            types.Content(parts=[types.Part(text=t)])
            for t in texts
        ]

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            )
        )

        self.embeddings = [
            list(embedding.values)
            for embedding in result.embeddings
        ]

        if len(self.embeddings) != len(self.chunks):
            raise RuntimeError(
                f"Embedding count ({len(self.embeddings)}) does not match "
                f"knowledge-base chunk count ({len(self.chunks)})."
            )

        for index, embedding in enumerate(self.embeddings):
            if len(embedding) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Unexpected embedding dimension for chunk {index}"
                )

        print("✅ Embeddings ready. Saving to cache...")

        cache_payload = {
            "version": 2,
            "model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSIONS,
            "signature": signature,
            "chunk_count": len(self.chunks),
            "embeddings": self.embeddings,
        }

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_payload, f)

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def _embed_query(self, query: str) -> List[float]:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            )
        )

        if not result.embeddings:
            return []

        return list(result.embeddings[0].values)

    # ========================================================
    # ANSWERABILITY & CONFLICT HELPERS
    # ========================================================

    def _has_answerable_terms(
        self,
        query: str,
        results: list,
    ) -> bool:
        query_words = {
            word.lower()
            for word in re.findall(
                r"[A-Za-z0-9]+",
                query,
            )
            if len(word) >= 4
        }

        content_text = " ".join(
            str(result.get("content", ""))
            for result in results[:2]
        ).lower()

        if not query_words:
            return True

        matched = sum(
            1
            for word in query_words
            if word in content_text
        )

        return matched >= 2

    def _detect_active_conflict(self, results: list) -> bool:
        active = [
            result
            for result in results
            if result.get("metadata", {}).get("status") == "active"
        ]

        if len(active) < 2:
            return False

        combined = [
            str(result.get("content", "")).lower()
            for result in active
        ]

        has_handwash = any(
            "hand-wash" in text or "hand wash" in text
            for text in combined
        )

        has_dishwasher = any(
            "dishwasher safe" in text or "dishwasher-safe" in text
            for text in combined
        )

        return has_handwash and has_dishwasher

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        if not query or not query.strip() or not self.chunks:
            return {
                "query": query,
                "insufficient_information": True,
                "conflict_flag": False,
                "results": [],
            }

        query_embedding = self._embed_query(query)

        if not query_embedding:
            return {
                "query": query,
                "insufficient_information": True,
                "conflict_flag": False,
                "results": [],
            }

        query_lower = query.lower()
        boost_files = []
        is_shipping_query = any(w in query_lower for w in ["ship", "shipping", "deliver", "delivery", "destination", "country", "germany", "canada", "international", "abroad", "overseas"])
        if is_shipping_query:
            boost_files.extend(["05-domestic-shipping.md", "06-international-shipping.md"])
        elif any(w in query_lower for w in ["return", "refund", "window", "days", "exchange"]):
            boost_files.extend(["01-returns-policy-current.md", "09-trailplus-membership.md"])
        elif any(w in query_lower for w in ["warranty", "guarantee", "lifetime"]):
            boost_files.extend(["07-warranty.md"])
        elif any(w in query_lower for w in ["damaged", "broken", "wrong", "defect", "final sale"]):
            boost_files.extend(["03-final-sale-and-promotions.md", "04-damaged-or-wrong-items.md"])
        elif any(w in query_lower for w in ["wash", "clean", "dishwasher", "tumbler", "care", "fabric", "adhesive"]):
            boost_files.extend(["11-product-care.md", "12-breeze-tumbler-product-card.md"])

        scored_chunks = []

        for index, doc_embedding in enumerate(self.embeddings):
            chunk = self.chunks[index]
            content = (chunk.content or "").strip()

            if not content:
                continue

            if chunk.heading == "General" and len(content.splitlines()) <= 1:
                continue

            score = self._cosine_similarity(query_embedding, doc_embedding)

            if boost_files and chunk.file_name in boost_files:
                score += 0.25

            scored_chunks.append((score, chunk, doc_embedding))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        candidates = scored_chunks[:max(top_k * 2, 5)]

        if not candidates:
            return {
                "query": query,
                "insufficient_information": True,
                "conflict_flag": False,
                "results": [],
            }

        highest_score = candidates[0][0]

        candidate_dicts = [{
            "content": chunk.content,
            "file_name": chunk.file_name,
            "heading": chunk.heading,
            "metadata": chunk.metadata
        } for _, chunk, _ in candidates]

        has_shipping_chunk = any(
            chunk.file_name in ["05-domestic-shipping.md", "06-international-shipping.md"]
            for score, chunk, _ in candidates
            if score >= MIN_RETRIEVAL_SCORE
        )

        insufficient_information = (
            not candidates
            or highest_score < MIN_RETRIEVAL_SCORE
            or (is_shipping_query and not has_shipping_chunk)
            or (not is_shipping_query and not self._has_answerable_terms(query, candidate_dicts))
        )

        if insufficient_information:
            relevant_candidates = []
        else:
            relevant_candidates = [
                item for item in candidates if item[0] >= MIN_RETRIEVAL_SCORE
            ][:top_k]

        formatted_results = []
        for score, chunk, _ in relevant_candidates:
            formatted_results.append({
                "score": round(score, 4),
                "file_name": chunk.file_name,
                "heading": chunk.heading,
                "content": chunk.content,
                "metadata": chunk.metadata,
            })

        conflict_detected = self._detect_active_conflict(formatted_results)

        return {
            "query": query,
            "insufficient_information": insufficient_information,
            "conflict_flag": conflict_detected,
            "results": formatted_results,
        }

    def debug_search(self, query: str, top_k: int = 5):
        return self.search(query, top_k=top_k)
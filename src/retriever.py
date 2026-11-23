import json
import re
import time
from typing import Any, Literal

import chromadb
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

from src import config

logger = config.get_logger("retriever")

RetrievalMode = Literal["vector", "hybrid", "hybrid_rerank"]

# Deterministic domain synonym map (zero LLM latency)
SYNONYM_MAP: dict[str, str] = {
    r"\bsupple\b": "supplementary examination",
    r"\bsupply\b": "supplementary examination",
    r"\bbacklog(s)?\b": "failed course grade F de-registered",
    r"\byear\s*lag\b": "promotion criteria minimum credits academic break",
    r"\bbranch\s*change\b": "change of branch",
    r"\bcg\b": "CGPA",
    r"\bsg\b": "SGPA",
    r"\bsem\b": "semester",
    r"\bmidsem\b": "mid-semester examination",
    r"\bendsem\b": "end-semester examination",
    r"\battnd\b": "attendance",
}

# Basic prompt-injection patterns to block before retrieval
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"reveal\s+(your\s+)?(system\s+prompt|api\s+key)",
    r"act\s+as\s+a\s+different",
]


def tokenize(text: str) -> list[str]:
    """Regex tokenizer that preserves decimals/acronyms (e.g., 6.5, b.tech) and strips punctuation."""
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", text.lower())


def rewrite_query(query: str) -> str:
    """Expand informal student slang into formal ordinance terminology."""
    rewritten = query.strip()
    for pattern, replacement in SYNONYM_MAP.items():
        if re.search(pattern, rewritten, flags=re.IGNORECASE):
            rewritten = f"{rewritten} ({replacement})"
    return rewritten


def is_prompt_injection(query: str) -> bool:
    """Detect obvious prompt-injection attempts."""
    return any(re.search(p, query, flags=re.IGNORECASE) for p in INJECTION_PATTERNS)


class HybridRetriever:
    """Unified 3-mode retriever: vector, hybrid (BM25 + RRF), and hybrid_rerank."""

    def __init__(self) -> None:
        if not config.CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunk store not found at {config.CHUNKS_PATH}. "
                "Run `python -m src.ingestion` first."
            )

        self.chunks_store: dict[str, dict[str, Any]] = json.loads(
            config.CHUNKS_PATH.read_text(encoding="utf-8")
        )
        if not self.chunks_store:
            raise ValueError("Index is empty. Add at least one valid PDF to data/ and run ingestion.")

        self.doc_ids: list[str] = list(self.chunks_store.keys())

        # 1. Persistent ChromaDB vector collection
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )

        # 2. BM25 keyword index built from the exact same synced chunks
        tokenized_corpus = [tokenize(self.chunks_store[cid]["text"]) for cid in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Lightweight ONNX Cross-Encoder Reranker (lazy-loaded on first rerank call)
        self._ranker: Ranker | None = None
        logger.info("HybridRetriever ready with %d indexed chunks.", len(self.doc_ids))

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            logger.info("Loading FlashRank reranker: %s", config.RERANKER_MODEL)
            self._ranker = Ranker(
                model_name=config.RERANKER_MODEL,
                cache_dir=str(config.FLASHRANK_CACHE_DIR),
            )
        return self._ranker

    def _vector_search(self, query: str, k: int) -> list[str]:
        res = self.collection.query(query_texts=[query], n_results=k)
        return res["ids"][0] if res and res.get("ids") else []

    def _bm25_search(self, query: str, k: int) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.doc_ids[i] for i in ranked_idx if scores[i] > 0]

    @staticmethod
    def _rrf_fuse(
        vector_ids: list[str],
        bm25_ids: list[str],
        k_constant: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vector_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_constant + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_constant + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid_rerank",
        top_k: int = 4,
        candidate_k: int = 12,
    ) -> dict[str, Any]:
        """Run retrieval in the requested mode and return structured chunks + telemetry."""
        start = time.perf_counter()

        if is_prompt_injection(query):
            logger.warning("Blocked potential prompt injection: %s", query[:80])
            return {
                "query": query,
                "rewritten_query": query,
                "mode": mode,
                "blocked": True,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "results": [],
            }

        rewritten = rewrite_query(query)
        pool_size = min(candidate_k, len(self.doc_ids))
        final_k = min(top_k, len(self.doc_ids))

        scored_ids: list[tuple[str, float]]

        if mode == "vector":
            vec_ids = self._vector_search(rewritten, k=final_k)
            scored_ids = [(cid, round(1.0 / (rank + 1), 4)) for rank, cid in enumerate(vec_ids)]

        elif mode == "hybrid":
            vec_ids = self._vector_search(rewritten, k=pool_size)
            bm25_ids = self._bm25_search(rewritten, k=pool_size)
            scored_ids = self._rrf_fuse(vec_ids, bm25_ids)[:final_k]

        elif mode == "hybrid_rerank":
            vec_ids = self._vector_search(rewritten, k=pool_size)
            bm25_ids = self._bm25_search(rewritten, k=pool_size)
            fused = self._rrf_fuse(vec_ids, bm25_ids)[:pool_size]
            candidate_ids = [cid for cid, _ in fused]

            passages = [
                {"id": i, "text": self.chunks_store[cid]["text"]}
                for i, cid in enumerate(candidate_ids)
            ]
            reranked = self.ranker.rerank(RerankRequest(query=rewritten, passages=passages))
            scored_ids = [
                (candidate_ids[r["id"]], round(float(r["score"]), 4))
                for r in reranked[:final_k]
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        results = []
        for cid, score in scored_ids:
            chunk = self.chunks_store[cid]
            results.append(
                {
                    "id": cid,
                    "score": score,
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "has_table": chunk["metadata"].get("has_table", False),
                    "text": chunk["text"],
                }
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "query": query,
            "rewritten_query": rewritten,
            "mode": mode,
            "blocked": False,
            "latency_ms": latency_ms,
            "results": results,
        }


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "explain the supple rule?"

    for m in ("vector", "hybrid", "hybrid_rerank"):
        out = retriever.retrieve(test_query, mode=m, top_k=4)
        pages = [r["page"] for r in out["results"]]
        top_score = out["results"][0]["score"] if out["results"] else 0.0
        print(
            f"\n[{m.upper()}] {out['latency_ms']}ms | Rewritten: {out['rewritten_query']}\n"
            f"  Pages: {pages} | Top Score: {top_score}"
        )
        if out["results"]:
            preview = out["results"][0]["text"].replace("\n", " ")[:140]
            print(f"  #1 Preview: \"{preview}...\"")
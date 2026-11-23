import argparse
import json
from pathlib import Path
from typing import Any

from src import config
from src.generator import RAGPipeline
from src.retriever import HybridRetriever, RetrievalMode

GOLDEN_PATH = config.BASE_DIR / "eval" / "golden_set.json"
RESULTS_PATH = config.BASE_DIR / "eval" / "benchmark_results.json"


def chunk_matches(text: str, expected_terms: list[str]) -> bool:
    """A chunk counts as relevant if it contains at least half of the expected ground-truth terms."""
    lowered = text.lower()
    hits = sum(1 for term in expected_terms if term.lower() in lowered)
    threshold = max(1, (len(expected_terms) + 1) // 2)
    return hits >= threshold


def evaluate_retrieval(
    retriever: HybridRetriever,
    cases: list[dict[str, Any]],
    top_k: int = 4,
) -> dict[str, dict[str, float]]:
    modes: tuple[RetrievalMode, ...] = ("vector", "hybrid", "hybrid_rerank")
    summary: dict[str, dict[str, float]] = {}

    for mode in modes:
        hits = 0
        rr_sum = 0.0
        total_ms = 0.0

        for case in cases:
            out = retriever.retrieve(case["query"], mode=mode, top_k=top_k)
            total_ms += out["latency_ms"]

            first_relevant_rank = None
            for rank, item in enumerate(out["results"], start=1):
                if chunk_matches(item["text"], case["expected_terms"]):
                    first_relevant_rank = rank
                    break

            if first_relevant_rank is not None:
                hits += 1
                rr_sum += 1.0 / first_relevant_rank

        n = len(cases)
        summary[mode] = {
            f"hit_at_{top_k}": round((hits / n) * 100, 1),
            "mrr": round(rr_sum / n, 3),
            "avg_latency_ms": round(total_ms / n, 2),
        }

    return summary


def evaluate_hallucination_traps(traps: list[str]) -> dict[str, Any]:
    rag = RAGPipeline()
    passed = 0
    details = []

    for q in traps:
        stream, meta = rag.stream_answer(q, mode="hybrid_rerank", use_cache=False)
        answer = "".join(stream).strip()
        rejected = "could not find" in answer.lower()
        if rejected:
            passed += 1
        details.append({"query": q, "rejected_hallucination": rejected, "answer": answer})

    return {
        "rejection_rate_pct": round((passed / len(traps)) * 100, 1),
        "details": details,
    }


def print_markdown_table(summary: dict[str, dict[str, float]], top_k: int) -> None:
    print("\n### Retrieval Benchmark Results (Copy into README.md)\n")
    print(f"| Pipeline Mode | Hit@{top_k} (%) | MRR | Avg Retrieval Latency (ms) |")
    print("|---|---:|---:|---:|")
    for mode, metrics in summary.items():
        print(
            f"| `{mode}` | {metrics[f'hit_at_{top_k}']}% | "
            f"{metrics['mrr']:.3f} | {metrics['avg_latency_ms']:.2f} ms |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval modes.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also run hallucination-trap checks against Gemini",
    )
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    retriever = HybridRetriever()

    retrieval_summary = evaluate_retrieval(
        retriever, golden["retrieval_cases"], top_k=args.top_k
    )
    print_markdown_table(retrieval_summary, top_k=args.top_k)

    report: dict[str, Any] = {"top_k": args.top_k, "retrieval": retrieval_summary}

    if args.llm:
        print("Running hallucination-trap checks against Gemini...")
        trap_report = evaluate_hallucination_traps(golden["hallucination_traps"])
        report["hallucination_traps"] = trap_report
        print(
            f"Hallucination Rejection Rate: {trap_report['rejection_rate_pct']}%"
        )

    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved full benchmark report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
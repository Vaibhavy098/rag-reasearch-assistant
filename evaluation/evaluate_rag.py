
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGResearchAssistant


DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"

TOP_K = 4

# Common words that contribute little to keyword matching.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "for", "from", "has", "have", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "this", "those", "to", "was", "were",
    "what", "when", "which", "while", "with", "within", "uses",
    "using", "helps", "help", "model", "models",
}


def load_dataset() -> list[dict]:
    """Load the predefined evaluation questions."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("The evaluation dataset must be a non-empty list.")

    return dataset


def normalize_tokens(text: str) -> set[str]:
    """Normalize text into a set of meaningful terms."""

    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = set()

    for word in words:
        if word in STOP_WORDS or len(word) < 2:
            continue

        # Basic normalization to make common word forms comparable.
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]

        tokens.add(word)

    return tokens


def calculate_source_recall(
    expected_sources: list[str],
    retrieved_chunks: list[dict],
) -> float:
    """Calculate the fraction of expected source papers retrieved."""

    expected = set(expected_sources)
    retrieved = {
        chunk.get("source")
        for chunk in retrieved_chunks
        if chunk.get("source")
    }

    if not expected:
        return 0.0

    return len(expected & retrieved) / len(expected)


def calculate_key_point_coverage(
    answer: str,
    key_points: list[str],
    minimum_overlap: float = 0.5,
) -> dict:
    """
    Estimate key-point coverage using lexical overlap.

    A key point passes if at least minimum_overlap of its meaningful
    terms appear in the answer. This is a heuristic, not semantic
    verification or a factual correctness score.
    """

    answer_tokens = normalize_tokens(answer)
    point_results = []

    for point in key_points:
        point_tokens = normalize_tokens(point)

        if not point_tokens:
            point_results.append(
                {
                    "key_point": point,
                    "overlap": 0.0,
                    "covered": False,
                }
            )
            continue

        matched_tokens = point_tokens & answer_tokens
        overlap = len(matched_tokens) / len(point_tokens)

        point_results.append(
            {
                "key_point": point,
                "overlap": round(overlap, 4),
                "covered": overlap >= minimum_overlap,
            }
        )

    covered_count = sum(
        item["covered"] for item in point_results
    )

    score = (
        covered_count / len(point_results)
        if point_results
        else 0.0
    )

    return {
        "score": round(score, 4),
        "covered_count": covered_count,
        "total_key_points": len(point_results),
        "details": point_results,
    }


def evaluate_question(
    assistant: RAGResearchAssistant,
    item: dict,
) -> dict:
    """Run one question through the RAG system and evaluate it."""

    # Keep questions independent. Previous evaluation answers should
    # not influence the retrieval query or generated answer.
    assistant.memory.clear()

    question = item["question"]

    try:
        result = assistant.answer_question(question, top_k=TOP_K)

        retrieved_chunks = result.get("retrieved_chunks", [])
        answer = result.get("answer", "")

        source_recall = calculate_source_recall(
            item["source_papers"],
            retrieved_chunks,
        )

        key_point_result = calculate_key_point_coverage(
            answer,
            item["key_points"],
        )

        return {
            "id": item["id"],
            "question": question,
            "reference_answer": item["reference_answer"],
            "generated_answer": answer,
            "expected_sources": item["source_papers"],
            "retrieved_sources": sorted(
                {
                    chunk.get("source")
                    for chunk in retrieved_chunks
                    if chunk.get("source")
                }
            ),
            "retrieved_chunk_count": len(retrieved_chunks),
            "source_recall": round(source_recall, 4),
            "key_point_coverage": key_point_result,
            "error": None,
        }

    except Exception as error:
        return {
            "id": item.get("id"),
            "question": question,
            "reference_answer": item.get("reference_answer"),
            "generated_answer": None,
            "expected_sources": item.get("source_papers", []),
            "retrieved_sources": [],
            "retrieved_chunk_count": 0,
            "source_recall": 0.0,
            "key_point_coverage": {
                "score": 0.0,
                "covered_count": 0,
                "total_key_points": len(item.get("key_points", [])),
                "details": [],
            },
            "error": f"{type(error).__name__}: {error}",
        }


def summarize_results(results: list[dict]) -> dict:
    """Calculate aggregate metrics, including failed evaluations."""

    total = len(results)

    successful = [
        result for result in results
        if result["error"] is None
    ]

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "total_questions": total,
        "successful_evaluations": len(successful),
        "failed_evaluations": total - len(successful),
        "source_recall_average": average(
            [result["source_recall"] for result in successful]
        ),
        "key_point_coverage_average": average(
            [
                result["key_point_coverage"]["score"]
                for result in successful
            ]
        ),
        "metric_notes": [
            "Source recall measures expected source-paper coverage.",
            "Key-point coverage is based on lexical overlap.",
            "Neither metric independently proves factual correctness.",
            "A failed evaluation is excluded from averages and reported separately.",
        ],
    }


def main() -> None:
    """Evaluate all questions and save a JSON report."""

    dataset = load_dataset()

    print(f"Loaded {len(dataset)} evaluation questions.")
    print(f"Retrieval top_k: {TOP_K}")
    print("Initializing the RAG assistant...")

    assistant = RAGResearchAssistant(memory_turns=4)
    results = []

    for index, item in enumerate(dataset, start=1):
        print(
            f"\n[{index}/{len(dataset)}] "
            f"Question {item['id']}: {item['question']}"
        )

        evaluation = evaluate_question(assistant, item)
        results.append(evaluation)

        if evaluation["error"]:
            print(f"Evaluation failed: {evaluation['error']}")
        else:
            print(
                "Expected source recall: "
                f"{evaluation['source_recall']:.1%}"
            )
            print(
                "Key-point coverage: "
                f"{evaluation['key_point_coverage']['score']:.1%}"
            )

    summary = summarize_results(results)

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "top_k": TOP_K,
        "summary": summary,
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Questions: {summary['total_questions']}")
    print(f"Successful: {summary['successful_evaluations']}")
    print(f"Failed: {summary['failed_evaluations']}")
    print(
        "Average expected source recall: "
        f"{summary['source_recall_average']:.1%}"
    )
    print(
        "Average key-point coverage: "
        f"{summary['key_point_coverage_average']:.1%}"
    )
    print(f"\nDetailed report saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

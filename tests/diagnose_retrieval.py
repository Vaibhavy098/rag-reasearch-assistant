from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search import ResearchPaperSearcher


DIAGNOSTIC_QUESTIONS = [
    {
        "id": 5,
        "question": (
            "What is the Next Sentence Prediction objective "
            "in BERT pretraining?"
        ),
        "expected_sources": {"bert.pdf"},
    },
    {
        "id": 9,
        "question": (
            "Compare the original Transformer encoder-decoder "
            "architecture with BERT's encoder-only architecture."
        ),
        "expected_sources": {
            "attention_is_all_you_need.pdf",
            "bert.pdf",
        },
    },
    {
        "id": 10,
        "question": (
            "Compare GPT-3 in-context few-shot learning with "
            "T5's unified text-to-text framework."
        ),
        "expected_sources": {
            "gpt3_few_shot_learners.pdf",
            "t5.pdf",
        },
    },
]


def main() -> None:
    searcher = ResearchPaperSearcher()
    top_k = 4

    for item in DIAGNOSTIC_QUESTIONS:
        print("\n" + "=" * 80)
        print(f"Question {item['id']}: {item['question']}")
        print("=" * 80)

        results = searcher.search(item["question"], top_k=top_k)

        retrieved_sources = {
            result["source"] for result in results
        }
        missing_sources = (
            item["expected_sources"] - retrieved_sources
        )

        print("\nRetrieved passages:")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Rank {rank} ---")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page_number']}")
            print(
                "Similarity: "
                f"{result['similarity_score']:.4f}"
            )
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text'][:900]}")

        print("\nExpected sources:")
        for source in sorted(item["expected_sources"]):
            print(f"  - {source}")

        print("\nRetrieved source files:")
        for source in sorted(retrieved_sources):
            print(f"  - {source}")

        if missing_sources:
            print("\nMissing expected sources:")
            for source in sorted(missing_sources):
                print(f"  - {source}")
        else:
            print("\nAll expected source files were retrieved.")


if __name__ == "__main__":
    main()



import json
from pathlib import Path


CHUNKS_FILE = Path("data/processed/chunks.jsonl")

EXPECTED_SOURCES = {
    "attention_is_all_you_need.pdf",
    "bert.pdf",
    "gpt3_few_shot_learners.pdf",
    "roberta.pdf",
    "t5.pdf",
}

CHUNK_SIZE = 1000


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Chunk file not found. Run python -m src.chunking first."
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}."
                ) from error

            chunks.append(chunk)

    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        raise AssertionError("No chunks were generated.")

    found_sources = {chunk.get("source") for chunk in chunks}
    missing_sources = EXPECTED_SOURCES - found_sources

    if missing_sources:
        raise AssertionError(
            f"Missing source documents: {sorted(missing_sources)}"
        )

    required_fields = {
        "chunk_id",
        "source",
        "page_number",
        "chunk_index",
        "text",
        "char_count",
    }

    invalid_chunks = []

    for chunk in chunks:
        missing_fields = required_fields - chunk.keys()

        if missing_fields:
            invalid_chunks.append(
                f"{chunk.get('chunk_id', 'unknown')}: "
                f"missing {sorted(missing_fields)}"
            )
            continue

        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            invalid_chunks.append(
                f"{chunk['chunk_id']}: empty text"
            )

        if len(chunk["text"]) > CHUNK_SIZE:
            invalid_chunks.append(
                f"{chunk['chunk_id']}: exceeds chunk size"
            )

        if chunk["char_count"] != len(chunk["text"]):
            invalid_chunks.append(
                f"{chunk['chunk_id']}: incorrect character count"
            )

        if not isinstance(chunk["page_number"], int) or chunk["page_number"] < 1:
            invalid_chunks.append(
                f"{chunk['chunk_id']}: invalid page number"
            )

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        invalid_chunks.append("Duplicate chunk IDs found")

    print(f"Unique source documents: {len(found_sources)}")
    print(f"Invalid chunks or metadata issues: {len(invalid_chunks)}")

    if invalid_chunks:
        for problem in invalid_chunks[:20]:
            print(f"[FAIL] {problem}")

        raise AssertionError("Chunk validation failed.")

    print("[PASS] All five source documents are represented.")
    print("[PASS] Chunk text and metadata checks passed.")


if __name__ == "__main__":
    main()

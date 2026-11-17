
# overlapping character chunks with metadata.

import json
from pathlib import Path


INPUT_FILE = Path("data/processed/pages.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""

    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end

        # Avoid cutting a word when a suitable space is available.
        if proposed_end < len(text):
            search_start = start + int(chunk_size * 0.7)
            boundary = text.rfind(" ", search_start, proposed_end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move backwards to preserve overlap with the previous chunk.
        next_start = max(start + 1, end - overlap)

        # Begin at a word boundary where possible.
        while (
            next_start < len(text)
            and not text[next_start].isspace()
        ):
            next_start += 1

        if next_start <= start or next_start >= end:
            next_start = end

        start = next_start

    return chunks


def load_page_records() -> list[dict]:
    """Load page records from the JSONL file."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run the Day 2 ingestion pipeline first."
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {INPUT_FILE}"
                ) from error

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"Missing or invalid text on line {line_number}."
                )

            records.append(record)

    return records


def create_chunks(page_records: list[dict]) -> list[dict]:
    """Create chunks while preserving source and page metadata."""

    all_chunks = []

    for page in page_records:
        source = page["source"]
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = split_text(page_text)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_id = (
                f"{Path(source).stem}"
                f"_p{page_number:03d}"
                f"_c{chunk_index:03d}"
            )

            all_chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })

    return all_chunks


def save_chunks(chunks: list[dict]) -> None:
    """Save each chunk as one JSON object per line."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )


def main() -> None:
    page_records = load_page_records()
    chunks = create_chunks(page_records)
    save_chunks(chunks)

    source_counts = {}

    for chunk in chunks:
        source = chunk["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Pages processed: {len(page_records)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size setting: {CHUNK_SIZE} characters")
    print(f"Overlap setting: {CHUNK_OVERLAP} characters")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nChunks per document:")

    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    if not chunks:
        print(
            "\n[WARNING] No chunks were generated. "
            "Check whether the extracted pages contain text."
        )


if __name__ == "__main__":
    main()

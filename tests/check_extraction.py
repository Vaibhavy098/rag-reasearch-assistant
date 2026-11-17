
import json
from pathlib import Path


PAGES_FILE = Path("data/processed/pages.jsonl")

EXPECTED_SOURCES = {
    "attention_is_all_you_need.pdf",
    "bert.pdf",
    "gpt3_few_shot_learners.pdf",
    "roberta.pdf",
    "t5.pdf",
}


def main():
    if not PAGES_FILE.exists():
        raise FileNotFoundError(
            "Run the ingestion pipeline before this validation."
        )

    records = []

    with PAGES_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

    found_sources = {record["source"] for record in records}

    print(f"Total page records: {len(records)}")
    print(f"Unique sources found: {len(found_sources)}")

    missing = EXPECTED_SOURCES - found_sources
    unexpected = found_sources - EXPECTED_SOURCES

    if missing:
        print(f"[FAIL] Missing source documents: {sorted(missing)}")

    if unexpected:
        print(f"[WARNING] Unexpected documents: {sorted(unexpected)}")

    invalid_records = [
        record for record in records
        if not record.get("text", "").strip()
        or not isinstance(record.get("page_number"), int)
        or record["page_number"] < 1
    ]

    print(f"Records with missing text or invalid page: {len(invalid_records)}")

    if missing or invalid_records:
        raise AssertionError("Extraction validation failed.")

    print("[PASS] All five source documents are represented.")


if __name__ == "__main__":
    main()

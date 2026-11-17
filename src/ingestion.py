
# extract text per pdf with pymupdf, clean , save.

import json
import re
from pathlib import Path

import fitz  # PyMuPDF


PDF_DIR = Path("data/pdfs")
OUTPUT_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    """Normalize extracted text without removing useful content."""
    # Join words broken by a hyphen at a line break.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Normalize Windows and old-style line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printing control characters, preserving newlines/tabs.
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove spaces around line breaks.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines, preserving paragraph boundaries.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page by page and preserve source metadata."""
    page_records = []

    with fitz.open(pdf_path) as pdf:
        if pdf.is_encrypted:
            raise ValueError(f"Encrypted PDF: {pdf_path.name}")

        for page_index, page in enumerate(pdf):
            raw_text = page.get_text("text", sort=True)
            text = clean_text(raw_text)

            page_records.append({
                "source": pdf_path.name,
                "page_number": page_index + 1,
                "text": text,
                "char_count": len(text),
            })

    return page_records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save one JSON object per line."""
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR.resolve()}"
        )

    all_pages = []
    failed_files = []

    for pdf_path in pdf_files:
        try:
            pages = extract_pdf(pdf_path)
            all_pages.extend(pages)

            nonempty_pages = sum(
                bool(page["text"]) for page in pages
            )
            total_chars = sum(
                page["char_count"] for page in pages
            )

            print(
                f"[OK] {pdf_path.name}: "
                f"{len(pages)} pages, "
                f"{nonempty_pages} nonempty pages, "
                f"{total_chars:,} characters"
            )

            # Save readable text for manual inspection.
            text_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"

            with text_path.open("w", encoding="utf-8") as file:
                for page in pages:
                    file.write(
                        f"\n\n===== PAGE {page['page_number']} =====\n\n"
                    )
                    file.write(page["text"])

            if not any(page["text"] for page in pages):
                print(
                    f"[WARNING] No extractable text in {pdf_path.name}. "
                    "This may require OCR."
                )

        except Exception as error:
            failed_files.append(pdf_path.name)
            print(f"[ERROR] {pdf_path.name}: {error}")

    jsonl_path = OUTPUT_DIR / "pages.jsonl"
    save_jsonl(all_pages, jsonl_path)

    print(f"\nTotal extracted pages: {len(all_pages)}")
    print(f"Structured output: {jsonl_path}")

    if failed_files:
        raise RuntimeError(
            "Extraction failed for: " + ", ".join(failed_files)
        )


if __name__ == "__main__":
    main()

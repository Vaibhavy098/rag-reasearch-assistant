from pathlib import Path


def check_project_folders():
    """Check that the main data directories exist."""
    folders = [
        Path("data/pdfs"),
        Path("data/processed"),
        Path("data/faiss_index"),
    ]

    for folder in folders:
        if folder.exists():
            print(f"[OK] Found: {folder}")
        else:
            print(f"[MISSING] {folder}")


def main():
    print("RAG Research Assistant")
    print("======================")
    check_project_folders()


if __name__ == "__main__":
    main()

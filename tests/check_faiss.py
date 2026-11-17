
import json
from pathlib import Path

import faiss


INDEX_FILE = Path("data/faiss_index/research_papers.index")
MANIFEST_FILE = Path("data/faiss_index/index_manifest.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def main() -> None:
    for path in (INDEX_FILE, MANIFEST_FILE, CHUNKS_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    index = faiss.read_index(str(INDEX_FILE))

    with MANIFEST_FILE.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    print(f"FAISS index type: {type(index).__name__}")
    print(f"Stored vectors: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")
    print(f"Available chunks: {len(chunks)}")

    if index.ntotal == 0:
        raise AssertionError("FAISS index is empty.")

    if index.ntotal != len(chunks):
        raise AssertionError(
            "Vector count does not match the chunk count."
        )

    if index.d != manifest["embedding_dimension"]:
        raise AssertionError(
            "Index dimensions do not match the manifest."
        )

    if manifest["vector_count"] != index.ntotal:
        raise AssertionError(
            "Manifest vector count does not match the index."
        )

    if manifest["chunk_ids"] != chunk_ids:
        raise AssertionError(
            "Manifest chunk IDs do not match the chunk file."
        )

    print("[PASS] FAISS index and metadata are consistent.")


if __name__ == "__main__":
    main()

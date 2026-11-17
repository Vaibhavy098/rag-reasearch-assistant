
import json
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMBEDDING_MANIFEST = Path("data/processed/embedding_manifest.json")

INDEX_DIR = Path("data/faiss_index")
INDEX_FILE = INDEX_DIR / "research_papers.index"
INDEX_MANIFEST = INDEX_DIR / "index_manifest.json"


def load_chunks() -> list[dict]:
    """Load chunks in their original saved order."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Chunks file is missing. Run python -m src.chunking first."
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("No chunks were found.")

    return chunks


def build_index() -> None:
    """Build an exact inner-product index from normalized vectors."""

    for path in (EMBEDDINGS_FILE, EMBEDDING_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file is missing: {path}. "
                "Run python -m src.embeddings first."
            )

    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)

    with EMBEDDING_MANIFEST.open("r", encoding="utf-8") as file:
        embedding_manifest = json.load(file)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    # Ensure that every vector corresponds to the correct chunk.
    if embedding_manifest["chunk_ids"] != chunk_ids:
        raise ValueError(
            "Chunk order does not match the embedding manifest. "
            "Regenerate embeddings before building the index."
        )

    if embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "The number of embeddings does not match the number of chunks."
        )

    if embeddings.shape[1] != embedding_manifest["embedding_dimension"]:
        raise ValueError("Embedding dimensions do not match the manifest.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Inner product is equivalent to cosine similarity for normalized vectors.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    manifest = {
        "index_type": "IndexFlatIP",
        "embedding_model": embedding_manifest["model_name"],
        "embedding_dimension": int(embeddings.shape[1]),
        "vector_count": int(index.ntotal),
        "chunk_ids": chunk_ids,
    }

    with INDEX_MANIFEST.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("[PASS] FAISS index created successfully.")
    print(f"Index type: {manifest['index_type']}")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Vector dimensions: {embeddings.shape[1]}")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Index manifest saved to: {INDEX_MANIFEST}")


if __name__ == "__main__":
    build_index()

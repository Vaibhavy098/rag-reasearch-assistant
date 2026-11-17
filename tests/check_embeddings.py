
import json
from pathlib import Path

import numpy as np


EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
MANIFEST_FILE = Path("data/processed/embedding_manifest.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")

EXPECTED_DIMENSION = 384


def main() -> None:
    for path in (EMBEDDINGS_FILE, MANIFEST_FILE, CHUNKS_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file is missing: {path}"
            )

    embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)

    with MANIFEST_FILE.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"Chunk count: {len(chunks)}")

    if embeddings.ndim != 2:
        raise AssertionError("Embeddings must be a 2D matrix.")

    if embeddings.shape[0] != len(chunks):
        raise AssertionError(
            "Number of vectors does not match number of chunks."
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise AssertionError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"got {embeddings.shape[1]}."
        )

    if not np.isfinite(embeddings).all():
        raise AssertionError(
            "Embeddings contain NaN or infinite values."
        )

    if manifest["vector_count"] != len(chunks):
        raise AssertionError(
            "Manifest vector count does not match the chunk count."
        )

    if manifest["embedding_dimension"] != EXPECTED_DIMENSION:
        raise AssertionError(
            "Manifest contains an unexpected embedding dimension."
        )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if manifest["chunk_ids"] != chunk_ids:
        raise AssertionError(
            "Chunk ID order in the manifest does not match the chunks."
        )

    if len(chunk_ids) != len(set(chunk_ids)):
        raise AssertionError("Duplicate chunk IDs were found.")

    # Our embeddings are normalized, so each vector should
    # have a Euclidean norm close to 1.
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise AssertionError(
            "Some embedding vectors are not normalized."
        )

    print(f"Model: {manifest['model_name']}")
    print("[PASS] Vector count matches chunk count.")
    print("[PASS] All vectors have the expected dimensions.")
    print("[PASS] All vector values are finite.")
    print("[PASS] Chunk metadata order is preserved.")
    print("[PASS] All embedding vectors are normalized.")


if __name__ == "__main__":
    main()

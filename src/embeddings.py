
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
MANIFEST_FILE = Path("data/processed/embedding_manifest.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    """Load chunks and their metadata from JSONL."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_FILE}. "
            "Run python -m src.chunking first."
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

            if not chunk.get("chunk_id"):
                raise ValueError(
                    f"Missing chunk_id on line {line_number}."
                )

            if not chunk.get("text", "").strip():
                raise ValueError(
                    f"Empty chunk text on line {line_number}."
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("The chunks file contains no usable chunks.")

    return chunks


def generate_embeddings(chunks: list[dict]):
    """Convert chunk text into normalized numerical vectors."""

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print(f"Model maximum sequence length: {model.max_seq_length}")
    print(f"Number of chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    return model, embeddings


def save_embeddings(
    chunks: list[dict],
    embeddings: np.ndarray,
    model,
) -> None:
    """Save vectors and a manifest linking them to their chunks."""

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks does not match the number of vectors."
        )

    np.save(EMBEDDINGS_FILE, embeddings)

    manifest = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "vector_count": int(embeddings.shape[0]),
        "normalized": True,
        "model_max_sequence_length": int(model.max_seq_length),
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Manifest saved to: {MANIFEST_FILE}")


def main() -> None:
    chunks = load_chunks()
    model, embeddings = generate_embeddings(chunks)

    save_embeddings(chunks, embeddings, model)

    print("\nEmbedding generation completed.")
    print(f"Vector matrix shape: {embeddings.shape}")
    print(f"Vector data type: {embeddings.dtype}")
    print(
        "Expected dimension: 384"
        if embeddings.shape[1] == 384
        else f"Unexpected dimension: {embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()

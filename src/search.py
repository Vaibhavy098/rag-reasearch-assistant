
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
INDEX_FILE = Path("data/faiss_index/research_papers.index")
INDEX_MANIFEST = Path("data/faiss_index/index_manifest.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ResearchPaperSearcher:
    """Search research-paper chunks using semantic similarity."""

    def __init__(self):
        for path in (CHUNKS_FILE, INDEX_FILE, INDEX_MANIFEST):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file is missing: {path}. "
                    "Run the ingestion, chunking, embedding, "
                    "and index-building steps first."
                )

        with CHUNKS_FILE.open("r", encoding="utf-8") as file:
            self.chunks = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        with INDEX_MANIFEST.open("r", encoding="utf-8") as file:
            self.manifest = json.load(file)

        chunk_ids = [chunk["chunk_id"] for chunk in self.chunks]

        if self.manifest["chunk_ids"] != chunk_ids:
            raise ValueError(
                "The saved index does not match the current chunks. "
                "Rebuild the FAISS index."
            )

        if self.manifest["embedding_model"] != MODEL_NAME:
            raise ValueError(
                "The index was built with a different embedding model."
            )

        self.index = faiss.read_index(str(INDEX_FILE))

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "The FAISS vector count does not match the chunk count."
            )

        if self.index.d != self.manifest["embedding_dimension"]:
            raise ValueError(
                "FAISS index dimensions do not match the manifest."
            )

        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print(f"FAISS index loaded: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the most similar chunks for a user's question."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimensions do not match the FAISS index."
            )

        number_to_retrieve = min(top_k, self.index.ntotal)

        scores, positions = self.index.search(
            query_vector,
            number_to_retrieve,
        )

        results = []

        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue

            chunk = self.chunks[int(position)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "similarity_score": float(score),
            })

        return results


def main() -> None:
    searcher = ResearchPaperSearcher()

    print("\nResearch Paper Semantic Search")
    print("Ask a question about the five papers.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            results = searcher.search(query, top_k=5)

            print(f"\nTop {len(results)} matching chunks:")

            for rank, result in enumerate(results, start=1):
                print(f"\n{'=' * 65}")
                print(f"Result #{rank}")
                print(f"Source: {result['source']}")
                print(f"Page: {result['page_number']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(
                    "Similarity score: "
                    f"{result['similarity_score']:.4f}"
                )
                print("\nRetrieved text:")
                print(result["text"][:1000])

        except Exception as error:
            print(f"[ERROR] Search failed: {error}")


if __name__ == "__main__":
    main()

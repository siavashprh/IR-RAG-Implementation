"""
chunked_retrieval.py
--------------------
Handles long documents by splitting them into overlapping chunks,
embedding each chunk separately, and retrieving at the chunk level.

When a chunk matches a query, the parent document is returned.
If multiple chunks from the same document match, the highest-scoring
chunk determines the document's rank (max-pooling over chunks).

This directly addresses the 512-token truncation problem: instead of
the sentence transformer seeing only the first ~512 tokens of a paper,
every section of the paper gets its own embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    from retrieval import RetrievalResult
except ImportError:
    from dataclasses import dataclass as _dc
    @_dc
    class RetrievalResult:
        dataset_index: int
        official_id: int | None
        score: float


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks.

    Args:
        text:       The full text to chunk.
        chunk_size: Number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings. Empty texts return a single empty-string chunk
        so every document has at least one chunk.
    """
    words = text.split()
    if len(words) == 0:
        return [""]

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        # If this chunk reached the end, stop
        if start + chunk_size >= len(words):
            break

    return chunks


@dataclass
class ChunkedNeuralRetriever:
    """Dense retriever that chunks long documents before embedding.

    Each document is split into overlapping chunks. Each chunk is embedded
    independently. At retrieval time, the query is compared against all
    chunks; the best-matching chunk per document determines the document's
    score (max-pooling).

    Usage:
        r = ChunkedNeuralRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            text_field="full_text",
            chunk_size=200, overlap=50,
        )
        r.fit(dataset)
        hits = r.retrieve("attention mechanism", k=10)
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_field: str = "full_text"
    prefix_fields: tuple[str, ...] = ("title",)  # prepended to each chunk for context
    id_field: str = "acl_id"
    chunk_size: int = 200      # words per chunk
    overlap: int = 50          # overlapping words between chunks
    batch_size: int = 64
    show_progress: bool = True

    # Populated by fit()
    model: "SentenceTransformer | None" = field(default=None, init=False, repr=False)
    chunk_embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    chunk_to_doc: list[int] | None = field(default=None, init=False)    # chunk index -> doc index
    doc_ids: list | None = field(default=None, init=False)              # doc index -> acl_id

    def fit(self, dataset) -> "ChunkedNeuralRetriever":
        """Chunk all documents, then encode all chunks."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

        all_chunks = []
        self.chunk_to_doc = []
        self.doc_ids = []

        for doc_idx, doc in enumerate(dataset):
            self.doc_ids.append(doc.get(self.id_field))

            # Build prefix from title (or other fields) for context
            prefix_parts = []
            for f in self.prefix_fields:
                val = doc.get(f)
                if val:
                    prefix_parts.append(str(val).strip())
            prefix = " ".join(prefix_parts)

            # Chunk the full text
            full_text = str(doc.get(self.text_field, ""))
            chunks = chunk_text(full_text, self.chunk_size, self.overlap)

            for chunk in chunks:
                # Prepend title to each chunk so the encoder has document context
                if prefix:
                    all_chunks.append(f"{prefix}. {chunk}")
                else:
                    all_chunks.append(chunk)
                self.chunk_to_doc.append(doc_idx)

        print(f"  {len(self.doc_ids)} documents -> {len(all_chunks)} chunks "
              f"(avg {len(all_chunks)/len(self.doc_ids):.1f} chunks/doc)")

        # Encode all chunks
        self.chunk_embeddings = self.model.encode(
            all_chunks,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,
        )

        return self

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k documents by best-chunk cosine similarity.

        For each document, the score is the maximum similarity across all
        its chunks (max-pooling). This ensures that a paper is ranked
        highly if *any* section is relevant to the query.
        """
        if self.chunk_embeddings is None:
            raise RuntimeError("Not fit yet.")

        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        chunk_scores = self.chunk_embeddings @ q_vec  # cosine for all chunks

        # Max-pool: for each document, keep only the best chunk score
        doc_best_score = {}
        for chunk_idx, score in enumerate(chunk_scores):
            doc_idx = self.chunk_to_doc[chunk_idx]
            if doc_idx not in doc_best_score or score > doc_best_score[doc_idx]:
                doc_best_score[doc_idx] = float(score)

        # Sort documents by best chunk score
        sorted_docs = sorted(doc_best_score.items(), key=lambda x: -x[1])[:k]

        results = []
        for doc_idx, score in sorted_docs:
            results.append(RetrievalResult(
                dataset_index=doc_idx,
                official_id=self.doc_ids[doc_idx],
                score=score,
            ))
        return results

"""
neural_retrieval.py
-------------------
Dense retrieval using a pre-trained sentence transformer.

Uses the `sentence-transformers` library to encode documents and queries
as dense vectors, then retrieves by cosine similarity.

The interface mirrors TfidfRetriever: .fit(dataset), .retrieve(query, k)
returning RetrievalResult objects — so evaluation.py works without changes.

Model choice: all-MiniLM-L6-v2 (default)
  - 22M parameters, 384-dim embeddings
  - Competitive on MTEB retrieval benchmarks for its size class
  - Encodes ~2500 docs in <1 min on CPU
  - Fits in Colab memory alongside Mistral-7B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# RetrievalResult is imported from retrieval.py so eval code works seamlessly.
try:
    from retrieval import RetrievalResult
except ImportError:
    from dataclasses import dataclass as _dc
    @_dc
    class RetrievalResult:
        dataset_index: int
        official_id: int | None
        score: float


def build_document_text_acl(doc: dict, fields: Iterable[str]) -> str:
    """Concatenate selected fields of an ACL paper into one string.

    Similar to build_document_text from retrieval.py but handles
    ACL-specific field types (author lists, etc.).
    """
    parts = []
    for f in fields:
        value = doc.get(f)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value)
        value = str(value).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


@dataclass
class NeuralRetriever:
    """Dense retriever using a sentence transformer.

    Usage:
        r = NeuralRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        r.fit(dataset, id_field="acl_id", fields=("title", "abstract"))
        hits = r.retrieve("attention mechanisms in NLP", k=10)
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fields: tuple[str, ...] = ("title", "abstract")
    id_field: str = "acl_id"
    batch_size: int = 64
    show_progress: bool = True

    # Populated by fit()
    model: "SentenceTransformer | None" = field(default=None, init=False, repr=False)
    doc_embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    doc_ids: list | None = field(default=None, init=False)

    def fit(self, dataset) -> "NeuralRetriever":
        """Encode all documents in the dataset."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

        texts = []
        self.doc_ids = []
        for doc in dataset:
            texts.append(build_document_text_acl(doc, self.fields))
            self.doc_ids.append(doc.get(self.id_field))

        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,  # L2-normalize so dot product = cosine
        )

        return self

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a dense vector."""
        if self.model is None:
            raise RuntimeError("Retriever not fit; call .fit(dataset) first.")
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return top-k documents by cosine similarity (= dot product, since L2-normalized)."""
        if self.doc_embeddings is None:
            raise RuntimeError("Retriever not fit; call .fit(dataset) first.")

        q_vec = self.encode_query(query)
        scores = self.doc_embeddings @ q_vec

        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for idx in top_idx:
            results.append(RetrievalResult(
                dataset_index=int(idx),
                official_id=self.doc_ids[idx],  # acl_id stored as official_id for eval compat
                score=float(scores[idx]),
            ))
        return results

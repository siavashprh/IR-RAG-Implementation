"""
meanpool_retrieval.py
---------------------
Handles long documents by encoding each sentence separately with a
sentence transformer, then mean-pooling the sentence embeddings into
one document vector.

This is directly from Lecture 4 (slide 47): "mean pool the sentence
representations across the document." Every sentence of the paper
contributes to the final embedding, avoiding the 512-token truncation
problem where only the first ~200 words are seen.

Contrast with chunked_retrieval.py (Strategy 1):
  - Chunking: one vector per chunk, max-pool at retrieval (ColBERT-inspired)
  - Mean-pooling: one vector per document, averaged from all sentences
  - Chunking preserves local detail; mean-pooling compresses into one vector
    (softer information bottleneck, but still a bottleneck)
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


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitting on period/question/exclamation + space.

    Good enough for academic papers. A proper sentence splitter (e.g.
    spaCy) would be better but adds a dependency.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.split()) >= 3]  # drop tiny fragments


@dataclass
class MeanPoolRetriever:
    """Dense retriever that mean-pools sentence embeddings per document.

    Each document is split into sentences, each sentence is embedded
    by the sentence transformer, and the document embedding is the
    mean of all its sentence embeddings. This avoids the 512-token
    truncation: every sentence contributes, no matter how long the paper.

    Usage:
        r = MeanPoolRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            text_fields=("abstract", "full_text"),
        )
        r.fit(dataset)
        hits = r.retrieve("attention mechanism", k=10)
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_fields: tuple[str, ...] = ("abstract", "full_text")
    id_field: str = "acl_id"
    batch_size: int = 256
    show_progress: bool = True

    # Populated by fit()
    model: "SentenceTransformer | None" = field(default=None, init=False, repr=False)
    doc_embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    doc_ids: list | None = field(default=None, init=False)

    def fit(self, dataset) -> "MeanPoolRetriever":
        """Encode all documents via sentence-level mean pooling."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        emb_dim = self.model.get_sentence_embedding_dimension()

        # Collect all sentences from all documents, tracking which doc each belongs to
        all_sentences = []
        sentence_to_doc = []
        self.doc_ids = []
        doc_sentence_counts = []

        for doc_idx, doc in enumerate(dataset):
            self.doc_ids.append(doc.get(self.id_field))

            # Concatenate selected text fields
            text_parts = []
            for f in self.text_fields:
                val = doc.get(f)
                if val:
                    text_parts.append(str(val).strip())
            full_text = " ".join(text_parts)

            sentences = _split_sentences(full_text)
            if not sentences:
                sentences = [""]  # ensure every doc has at least one "sentence"

            doc_sentence_counts.append(len(sentences))
            for s in sentences:
                all_sentences.append(s)
                sentence_to_doc.append(doc_idx)

        total_sents = len(all_sentences)
        print(f"  {len(self.doc_ids)} documents -> {total_sents} sentences "
              f"(avg {total_sents/len(self.doc_ids):.1f} sentences/doc)")

        # Encode all sentences in one batch
        all_embeddings = self.model.encode(
            all_sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=False,  # normalize AFTER mean pooling
        )

        # Mean pool per document
        self.doc_embeddings = np.zeros((len(self.doc_ids), emb_dim), dtype=np.float32)
        for sent_idx, doc_idx in enumerate(sentence_to_doc):
            self.doc_embeddings[doc_idx] += all_embeddings[sent_idx]

        for doc_idx, count in enumerate(doc_sentence_counts):
            if count > 0:
                self.doc_embeddings[doc_idx] /= count

        # L2-normalize so dot product = cosine
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self.doc_embeddings /= norms

        return self

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k documents by cosine similarity."""
        if self.doc_embeddings is None:
            raise RuntimeError("Not fit yet.")

        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
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
                official_id=self.doc_ids[idx],
                score=float(scores[idx]),
            ))
        return results

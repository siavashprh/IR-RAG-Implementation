"""
retrieval.py
------------
TF-IDF retrieval over the recipes dataset.

Design notes:
- Documents are constructed by concatenating selected fields per recipe.
  Chosen fields after ablation: name + ingredients + tags + steps.
- We use sklearn's TfidfVectorizer (defaults: sublinear_tf=False,
  norm='l2', smooth_idf=True). The IDF formula is:
      idf(t) = ln((1 + N) / (1 + df(t))) + 1
  and each document/query vector is L2-normalized after weighting.
- Query vectors use the same vectorizer (fit on the corpus), so OOV
  query terms are simply dropped. A fully-OOV query yields a zero vector,
  which we detect and report.
- Similarity: cosine. Because vectors are L2-normalized, cosine == dot product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Field handling ---------------------------------------------------------

def _normalize_field(value) -> str:
    """Coerce any dataset field into a single space-joined string.

    Recipes fields may be strings, list-of-strings (ingredients, steps, tags),
    or occasionally None. We flatten all of them to plain text so the
    vectorizer can tokenize uniformly.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_normalize_field(v) for v in value)
    return str(value)


def build_document_text(recipe: dict, fields: Iterable[str]) -> str:
    """Concatenate selected fields of a recipe into one string.

    Tags often contain hyphens (e.g. '60-minutes-or-less'). Hyphens are
    replaced with spaces so the tokenizer sees the constituent words.
    """
    parts = []
    for f in fields:
        text = _normalize_field(recipe.get(f))
        if f == "tags":
            text = text.replace("-", " ")
        if text:
            parts.append(text)
    return " ".join(parts)


# --- Retriever --------------------------------------------------------------

DEFAULT_FIELDS = ("name", "ingredients", "tags")


@dataclass
class RetrievalResult:
    """One retrieval result: dataset index, recipe id, similarity score."""
    dataset_index: int
    official_id: int | None
    score: float


@dataclass
class TfidfRetriever:
    """TF-IDF retriever with cosine similarity over a recipe corpus.

    Usage:
        r = TfidfRetriever(fields=("name", "ingredients", "tags"))
        r.fit(dataset)                       # dataset = HF Dataset or list of dicts
        hits = r.retrieve("shrimp tacos", k=10)

    Optional MWE support:
        from mwe import MWEDetector
        mwe = MWEDetector().fit(doc_texts)
        r = TfidfRetriever(fields=..., mwe=mwe)
        r.fit(dataset)
        # Now both documents and queries get MWE-merged before vectorization.
    """

    fields: tuple[str, ...] = DEFAULT_FIELDS

    # Vectorizer hyperparameters — exposed so experiments can vary them
    # without subclassing.
    lowercase: bool = True
    min_df: int = 2          # drop hapax legomena; reduces vocab + noise
    max_df: float = 0.95     # drop terms in >95% of docs (very weak signal)
    ngram_range: tuple[int, int] = (1, 1)
    stop_words: str | None = "english"
    token_pattern: str = r"(?u)\b[a-z][a-z_]+\b"  # alphabetic OR underscore (for MWEs), length >= 2

    # Optional MWE detector. If provided, .fit() and .encode_query() apply
    # the detector before handing text to the vectorizer.
    mwe: "MWEDetector | None" = None

    # Populated by fit()
    vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    doc_matrix: csr_matrix | None = field(default=None, init=False)
    official_ids: list[int] | None = field(default=None, init=False)

    # ---- fit / index ----

    def fit(self, dataset) -> "TfidfRetriever":
        """Build the term vocabulary and document-term matrix.

        `dataset` can be a HuggingFace Dataset, a list of dicts, or anything
        that iterates over recipe dicts containing at least `self.fields`.
        """
        texts = []
        official_ids = []
        for recipe in dataset:
            texts.append(build_document_text(recipe, self.fields))
            official_ids.append(recipe.get("official_id"))

        # If an MWE detector is attached, merge collocations in each document
        # before fitting the vectorizer. "olive oil" -> "olive_oil".
        if self.mwe is not None:
            texts = [self.mwe.transform_text(t) for t in texts]

        self.vectorizer = TfidfVectorizer(
            lowercase=self.lowercase,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            token_pattern=self.token_pattern,
            sublinear_tf=False,
            norm="l2",
            smooth_idf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.official_ids = official_ids
        return self

    # ---- query ----

    def encode_query(self, query: str) -> csr_matrix:
        """Turn a query string into a TF-IDF row vector using the fitted vocab.

        OOV terms are silently dropped (sklearn behavior). If *all* terms are
        OOV, the returned vector has zero nnz — callers should check for this.
        """
        if self.vectorizer is None:
            raise RuntimeError("Retriever not fit; call .fit(dataset) first.")
        # Apply the same MWE transform to queries that we applied to docs,
        # so a query for "olive oil" matches against the merged "olive_oil"
        # token in the vocabulary.
        if self.mwe is not None:
            query = self.mwe.transform_text(query)
        return self.vectorizer.transform([query])

    def is_query_in_vocab(self, query: str) -> bool:
        """True iff the query has at least one in-vocabulary token."""
        return self.encode_query(query).nnz > 0

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return top-k recipes by cosine similarity.

        Vectors are L2-normalized, so cosine = dot product. We compute the
        sparse-dense product, then partial-sort for the top-k.
        """
        if self.doc_matrix is None:
            raise RuntimeError("Retriever not fit; call .fit(dataset) first.")

        q_vec = self.encode_query(query)
        # (1 x V) @ (V x N).T -> (1 x N) dense scores
        scores = (self.doc_matrix @ q_vec.T).toarray().ravel()

        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            # argpartition is O(N); sort only the top-k slice
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for idx in top_idx:
            results.append(RetrievalResult(
                dataset_index=int(idx),
                official_id=self.official_ids[idx],
                score=float(scores[idx]),
            ))
        return results

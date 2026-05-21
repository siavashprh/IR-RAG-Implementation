"""
query_transforms.py
-------------------
Query rewriting and HyDE (Hypothetical Document Embeddings) for Part 2.

Both techniques use the LM to transform the query before retrieval:
- Query rewriting: LM rephrases the query as a better search query.
- HyDE: LM generates a fake document that answers the query; we embed
  that document instead of the query.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from retrieval import RetrievalResult
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class RetrievalResult:
        dataset_index: int
        official_id: int | None
        score: float

from evaluation import evaluate_query


# --- Query Rewriting -------------------------------------------------------

def rewrite_query(query: str, tokenizer, model, n_rewrites: int = 1,
                  max_new_tokens: int = 100) -> list[str]:
    """Use the LM to rewrite a query into a better search query."""
    prompt = (
        "Rewrite the following question as a short, precise search query "
        "for finding relevant academic papers. Use technical terminology. "
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Original question: {query}\n\n"
        "Rewritten search query:"
    )
    messages = [{"role": "user", "content": prompt}]

    rewrites = []
    for i in range(n_rewrites):
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            return_dict=False,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7 + (i * 0.1),
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0, encoded.shape[1]:]
        rewrite = tokenizer.decode(generated, skip_special_tokens=True).strip()
        rewrite = rewrite.split("\n")[0].strip()
        rewrites.append(rewrite)

    return rewrites


def retrieve_with_rewrite(query: str, retriever, tokenizer, model,
                          k: int = 10, n_rewrites: int = 1):
    """Retrieve using rewritten queries.

    For multiple rewrites: combine via Reciprocal Rank Fusion (RRF),
    including the original query.
    """
    rewrites = rewrite_query(query, tokenizer, model, n_rewrites=n_rewrites)

    if n_rewrites == 1:
        return retriever.retrieve(rewrites[0], k=k), rewrites

    # Multiple rewrites: Reciprocal Rank Fusion
    doc_rrf_scores = {}
    for rw in rewrites:
        hits = retriever.retrieve(rw, k=k * 2)
        for rank, h in enumerate(hits):
            doc_id = h.official_id
            if doc_id not in doc_rrf_scores:
                doc_rrf_scores[doc_id] = {"score": 0.0, "dataset_index": h.dataset_index}
            doc_rrf_scores[doc_id]["score"] += 1.0 / (rank + 60)

    # Include original query in fusion
    orig_hits = retriever.retrieve(query, k=k * 2)
    for rank, h in enumerate(orig_hits):
        doc_id = h.official_id
        if doc_id not in doc_rrf_scores:
            doc_rrf_scores[doc_id] = {"score": 0.0, "dataset_index": h.dataset_index}
        doc_rrf_scores[doc_id]["score"] += 1.0 / (rank + 60)

    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: -x[1]["score"])[:k]

    results = []
    for doc_id, info in sorted_docs:
        results.append(RetrievalResult(
            dataset_index=info["dataset_index"],
            official_id=doc_id,
            score=info["score"],
        ))
    return results, rewrites


# --- HyDE ------------------------------------------------------------------

def generate_hypothetical_document(query: str, tokenizer, model,
                                    max_new_tokens: int = 200) -> str:
    """Generate a fake document that looks like it answers the query."""
    prompt = (
        "Write a short academic paragraph that answers the following "
        "research question. Write it as if it were part of a published NLP paper. "
        "Use technical language. Do not include citations or references.\n\n"
        f"Research question: {query}\n\n"
        "Paragraph:"
    )
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, encoded.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def retrieve_with_hyde(query: str, retriever, tokenizer, model,
                       k: int = 10):
    """Retrieve by embedding a hypothetical document instead of the query."""
    hypo_doc = generate_hypothetical_document(query, tokenizer, model)

    q_vec = retriever.model.encode([hypo_doc], normalize_embeddings=True)[0]
    scores = retriever.doc_embeddings @ q_vec

    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for idx in top_idx:
        results.append(RetrievalResult(
            dataset_index=int(idx),
            official_id=retriever.doc_ids[idx],
            score=float(scores[idx]),
        ))
    return results, hypo_doc

"""
evaluation.py
-------------
Evaluate an IR retriever against a gold-query set.

Metrics (binary relevance):
- Per-query precision, recall, F1 at fixed k.
- Macro-average: mean of per-query metrics.
- Micro-average: pool TP/FP/FN across all queries, then compute.
- Mean Average Precision (MAP): mean over queries of Average Precision (AP).

Average Precision formula (for binary relevance):
    AP(q) = (1 / |R_q|) * sum_{k=1..K} [ P@k(q) * rel(k) ]
where:
    R_q       = set of all relevant documents for query q
    K         = number of retrieved documents (the system's chosen k)
    P@k(q)    = precision considering only the top-k retrieved docs
    rel(k)    = 1 if the k-th retrieved doc is relevant, else 0

Then MAP = (1/|Q|) * sum_{q in Q} AP(q).

Note on the "at k" question: AP here is computed *over the system's
retrieved list of size K*, not over the full corpus. This is the standard
formulation when the system commits to a fixed retrieval cutoff. If the
system retrieves fewer than |R_q| documents, AP is bounded above by K/|R_q|.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# --- Mapping: official_id <-> dataset index --------------------------------

def build_id_to_index(dataset) -> dict[int, int]:
    """Map each recipe's official_id to its position in the dataset.

    The gold queries reference official_id; the retriever returns dataset
    positions. We need to translate between them in both directions.
    """
    mapping = {}
    for i, recipe in enumerate(dataset):
        oid = recipe.get("official_id")
        if oid is not None:
            mapping[oid] = i
    return mapping


# --- Per-query metrics -----------------------------------------------------

@dataclass
class QueryEval:
    """Per-query evaluation result."""
    query: str
    retrieved_ids: list[int]       # official_ids returned by the system, in rank order
    relevant_ids: set[int]         # official_ids marked relevant in gold
    tp: int                        # # retrieved AND relevant
    fp: int                        # # retrieved AND NOT relevant
    fn: int                        # # NOT retrieved BUT relevant
    precision: float
    recall: float
    f1: float
    average_precision: float


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_query(query: str,
                   retrieved_ids: list[int],
                   relevant_ids: Iterable[int]) -> QueryEval:
    """Compute precision/recall/F1/AP for a single query.

    Args:
        query:         The query string (stored for inspection).
        retrieved_ids: Official IDs returned by the system, in rank order.
        relevant_ids:  Official IDs that are gold-relevant (any score >= 1).
    """
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)

    tp = len(retrieved_set & relevant_set)
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)

    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0
    f1 = _f1(precision, recall)

    # --- Average Precision (rank-aware) ---
    # Walk down the retrieved list; at each position where the doc is
    # relevant, record the running precision; average those values over
    # the total count of relevant docs.
    ap_sum = 0.0
    hits_so_far = 0
    for k, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            hits_so_far += 1
            precision_at_k = hits_so_far / k
            ap_sum += precision_at_k
    ap = ap_sum / len(relevant_set) if relevant_set else 0.0

    return QueryEval(
        query=query,
        retrieved_ids=list(retrieved_ids),
        relevant_ids=relevant_set,
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        average_precision=ap,
    )


# --- Aggregate metrics over a query set ------------------------------------

@dataclass
class EvalReport:
    """Summary statistics across a query set."""
    n_queries: int
    k: int

    # Macro: mean of per-query metrics
    macro_precision: float
    macro_recall: float
    macro_f1: float

    # Micro: pooled counts then compute
    micro_precision: float
    micro_recall: float
    micro_f1: float

    # Ranking metric
    mean_average_precision: float

    # Keep per-query results for drill-down
    per_query: list[QueryEval]

    def pretty(self) -> str:
        lines = [
            f"Evaluation report  (N={self.n_queries} queries, k={self.k})",
            "-" * 50,
            f"  Macro-avg precision: {self.macro_precision:.4f}",
            f"  Macro-avg recall:    {self.macro_recall:.4f}",
            f"  Macro-avg F1:        {self.macro_f1:.4f}",
            "",
            f"  Micro-avg precision: {self.micro_precision:.4f}",
            f"  Micro-avg recall:    {self.micro_recall:.4f}",
            f"  Micro-avg F1:        {self.micro_f1:.4f}",
            "",
            f"  MAP:                 {self.mean_average_precision:.4f}",
        ]
        return "\n".join(lines)


def evaluate_retriever(retriever, queries: list[dict], k: int) -> EvalReport:
    """Run the retriever over a gold-query set and compute all metrics.

    Args:
        retriever:  An object with a `.retrieve(query, k)` method returning
                    objects that have an `.official_id` attribute (matches
                    TfidfRetriever's RetrievalResult).
        queries:    List of dicts with keys 'q' (query string) and
                    'r' (list of [official_id, relevance_score] pairs).
        k:          Fixed retrieval cutoff for the system.

    Returns:
        EvalReport with macro, micro, and MAP metrics.
    """
    per_query = []
    total_tp = total_fp = total_fn = 0

    for q in queries:
        query_text = q["q"]
        # Handle both formats:
        #   Part 1: q["r"] = [[official_id, score], ...]  (list of pairs)
        #   Part 2: q["r"] = [acl_id, ...]                (flat list of IDs)
        raw_r = q["r"]
        if raw_r and isinstance(raw_r[0], (list, tuple)):
            relevant_ids = [pair[0] for pair in raw_r]
        else:
            relevant_ids = list(raw_r)

        hits = retriever.retrieve(query_text, k=k)
        retrieved_ids = [h.official_id for h in hits if h.official_id is not None]

        eval_q = evaluate_query(query_text, retrieved_ids, relevant_ids)
        per_query.append(eval_q)

        total_tp += eval_q.tp
        total_fp += eval_q.fp
        total_fn += eval_q.fn

    # Macro: mean over per-query metrics
    macro_p = float(np.mean([q.precision for q in per_query]))
    macro_r = float(np.mean([q.recall for q in per_query]))
    macro_f1 = float(np.mean([q.f1 for q in per_query]))

    # Micro: pool counts then compute
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = _f1(micro_p, micro_r)

    mean_ap = float(np.mean([q.average_precision for q in per_query]))

    return EvalReport(
        n_queries=len(queries),
        k=k,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        mean_average_precision=mean_ap,
        per_query=per_query,
    )


# --- Convenience: scan over multiple k values ------------------------------

def evaluate_at_multiple_k(retriever, queries: list[dict],
                           ks: Iterable[int]) -> dict[int, EvalReport]:
    """Run evaluation for several values of k. Useful for k-sweeps."""
    return {k: evaluate_retriever(retriever, queries, k=k) for k in ks}

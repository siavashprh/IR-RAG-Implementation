"""
main.py
-------
Central entry point for the IRSE RAG project.
Reproduces all results from both Part 1 and Part 2.

Usage:
    python main.py --part1          # Run Part 1 (recipes) — CPU only for IR, GPU for LM
    python main.py --part2          # Run Part 2 (ACL papers) — CPU for IR, GPU for LM
    python main.py --part1 --part2  # Run both

Requirements:
    pip install datasets scikit-learn gensim sentence-transformers
    pip install transformers bitsandbytes accelerate  # for LM inference (GPU)

Notes:
    - Part 1 IR experiments (TF-IDF, eval, ablation) run on CPU in ~5 min.
    - Part 2 IR experiments (neural, chunked, mean-pool) run on CPU in ~5 min.
    - LM-dependent tasks (generation, query rewriting, HyDE) require GPU
      and take 30-60 min. These are skipped unless --with-lm is passed.
    - Dataset downloads require internet access.
"""

import argparse
import json
import time


# ==========================================================================
# Part 1: Recipes dataset
# ==========================================================================

def run_part1(with_lm: bool = False):
    """Reproduce all Part 1 results."""
    import datasets
    from retrieval import TfidfRetriever, build_document_text
    from evaluation import evaluate_retriever, evaluate_at_multiple_k
    from mwe import MWEDetector

    print("=" * 70)
    print("PART 1: Recipes Dataset")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1] Loading dataset...")
    ds = datasets.load_dataset(
        "parquet",
        data_files="https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2026_recipes.parquet"
    )["train"]
    print(f"    {len(ds)} recipes loaded.")

    print("\n[2] Loading queries...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2026_recipes.json",
        "irse_queries_2026_recipes.json"
    )
    with open("irse_queries_2026_recipes.json") as f:
        queries = json.load(f)
    print(f"    {len(queries['queries'])} queries loaded.")

    # ---- Task 3: Field ablation ----
    print("\n[3] Field ablation experiment...")
    field_configs = {
        "name+ing+tags":   ("name", "ingredients", "tags"),
        "+steps":          ("name", "ingredients", "tags", "steps"),
        "+description":    ("name", "ingredients", "tags", "description"),
        "all 5 fields":    ("name", "ingredients", "tags", "steps", "description"),
        "name+ing only":   ("name", "ingredients"),
    }
    for label, fields in field_configs.items():
        r = TfidfRetriever(fields=fields).fit(ds)
        rep = evaluate_retriever(r, queries["queries"], k=10)
        print(f"    {label:<20s}  vocab={len(r.vectorizer.vocabulary_):>6d}  "
              f"MAP={rep.mean_average_precision:.3f}  F1={rep.macro_f1:.3f}")

    # ---- Fit the working retriever (+steps) ----
    print("\n[4] Fitting working retriever (name+ingredients+tags+steps)...")
    r = TfidfRetriever(fields=("name", "ingredients", "tags", "steps")).fit(ds)
    rep = evaluate_retriever(r, queries["queries"], k=10)
    print(rep.pretty())

    # ---- Task 4: K-sweep ----
    print("\n[5] K-sweep...")
    for k in [5, 10, 20, 50, 100]:
        rep_k = evaluate_retriever(r, queries["queries"], k=k)
        print(f"    k={k:3d}  P={rep_k.macro_precision:.3f}  R={rep_k.macro_recall:.3f}  "
              f"F1={rep_k.macro_f1:.3f}  MAP={rep_k.mean_average_precision:.3f}")

    # ---- Task 2: MWE ----
    print("\n[6] MWE experiment...")
    doc_texts = [build_document_text(rec, ("name", "ingredients", "tags", "steps"))
                 for rec in ds]
    mwe = MWEDetector(min_count=200, threshold=0.5).fit(doc_texts)
    print("    Top 10 MWEs:")
    for phrase, score in mwe.inspect_top_mwes(n=10):
        print(f"      {phrase:<30s} NPMI={score:.3f}")
    r_mwe = TfidfRetriever(
        fields=("name", "ingredients", "tags", "steps"), mwe=mwe
    ).fit(ds)
    rep_mwe = evaluate_retriever(r_mwe, queries["queries"], k=10)
    print(f"    MWE retriever: MAP={rep_mwe.mean_average_precision:.3f} "
          f"(baseline: {rep.mean_average_precision:.3f})")

    # ---- LM tasks (Tasks 6, 7) ----
    if with_lm:
        print("\n[7] Loading LM (Mistral-7B-Instruct-v0.2)...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from generation import (rag_generate, build_prompt_a, build_prompt_b,
                                rag_generate_with_fake_context)

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config,
        )

        # Task 6: Prompt comparison
        print("\n[8] Prompt A vs B...")
        test_q = "What temperature should I pre-heat my oven to when making chicken quesadillas?"
        result_a = rag_generate(test_q, r, ds, tokenizer, model, k=5, prompt_fn=build_prompt_a)
        print(f"    Prompt A response: {result_a['response'][:200]}...")
        result_b = rag_generate(test_q, r, ds, tokenizer, model, k=5, prompt_fn=build_prompt_b)
        print(f"    Prompt B response: {result_b['response'][:200]}...")

        # Task 7: Grounding test
        print("\n[9] Grounding test...")
        irrelevant_context = """Richard Gary Brautigan (January 30, 1935 – c. September 16, 1984)
was an American novelist, poet, and short story writer."""
        grounding = rag_generate_with_fake_context(
            test_q, irrelevant_context, tokenizer, model
        )
        print(f"    Irrelevant context response: {grounding[:200]}...")

        # Task 7: Reasoning queries
        print("\n[10] Reasoning queries...")
        reasoning_queries = [
            "Between chicken quesadillas and beef tacos, which requires fewer ingredients?",
            "What spice or seasoning appears most often across different chili recipes?",
            "If I'm allergic to dairy, which pasta recipes can I make without modification?",
            "What are the different ways these recipes suggest cooking shrimp?",
            "Based on these soup recipes, what is the typical cooking time for a homemade soup?",
        ]
        for rq in reasoning_queries:
            result = rag_generate(rq, r, ds, tokenizer, model, k=5, prompt_fn=build_prompt_a)
            print(f"    Q: {rq}")
            print(f"    A: {result['response'][:150]}...")
            print()
    else:
        print("\n[7-10] Skipping LM tasks (pass --with-lm to enable).")

    print("\nPart 1 complete.")


# ==========================================================================
# Part 2: ACL Anthology dataset
# ==========================================================================

def run_part2(with_lm: bool = False):
    """Reproduce all Part 2 results."""
    import datasets
    import numpy as np
    from retrieval import TfidfRetriever
    from evaluation import evaluate_retriever, evaluate_query
    from neural_retrieval import NeuralRetriever
    from chunked_retrieval import ChunkedNeuralRetriever
    from meanpool_retrieval import MeanPoolRetriever

    print("=" * 70)
    print("PART 2: ACL Anthology Dataset")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1] Loading ACL dataset...")
    import urllib.request, random
    urllib.request.urlretrieve(
        "https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2026_acl.parquet",
        "irse_documents_2026_acl.parquet"
    )
    urllib.request.urlretrieve(
        "https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2026_acl.json",
        "irse_queries_2026_acl.json"
    )
    ds_full = datasets.load_dataset("parquet",
        data_files="irse_documents_2026_acl.parquet")["train"]

    with open("irse_queries_2026_acl.json") as f:
        queries = json.load(f)

    r_number = 1081091
    random.seed(r_number)
    relevant_ids = set()
    for q in queries["queries"]:
        for rid in q["r"]:
            relevant_ids.add(rid if isinstance(rid, str) else rid[0])
    all_ids = [doc["acl_id"] for doc in ds_full]
    non_relevant = [i for i, aid in enumerate(all_ids) if aid not in relevant_ids]
    sample_idx = random.sample(non_relevant, min(2000, len(non_relevant)))
    relevant_idx = [i for i, aid in enumerate(all_ids) if aid in relevant_ids]
    keep_idx = sorted(set(sample_idx + relevant_idx))
    anthology_sample = ds_full.select(keep_idx)
    print(f"    {len(anthology_sample)} documents sampled.")
    print(f"    {len(queries['queries'])} queries loaded.")

    # ---- Task 1: TF-IDF baseline ----
    print("\n[2] TF-IDF baseline (title + abstract)...")
    acl_wrapped = []
    for doc in anthology_sample:
        d = dict(doc)
        d["official_id"] = d["acl_id"]
        acl_wrapped.append(d)
    r_tfidf = TfidfRetriever(fields=("title", "abstract"), min_df=2).fit(acl_wrapped)
    rep_tfidf = evaluate_retriever(r_tfidf, queries["queries"], k=10)
    print(f"    MAP={rep_tfidf.mean_average_precision:.3f}  F1={rep_tfidf.macro_f1:.3f}")

    # ---- Task 1: Neural retriever ----
    print("\n[3] Neural retriever (title + abstract)...")
    r_neural = NeuralRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        fields=("title", "abstract"), id_field="acl_id",
    ).fit(anthology_sample)
    rep_neural = evaluate_retriever(r_neural, queries["queries"], k=10)
    print(f"    MAP={rep_neural.mean_average_precision:.3f}  F1={rep_neural.macro_f1:.3f}")

    # ---- Task 2: Chunked retriever ----
    print("\n[4] Chunked retriever (full_text, 200-word chunks)...")
    r_chunked = ChunkedNeuralRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        text_field="full_text", prefix_fields=("title",),
        id_field="acl_id", chunk_size=200, overlap=50,
    ).fit(anthology_sample)
    rep_chunked = evaluate_retriever(r_chunked, queries["queries"], k=10)
    print(f"    MAP={rep_chunked.mean_average_precision:.3f}  F1={rep_chunked.macro_f1:.3f}")

    # ---- Task 2: Mean-pool retriever ----
    print("\n[5] Mean-pool retriever (abstract + full_text)...")
    r_meanpool = MeanPoolRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        text_fields=("abstract", "full_text"), id_field="acl_id",
    ).fit(anthology_sample)
    rep_meanpool = evaluate_retriever(r_meanpool, queries["queries"], k=10)
    print(f"    MAP={rep_meanpool.mean_average_precision:.3f}  F1={rep_meanpool.macro_f1:.3f}")

    # ---- Comparison table ----
    print(f"\n{'Method':<25s} {'MAP':>8s} {'F1':>8s}")
    print("-" * 45)
    print(f"  {'TF-IDF (title+abs)':<23s} {rep_tfidf.mean_average_precision:>7.3f} {rep_tfidf.macro_f1:>7.3f}")
    print(f"  {'Neural (title+abs)':<23s} {rep_neural.mean_average_precision:>7.3f} {rep_neural.macro_f1:>7.3f}")
    print(f"  {'Chunked (full_text)':<23s} {rep_chunked.mean_average_precision:>7.3f} {rep_chunked.macro_f1:>7.3f}")
    print(f"  {'Mean-pool (full_text)':<23s} {rep_meanpool.mean_average_precision:>7.3f} {rep_meanpool.macro_f1:>7.3f}")

    # ---- LM tasks (Tasks 3, 4, 5) ----
    if with_lm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from query_transforms import (rewrite_query, retrieve_with_rewrite,
                                       generate_hypothetical_document, retrieve_with_hyde)

        print("\n[6] Loading LM...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config,
        )

        # Task 3: Query rewriting (~50 min for 98 queries)
        print("\n[7] Query rewriting experiment... (this takes ~50 min)")
        single_results = []
        for qi, q in enumerate(queries["queries"]):
            if qi % 10 == 0:
                print(f"    query {qi+1}/{len(queries['queries'])}...")
            hits, _ = retrieve_with_rewrite(q["q"], r_chunked, tokenizer, model, k=10, n_rewrites=1)
            retrieved_ids = [h.official_id for h in hits]
            raw_r = q["r"]
            relevant_ids_q = list(raw_r) if not isinstance(raw_r[0], (list, tuple)) else [p[0] for p in raw_r]
            single_results.append(evaluate_query(q["q"], retrieved_ids, relevant_ids_q))
        single_map = float(np.mean([r.average_precision for r in single_results]))
        print(f"    Single rewrite MAP: {single_map:.3f} (baseline: {rep_chunked.mean_average_precision:.3f})")

        # Task 4: HyDE (~25 min for 98 queries)
        print("\n[8] HyDE experiment... (this takes ~25 min)")
        hyde_results = []
        for qi, q in enumerate(queries["queries"]):
            if qi % 10 == 0:
                print(f"    query {qi+1}/{len(queries['queries'])}...")
            hits, _ = retrieve_with_hyde(q["q"], r_neural, tokenizer, model, k=10)
            retrieved_ids = [h.official_id for h in hits]
            raw_r = q["r"]
            relevant_ids_q = list(raw_r) if not isinstance(raw_r[0], (list, tuple)) else [p[0] for p in raw_r]
            hyde_results.append(evaluate_query(q["q"], retrieved_ids, relevant_ids_q))
        hyde_map = float(np.mean([r.average_precision for r in hyde_results]))
        print(f"    HyDE MAP: {hyde_map:.3f} (baseline: {rep_neural.mean_average_precision:.3f})")
    else:
        print("\n[6-8] Skipping LM tasks (pass --with-lm to enable).")

    print("\nPart 2 complete.")


# ==========================================================================
# Entry point
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRSE RAG Project")
    parser.add_argument("--part1", action="store_true", help="Run Part 1 (recipes)")
    parser.add_argument("--part2", action="store_true", help="Run Part 2 (ACL)")
    parser.add_argument("--with-lm", action="store_true",
                        help="Run LM-dependent tasks (requires GPU, slow)")
    args = parser.parse_args()

    if not args.part1 and not args.part2:
        print("Usage: python main.py --part1 [--part2] [--with-lm]")
        print("  --part1    Run Part 1 (recipes dataset)")
        print("  --part2    Run Part 2 (ACL papers)")
        print("  --with-lm  Include LM generation tasks (requires GPU)")
        exit(1)

    if args.part1:
        run_part1(with_lm=args.with_lm)
    if args.part2:
        run_part2(with_lm=args.with_lm)

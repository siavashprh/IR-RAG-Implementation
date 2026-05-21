"""
generation.py
-------------
LM generation for the RAG pipeline.

Two prompt templates:

Prompt A ("Structured"): Gives the LM a clear role, numbered recipes with
    specific fields, and explicit instructions to base its answer on the
    provided context only.

Prompt B ("Minimal"): Gives the LM the raw recipe text and a short
    instruction. Less guidance, more freedom — useful for comparing whether
    the LM benefits from structure.

Both use Mistral-v0.2's chat template via `tokenizer.apply_chat_template()`.
Mistral v0.2 does NOT natively support a system role, so we prepend system
instructions to the first user message.

Dataset fields in the prompt:
    We include `name`, `ingredients`, and `steps` in the prompt context.
    Rationale: `name` identifies the recipe; `ingredients` lets the LM
    reason about substitutions/allergies; `steps` lets it answer procedure
    and temperature questions. We exclude `tags` (redundant with name) and
    `description` (chatty noise). This is experimentally validated by
    comparing prompt A vs prompt B and by the irrelevant-context test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval import TfidfRetriever, RetrievalResult


# --------------------------------------------------------------------------
# Prompt template A: Structured
# --------------------------------------------------------------------------

SYSTEM_A = """\
You are a helpful cooking assistant. You answer questions about recipes \
based ONLY on the recipe context provided below. If the context does not \
contain enough information to answer the question, say so explicitly. \
Do not use any knowledge beyond the provided recipes."""

def _format_recipe(rank: int, recipe: dict, score: float | None = None) -> str:
    """Format one retrieved recipe for inclusion in the prompt."""
    parts = [f"Recipe {rank}: {recipe['name']}"]
    if score is not None:
        parts[0] += f"  (relevance score: {score:.3f})"
    parts.append(f"  Ingredients: {recipe['ingredients']}")
    parts.append(f"  Steps: {recipe['steps']}")
    return "\n".join(parts)


def build_prompt_a(query: str,
                   recipes: list[dict],
                   scores: list[float] | None = None,
                   include_scores: bool = False) -> list[dict]:
    """Build Prompt A (structured) as a messages list for apply_chat_template.

    Returns a list of {"role": ..., "content": ...} dicts. The system
    message is prepended to the user message (Mistral v0.2 doesn't support
    system role natively).
    """
    context_parts = []
    for i, recipe in enumerate(recipes, 1):
        s = scores[i - 1] if (include_scores and scores) else None
        context_parts.append(_format_recipe(i, recipe, score=s))
    context_block = "\n\n".join(context_parts)

    user_content = f"""{SYSTEM_A}

--- RECIPE CONTEXT ---
{context_block}
--- END CONTEXT ---

Question: {query}

Answer the question based on the recipes above."""

    return [{"role": "user", "content": user_content}]


# --------------------------------------------------------------------------
# Prompt template B: Minimal
# --------------------------------------------------------------------------

SYSTEM_B = """\
Answer the following cooking question using only the recipes provided."""

def build_prompt_b(query: str,
                   recipes: list[dict],
                   scores: list[float] | None = None,
                   include_scores: bool = False) -> list[dict]:
    """Build Prompt B (minimal) as a messages list."""
    recipe_texts = []
    for i, recipe in enumerate(recipes, 1):
        text = f"[{i}] {recipe['name']}: {recipe['ingredients']}. {recipe['steps']}"
        if include_scores and scores:
            text = f"[{i}] (score {scores[i-1]:.3f}) {recipe['name']}: {recipe['ingredients']}. {recipe['steps']}"
        recipe_texts.append(text)
    context = "\n".join(recipe_texts)

    user_content = f"""{SYSTEM_B}

Recipes:
{context}

Question: {query}"""

    return [{"role": "user", "content": user_content}]


# --------------------------------------------------------------------------
# Full RAG pipeline: query -> retrieve -> prompt -> generate
# --------------------------------------------------------------------------

def rag_generate(query: str,
                 retriever: "TfidfRetriever",
                 dataset,
                 tokenizer,
                 model,
                 k: int = 5,
                 prompt_fn=None,
                 include_scores: bool = False,
                 max_new_tokens: int = 512,
                 do_sample: bool = True,
                 temperature: float = 0.7) -> dict:
    """End-to-end RAG: retrieve, format prompt, generate.

    Args:
        query:          Natural language question.
        retriever:      Fitted TfidfRetriever.
        dataset:        HF Dataset (for looking up recipe fields).
        tokenizer:      HuggingFace tokenizer for the LM.
        model:          HuggingFace model (already on GPU).
        k:              Number of documents to retrieve.
        prompt_fn:      Prompt builder (default: build_prompt_a).
        include_scores: Whether to include retrieval scores in the prompt.
        max_new_tokens: Generation length limit.
        do_sample:      Sampling vs greedy decoding.
        temperature:    Sampling temperature (only if do_sample=True).

    Returns:
        Dict with keys: 'query', 'hits', 'prompt_text', 'response'.
    """
    import torch

    if prompt_fn is None:
        prompt_fn = build_prompt_a

    hits = retriever.retrieve(query, k=k)
    recipes = [dataset[h.dataset_index] for h in hits]
    scores = [h.score for h in hits]

    messages = prompt_fn(query, recipes, scores=scores,
                         include_scores=include_scores)

    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(encoded, **gen_kwargs)

    generated_ids = output_ids[0, encoded.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    prompt_text = tokenizer.decode(encoded[0], skip_special_tokens=True)

    return {
        "query": query,
        "hits": hits,
        "recipes": recipes,
        "scores": scores,
        "prompt_text": prompt_text,
        "response": response.strip(),
    }


# --------------------------------------------------------------------------
# Grounding test: swap real context for irrelevant context
# --------------------------------------------------------------------------

def rag_generate_with_fake_context(
        query: str,
        fake_context: str,
        tokenizer,
        model,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7) -> str:
    """Generate using irrelevant context instead of real recipes.

    Used for Task 7: verify the LM is actually using the provided context.
    If the model produces a sensible recipe answer from irrelevant context,
    it's relying on its own knowledge rather than the context.
    """
    import torch

    user_content = f"""{SYSTEM_A}

--- RECIPE CONTEXT ---
{fake_context}
--- END CONTEXT ---

Question: {query}

Answer the question based on the recipes above."""

    messages = [{"role": "user", "content": user_content}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(encoded, **gen_kwargs)

    generated_ids = output_ids[0, encoded.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

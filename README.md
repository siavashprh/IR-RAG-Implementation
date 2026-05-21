# IRSE RAG Project — Code

## Structure

```
main.py                  Entry point — reproduces all results
retrieval.py             TF-IDF retriever (Part 1 + Part 2 baseline)
evaluation.py            Precision/Recall/F1/MAP metrics
mwe.py                   Multi-word expression detection (gensim Phrases, NPMI)
generation.py            LM generation + prompt templates (Part 1)
neural_retrieval.py      Sentence transformer retriever (Part 2, Task 1)
chunked_retrieval.py     Chunked retriever for long documents (Part 2, Task 2)
meanpool_retrieval.py    Mean-pool sentence retriever (Part 2, Task 2)
query_transforms.py      Query rewriting + HyDE (Part 2, Tasks 3-4)
```

## Requirements

```
pip install datasets scikit-learn gensim sentence-transformers
pip install transformers==4.51.3 bitsandbytes accelerate  # for LM (GPU)
```

## Usage

```bash
# Part 1 IR experiments only (CPU, ~5 min)
python main.py --part1

# Part 2 IR experiments only (CPU, ~5 min)
python main.py --part2

# Include LM-dependent tasks (GPU required, ~2 hours)
python main.py --part1 --part2 --with-lm
```

## Important

- LM tasks require a CUDA GPU with ≥8GB VRAM (e.g., Colab T4).
- Dataset downloads require internet access to `people.cs.kuleuven.be`.

## Time estimates

| Task | Approx. time |
|---|---|
| Part 1 IR (TF-IDF, ablation, MWE, eval) | ~5 min (CPU) |
| Part 2 IR (neural, chunked, mean-pool) | ~5 min (CPU) |
| Part 1 LM (prompts, grounding, reasoning) | ~15 min (GPU) |
| Part 2 query rewriting (98 queries) | ~50 min (GPU) |
| Part 2 HyDE (98 queries) | ~25 min (GPU) |

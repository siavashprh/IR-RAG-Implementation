"""
mwe.py
------
Multi-word expression (MWE) detection for the recipes corpus.

We use gensim's `Phrases` with NPMI (Normalized Pointwise Mutual
Information) scoring:

    PMI(w1, w2)  = log( P(w1 w2) / (P(w1) * P(w2)) )
    NPMI(w1, w2) = PMI(w1, w2) / -log( P(w1 w2) )

NPMI is bounded in [-1, 1], where:
    -1 = w1 and w2 never co-occur
     0 = independent (no association)
    +1 = w1 and w2 always co-occur (one fully predicts the other)

We keep bigrams with NPMI above `threshold` (default 0.5: strong
collocation). Applying the model twice allows longer MWEs to form
(e.g. "extra_virgin_olive_oil" from "extra_virgin" + "olive_oil").

Integration with sklearn TfidfVectorizer:
- `transform_text(text) -> str` returns a string with detected MWEs
  joined by underscores.
- The retriever applies this transform to every document before fitting
  and to every query before transform. sklearn's `token_pattern` is
  configured to accept underscores so the merged tokens survive
  re-tokenization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from gensim.models.phrases import Phrases


# Match the retriever's tokenizer: lowercase alphabetic tokens of length >= 2.
# Keeping these aligned matters: gensim must see the same tokens that
# sklearn will eventually see after merging.
_TOKEN_RE = re.compile(r"\b[a-z][a-z]+\b")


def _tokenize(text: str) -> list[str]:
    """Same tokenization as TfidfRetriever's default token_pattern."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class MWEDetector:
    """Two-pass NPMI-based MWE detection via gensim Phrases.

    Two passes:
      pass 1: bigrams        (olive + oil -> olive_oil)
      pass 2: bigrams again  (extra_virgin + olive_oil -> extra_virgin_olive_oil)

    Parameters:
      min_count: ignore bigrams whose total count is below this.
                 Higher = fewer, more confident MWEs.
      threshold: NPMI score threshold in [-1, 1]. Higher = stricter.
                 0.5 is a reasonable default for "clearly collocated".
    """

    min_count: int = 20
    threshold: float = 0.5
    two_passes: bool = True

    bigram_model: Phrases | None = field(default=None, init=False)
    trigram_model: Phrases | None = field(default=None, init=False)

    # ---- fit ----

    def fit(self, texts: Iterable[str]) -> "MWEDetector":
        """Train the Phrases model(s) on a corpus of documents.

        `texts` is an iterable of pre-built document strings (e.g. the same
        strings the retriever feeds to sklearn).
        """
        tokenized = [_tokenize(t) for t in texts]

        self.bigram_model = Phrases(
            tokenized,
            min_count=self.min_count,
            threshold=self.threshold,
            delimiter="_",
            scoring="npmi",
        )

        if self.two_passes:
            # Apply the bigram model to the corpus, then train a second
            # pass on the merged tokens. Already-merged "olive_oil" can now
            # combine with another token (e.g. "virgin_olive_oil").
            merged = [list(self.bigram_model[toks]) for toks in tokenized]
            self.trigram_model = Phrases(
                merged,
                min_count=self.min_count,
                threshold=self.threshold,
                delimiter="_",
                scoring="npmi",
            )

        return self

    # ---- apply ----

    def transform_tokens(self, tokens: list[str]) -> list[str]:
        """Apply the trained model(s) to a list of tokens."""
        if self.bigram_model is None:
            raise RuntimeError("MWEDetector not fit; call .fit(texts) first.")
        out = list(self.bigram_model[tokens])
        if self.two_passes and self.trigram_model is not None:
            out = list(self.trigram_model[out])
        return out

    def transform_text(self, text: str) -> str:
        """Tokenize, merge MWEs, and rejoin as a space-separated string.

        The output is suitable for direct feeding to TfidfVectorizer.
        """
        tokens = _tokenize(text)
        merged = self.transform_tokens(tokens)
        return " ".join(merged)

    # ---- inspection ----

    def inspect_top_mwes(self, n: int = 30) -> list[tuple[str, float]]:
        """Return the top-n highest-scoring detected MWEs (bigrams only).

        Useful for sanity-checking what got merged.
        """
        if self.bigram_model is None:
            raise RuntimeError("MWEDetector not fit; call .fit(texts) first.")
        phrases = dict(self.bigram_model.export_phrases())
        items = sorted(phrases.items(), key=lambda kv: -kv[1])
        return items[:n]

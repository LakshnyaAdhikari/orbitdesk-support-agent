"""
Retrieval over the combined KB + resolved-case passage set.

Key design decision (per README.md's explicit instruction): current KB docs
outrank resolved cases, and 'superseded' cases must never be presented as
current guidance. We implement this as a score penalty rather than a hard
exclude, because a superseded case is still useful *evidence that something
changed* (see KB-005's Legacy Personal Tokens section) -- verification can
use it to add a warning instead of silently hiding it.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import Passage
from src.models import Embedder

SUPERSEDED_PENALTY = 0.35  # subtracted from similarity score for superseded cases
CASE_PENALTY = 0.05        # small penalty so KB docs win ties against resolved cases


@dataclass
class RetrievedPassage:
    passage: Passage
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Retriever:
    def __init__(self, passages: list[Passage], embedder: Embedder):
        self.passages = passages
        self.embedder = embedder
        self._passage_vectors: list[list[float]] | None = None

    def _ensure_indexed(self):
        if self._passage_vectors is None:
            texts = [f"{p.section}. {p.text}" for p in self.passages]
            self._passage_vectors = self.embedder.embed(texts)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedPassage]:
        self._ensure_indexed()
        query_vec = self.embedder.embed([query])[0]

        scored: list[RetrievedPassage] = []
        for passage, vec in zip(self.passages, self._passage_vectors):
            score = _cosine(query_vec, vec)
            if passage.source_type == "resolved_case":
                score -= CASE_PENALTY
            if passage.status == "superseded":
                score -= SUPERSEDED_PENALTY
            scored.append(RetrievedPassage(passage=passage, score=score))

        scored.sort(key=lambda rp: rp.score, reverse=True)
        return scored[:top_k]

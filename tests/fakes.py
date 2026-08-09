"""
Fake Embedder/Generator implementations used only in tests.

These exist so the required routing test can run with zero network access
and zero model downloads, and so its assertions don't depend on the exact
wording a real local LLM would produce (per the assignment's explicit ask).
They conform to the same Protocols as models.LocalEmbedder / LocalGenerator.
"""
from __future__ import annotations

import hashlib
import re


class FakeEmbedder:
    """Deterministic bag-of-words style embedding: good enough that
    semantically related test strings score higher than unrelated ones,
    without needing sentence-transformers or a network call."""

    VOCAB_DIM = 64

    def _vectorize(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self.VOCAB_DIM
        for w in words:
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.VOCAB_DIM
            vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, texts):
        return [self._vectorize(t) for t in texts]


class ScriptedGenerator:
    """Returns pre-scripted outputs in order, one per .generate() call.
    Lets a test control exactly what triage/generation 'decide' without
    depending on a real model's wording."""

    def __init__(self, scripted_outputs: list[str]):
        self._outputs = list(scripted_outputs)
        self.calls: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        self.calls.append(prompt)
        if not self._outputs:
            return ""
        return self._outputs.pop(0)


class KeywordGenerator:
    """A slightly-smarter fake: for triage prompts, returns a label based on
    simple keyword rules (mirrors what a real small instruct model would
    plausibly output); for generation prompts, echoes evidence source_ids so
    verification's citation check can pass deterministically."""

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        if prompt.startswith("You are the triage step"):
            query = prompt.split("User request:")[-1].split("Label:")[0].lower()
            if "refund" in query or "ignore" in query or "legal advice" in query:
                return "out_of_scope"
            if "not working" in query and "error" not in query:
                return "requires_clarification"
            if "already checked" in query or "two export runs" in query or "twice" in query:
                return "requires_escalation"
            return "answerable"

        if prompt.startswith("You are an OrbitDesk support assistant"):
            evidence_ids = re.findall(r"\[([A-Z]+-\d+)\]", prompt)
            cited = evidence_ids[0] if evidence_ids else "KB-000"
            # Pull real words from the evidence block so the groundedness heuristic
            # sees genuine overlap -- mirrors how a real grounded LLM answer would
            # naturally reuse terminology from the retrieved passages.
            evidence_block = prompt.split("Evidence:")[-1].split("User question:")[0]
            words = re.findall(r"[A-Za-z][A-Za-z0-9_]{4,}", evidence_block)
            key_terms = " ".join(dict.fromkeys(words[:25]))  # dedupe, keep order
            return f"Based on {cited}: {key_terms}. Follow the documented steps above to resolve this."

        return "answerable"

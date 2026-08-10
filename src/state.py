"""
Shared typed state passed between all graph nodes.

Two fields deserve a note:

- `classification`: this is the *triage* classification for most of the
  graph's life, but verification is allowed to overwrite it to
  "safe_failure" if the answer can't be salvaged after one retry. The output
  schema's enum includes "safe_failure" for exactly this reason -- it is a
  terminal state, not something triage ever picks directly.
- `retry_count`: the infinite-loop guard. verification_node only allows a
  revise-and-retry once; the conditional edge routes to safe_failure_node
  once retry_count >= MAX_RETRIES.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

MAX_RETRIES = 1

VALID_CLASSIFICATIONS = {
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
}

# Triage may only ever choose from this subset; "safe_failure" is set later.
TRIAGE_CLASSIFICATIONS = {
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
}


class Source(TypedDict):
    source_id: str
    passage: str


class SupportState(TypedDict, total=False):
    # ---- input ----
    query: str

    # ---- output-schema fields (must match data/output_schema.json exactly) ----
    classification: str
    answer: str
    sources: list[Source]
    confidence: float
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: list[str]

    # ---- internal working state, stripped before returning the final JSON ----
    retrieved: list[dict[str, Any]]   # raw RetrievedPassage-derived dicts for generation/verification
    retry_count: int
    node_trace: list[str]             # ordered list of node names executed, for logs + routing tests
    draft_answer: Optional[str]       # generation output before verification approves it
    revision_feedback: Optional[str]  # verifier feedback injected into the single retry prompt


OUTPUT_SCHEMA_FIELDS = {
    "classification", "answer", "sources", "confidence",
    "requires_human", "reason", "clarification_question", "warnings",
}


def to_output_json(state: SupportState) -> dict:
    """Strip internal-only fields, returning exactly what output_schema.json expects."""
    return {k: state[k] for k in OUTPUT_SCHEMA_FIELDS if k in state}

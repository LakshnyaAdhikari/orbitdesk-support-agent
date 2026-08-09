"""
Loads the OrbitDesk knowledge base and resolved cases into a flat list of
retrievable Passage objects. Deterministic, no model calls here on purpose --
retrieval scoring is a separate concern (see retrieval.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Passage:
    source_id: str          # e.g. "KB-003" or "CASE-1041"
    source_type: str        # "knowledge_base" | "resolved_case"
    section: str            # heading text or case title, for readability
    text: str                # the actual chunk content used for retrieval + citation
    status: str = "current"  # "current" | "superseded" | "resolved" | "escalated"
    doc_status: str = "current"  # frontmatter/document-level status (KB docs are all "current")
    metadata: dict = field(default_factory=dict)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Minimal frontmatter parser -- avoids a PyYAML dependency for a handful
    of flat key: value fields. Falls back gracefully if format is unexpected."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_block, body = match.group(1), match.group(2)
    meta = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        meta[key] = value
    return meta, body


def _chunk_by_heading(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (heading, text) chunks on '##' headings.
    Content before the first '##' (usually the '#' title + intro) becomes
    its own chunk labelled with the document title line if present."""
    lines = body.strip().splitlines()
    chunks: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("# ") and not current_lines:
            current_heading = line[2:].strip()
            continue
        if line.startswith("## "):
            if current_lines:
                chunks.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, t) for h, t in chunks if t]


def load_knowledge_base(kb_dir: str | Path) -> list[Passage]:
    kb_dir = Path(kb_dir)
    passages: list[Passage] = []

    for md_file in sorted(kb_dir.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        doc_id = meta.get("document_id", md_file.stem)
        doc_status = meta.get("status", "current")

        for heading, text in _chunk_by_heading(body):
            passages.append(
                Passage(
                    source_id=doc_id,
                    source_type="knowledge_base",
                    section=heading,
                    text=text,
                    status=doc_status,
                    doc_status=doc_status,
                    metadata={"title": meta.get("title", doc_id), "tags": meta.get("tags", [])},
                )
            )
    return passages


def load_resolved_cases(cases_path: str | Path) -> list[Passage]:
    cases_path = Path(cases_path)
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    passages: list[Passage] = []

    for case in data.get("cases", []):
        symptoms = "; ".join(case.get("symptoms", []))
        resolution = " -> ".join(case.get("resolution", []))
        limit = case.get("important_limit") or case.get("superseded_reason", "")

        text_parts = [f"Title: {case['title']}"]
        if symptoms:
            text_parts.append(f"Symptoms: {symptoms}")
        if resolution:
            text_parts.append(f"Resolution: {resolution}")
        if limit:
            text_parts.append(f"Note: {limit}")

        passages.append(
            Passage(
                source_id=case["case_id"],
                source_type="resolved_case",
                section=case["title"],
                text="\n".join(text_parts),
                status=case.get("status", "resolved"),
                doc_status=case.get("status", "resolved"),
                metadata={
                    "product_version": case.get("product_version"),
                    "source_documents": case.get("source_documents", []),
                },
            )
        )
    return passages


def load_all_passages(data_dir: str | Path) -> list[Passage]:
    data_dir = Path(data_dir)
    kb = load_knowledge_base(data_dir / "knowledge_base")
    cases = load_resolved_cases(data_dir / "resolved_cases.json")
    return kb + cases


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent / "data"
    all_passages = load_all_passages(here)
    print(f"Loaded {len(all_passages)} passages")
    for p in all_passages[:5]:
        print(f"  [{p.source_id}] ({p.status}) {p.section} -- {len(p.text)} chars")

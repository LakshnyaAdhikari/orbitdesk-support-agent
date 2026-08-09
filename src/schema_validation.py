"""Validates a final graph output against data/output_schema.json exactly."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "output_schema.json"


def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_output(output: dict) -> list[str]:
    """Returns a list of validation error messages; empty list means valid."""
    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'.'.join(str(p) for p in e.path)}: {e.message}" for e in validator.iter_errors(output)]

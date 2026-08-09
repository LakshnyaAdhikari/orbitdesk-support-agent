"""
Real CLI entry point -- uses the actual local Hugging Face models
(LocalEmbedder / LocalGenerator from src/models.py), not the test fakes.

Usage:
    python scripts/run_cli.py                     # runs all 5 sample questions
    python scripts/run_cli.py "your question here"  # runs one ad-hoc question
    python scripts/run_cli.py --log-level DEBUG "..."  # verbose node logs

First run will download ~1-3GB of model weights (embedding model is tiny,
the 1.5B generation model is the bulk of it). After that, run with your
network/wifi off to confirm the demo works fully offline, per the
assignment's requirement.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all_passages
from src.graph import run_query
from src.models import LocalEmbedder, LocalGenerator
from src.retrieval import Retriever
from src.state import to_output_json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger("orbitdesk_agent")


def load_sample_questions() -> list[dict]:
    data = json.loads((DATA_DIR / "sample_questions.json").read_text(encoding="utf-8"))
    return data["questions"]


def main():
    parser = argparse.ArgumentParser(description="OrbitDesk local support agent")
    parser.add_argument("query", nargs="?", help="A single question to ask. If omitted, runs all sample questions.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--save", action="store_true", help="Save results to outputs/sample_run_outputs.json")
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    logger.info("Loading knowledge base + resolved cases...")
    passages = load_all_passages(DATA_DIR)
    logger.info("Loaded %d passages.", len(passages))

    embedder = LocalEmbedder()
    generator = LocalGenerator()

    logger.info("Loading embedding model %s ...", embedder.model_name)
    t0 = time.time()
    retriever = Retriever(passages, embedder)
    retriever._ensure_indexed()  # forces load + indexing now, so timing below is meaningful
    logger.info("Embedding model load + index time: %.1fs", time.time() - t0)

    logger.info("Loading generation model %s (this can take a while on CPU)...", generator.model_name)
    t1 = time.time()
    generator.generate("warmup", max_new_tokens=1)  # forces load
    logger.info("Generation model load time: %.1fs", time.time() - t1)

    if args.query:
        queries = [{"question_id": "AD-HOC", "question": args.query}]
    else:
        queries = load_sample_questions()

    results = []
    for q in queries:
        logger.info("=" * 70)
        logger.info("[%s] %s", q["question_id"], q["question"])
        t_start = time.time()
        final_state = run_query(q["question"], generator, retriever)
        latency = time.time() - t_start

        output = to_output_json(final_state)
        logger.info("Node trace: %s", final_state.get("node_trace"))
        logger.info("Latency: %.1fs", latency)
        print(json.dumps(output, indent=2))

        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "node_trace": final_state.get("node_trace"),
            "latency_seconds": round(latency, 2),
            "output": output,
        })

    if args.save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        out_path = OUTPUTS_DIR / "sample_run_outputs.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()

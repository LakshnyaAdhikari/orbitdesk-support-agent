"""
Local Hugging Face model wrappers.

Design choice: both classes lazy-load their model on first real use, and both
conform to tiny Protocols (Embedder / Generator) so tests and the graph can
swap in fakes without touching real weights or the network. This is the
"clear separation between deterministic code and model reasoning" the
assignment asks for -- nodes.py depends only on the Protocol, never on
transformers/sentence-transformers directly.

Pin exact revisions here once you've confirmed them after downloading, e.g.
by running: python -c "from transformers import AutoModel; AutoModel.from_pretrained('...')"
and checking the resolved commit in ~/.cache/huggingface.
"""
from __future__ import annotations

import time
from typing import Protocol, Sequence

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATION_MODEL_REVISION = "775b11afaf83e0dc75bd5abaf90133e47b3ec082"


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Generator(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 400, sample: bool = False) -> str: ...


class LocalEmbedder:
    """Wraps sentence-transformers. Loaded lazily so importing this module
    (e.g. in tests) never triggers a download or a multi-second load."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, revision: str = EMBEDDING_MODEL_REVISION):
        self.model_name = model_name
        self.revision = revision
        self._model = None
        self.load_time_seconds: float | None = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: avoid hard dep for tests

            start = time.time()
            self._model = SentenceTransformer(self.model_name, revision=self.revision)
            self.load_time_seconds = time.time() - start

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_loaded()
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()


class LocalGenerator:
    """Wraps a small instruction-tuned causal LM via transformers. CPU-only
    by default (device_map='cpu'); swap to 'auto' if a GPU is present."""

    def __init__(self, model_name: str = GENERATION_MODEL_NAME, revision: str = GENERATION_MODEL_REVISION,
                 device: str = "cpu"):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._tokenizer = None
        self._model = None
        self.load_time_seconds: float | None = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            start = time.time()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, revision=self.revision, torch_dtype=torch.bfloat16
            ).to(self.device)
            self._model.eval()
            self.load_time_seconds = time.time() - start

    def generate(self, prompt: str, max_new_tokens: int = 400, sample: bool = False) -> str:
        """Generate deterministically by default; sample only for a revised retry.

        A retry receives verifier feedback in ``generation_node``. Sampling there
        avoids repeating the same rejected greedy-decoded answer verbatim.
        """
        self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if sample:
            generation_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            generation_kwargs.update(do_sample=False)
        output = self._model.generate(input_ids, **generation_kwargs)
        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

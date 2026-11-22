"""Embedding backends.

`SentenceTransformerEmbedder` loads any sentence-transformers compatible
checkpoint on CPU in full precision. `HashingEmbedder` is a deterministic,
model-free bag-of-subtokens embedder used by the test-suite and for debugging
the pipeline; it is not meant for real retrieval experiments.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import ModelConfig, RerankConfig


class Embedder(Protocol):
    model_key: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def describe(self) -> dict: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _prefix_tag(prefix: str) -> str:
    return hashlib.sha256(prefix.encode()).hexdigest()[:8]


_SUBTOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def code_subtokens(text: str) -> list[str]:
    """Identifiers plus their camelCase / snake_case parts, lower-cased."""
    out: list[str] = []
    for word in _WORD.findall(text):
        lw = word.lower()
        out.append(lw)
        parts = [p.lower() for p in _SUBTOKEN.findall(word)]
        if len(parts) > 1:
            out.extend(parts)
    return out


class HashingEmbedder:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.dim = cfg.hashing_dim
        self.model_key = f"hashing|dim={self.dim}|dp={_prefix_tag(cfg.document_prefix)}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in code_subtokens(t):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(out)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.cfg.document_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.cfg.query_prefix + t for t in texts])

    def describe(self) -> dict:
        return {"backend": "hashing", "dim": self.dim}


def _seed_torch(seed: int, num_threads: int | None) -> None:
    import torch

    torch.manual_seed(seed)
    if num_threads:
        torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


class SentenceTransformerEmbedder:
    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._model = None
        # Everything that changes the produced vectors (except the input text,
        # which is covered by the chunk content hash) goes into the key.
        self.model_key = (
            f"st|{cfg.name}@{cfg.revision or 'latest'}|len={cfg.max_seq_length}"
            f"|dp={_prefix_tag(cfg.document_prefix)}"
        )

    @property
    def model(self):
        if self._model is None:
            _seed_torch(self.seed, self.cfg.num_threads)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.cfg.name,
                revision=self.cfg.revision,
                device=self.cfg.device,
                trust_remote_code=self.cfg.trust_remote_code,
            )
            self._model.max_seq_length = self.cfg.max_seq_length
            self._model.eval()
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            vecs = self.model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        return _normalize(vecs)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.cfg.document_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.cfg.query_prefix + t for t in texts])

    def describe(self) -> dict:
        info = {"backend": "sentence-transformers", "name": self.cfg.name,
                "revision": self.cfg.revision, "max_seq_length": self.cfg.max_seq_length}
        try:
            import sentence_transformers
            import torch
            import transformers

            info["versions"] = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
            }
        except ImportError:
            pass
        return info


def make_embedder(cfg: ModelConfig, seed: int = 0) -> Embedder:
    if cfg.backend == "hashing":
        return HashingEmbedder(cfg)
    return SentenceTransformerEmbedder(cfg, seed=seed)


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self._model = None

    @property
    def model(self):
        if self._model is None:
            _seed_torch(self.seed, None)
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.cfg.name,
                revision=self.cfg.revision,
                device="cpu",
                max_length=self.cfg.max_length,
                trust_remote_code=self.cfg.trust_remote_code,
            )
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        import torch

        with torch.inference_mode():
            scores = self.model.predict(
                [(query, d) for d in docs],
                batch_size=self.cfg.batch_size,
                show_progress_bar=False,
            )
        return [float(s) for s in scores]

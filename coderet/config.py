"""Configuration: typed dataclasses loaded from TOML.

Unknown keys are rejected so that typos never silently change an experiment.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "coderet.toml"


@dataclass
class ModelConfig:
    # "sentence-transformers" (real model) or "hashing" (deterministic, model-free;
    # for tests and pipeline debugging only).
    backend: str = "sentence-transformers"
    # Default: CodeRankEmbed (137M params, 8k context) - strong on code retrieval
    # benchmarks while remaining usable on CPU.
    name: str = "nomic-ai/CodeRankEmbed"
    # Pin a commit hash here for reproducible experiments.
    revision: str | None = None
    query_prefix: str = "Represent this query for searching relevant code: "
    document_prefix: str = ""
    trust_remote_code: bool = True
    max_seq_length: int = 1024
    batch_size: int = 8
    device: str = "cpu"
    # None = library default. Set e.g. 2 to keep the machine responsive.
    num_threads: int | None = None
    hashing_dim: int = 512


@dataclass
class RerankConfig:
    # Optional cross-encoder applied to the fused candidate pool.
    enabled: bool = False
    name: str = "jinaai/jina-reranker-v2-base-multilingual"
    revision: str | None = None
    trust_remote_code: bool = True
    max_length: int = 1024
    batch_size: int = 4
    # How many fused candidates are rescored.
    pool: int = 30


@dataclass
class ChunkingConfig:
    # Syntax units larger than this are split into nested units or line windows.
    max_chunk_chars: int = 3000
    window_lines: int = 60
    window_overlap: int = 10
    # Chunks shorter than this (after stripping) are dropped.
    min_chunk_chars: int = 20
    # Prepend "File: ... / Symbol: ..." to the embedded text.
    embed_header: bool = True


@dataclass
class IndexConfig:
    # Index database location, relative to the repository root.
    path: str = ".coderet/index.db"
    max_file_bytes: int = 1_000_000
    # Files processed per transaction.
    file_batch: int = 32
    # Chunks fetched from the DB per embedding round (model batches are smaller).
    embed_fetch: int = 64
    # Use `git ls-files` when the repository is a git checkout.
    use_git: bool = True
    exclude_dirs: list[str] = field(
        default_factory=lambda: [
            ".git", ".hg", ".svn", ".coderet", "node_modules", ".venv", "venv",
            "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "build", "dist",
            "target", ".idea", ".vscode",
        ]
    )
    # None = every extension known to coderet.languages.
    extensions: list[str] | None = None


@dataclass
class SearchConfig:
    # "hybrid", "dense" or "lexical".
    mode: str = "hybrid"
    top_k: int = 10
    # Candidates taken from each retriever before fusion.
    candidate_pool: int = 100
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    # Embedding rows scanned per step during dense search (bounds memory).
    scan_batch: int = 2048


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _merge(obj: Any, data: dict[str, Any], where: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key '{where}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config section '{where}{key}' must be a table")
            _merge(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, repo: str | Path | None = None) -> Config:
    """Load config from `path`, else `<repo>/coderet.toml` if present, else defaults."""
    cfg = Config()
    candidate: Path | None = None
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"config file not found: {candidate}")
    elif repo is not None and (Path(repo) / CONFIG_FILENAME).is_file():
        candidate = Path(repo) / CONFIG_FILENAME
    if candidate is not None:
        with open(candidate, "rb") as fh:
            _merge(cfg, tomllib.load(fh), "")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.search.mode not in {"hybrid", "dense", "lexical"}:
        raise ValueError(f"search.mode must be hybrid|dense|lexical, got {cfg.search.mode!r}")
    if cfg.model.backend not in {"sentence-transformers", "hashing"}:
        raise ValueError(f"model.backend must be sentence-transformers|hashing, got {cfg.model.backend!r}")
    if cfg.chunking.window_overlap >= cfg.chunking.window_lines:
        raise ValueError("chunking.window_overlap must be smaller than chunking.window_lines")
    for name in ("batch_size",):
        if getattr(cfg.model, name) < 1:
            raise ValueError(f"model.{name} must be >= 1")

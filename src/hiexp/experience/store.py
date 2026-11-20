"""HEK store: load JSONL by level, encode titles, retrieve by query.

Disk layout (paper §3.1 + §A; chosen by us where the paper does not
specify):

    <hek_path>/
      e1.jsonl              # one ExperienceEntry per line, level=1
      e2.jsonl              # level=2 (default strategy injection level)
      e3.jsonl              # level=3 (optional, for ablation)
      .embeddings/
        e1.npy              # (n_e1, d) L2-normalized passage embeddings
        e1.fingerprint.txt  # SHA1 of e1.jsonl bytes; cache invalidation
        e2.npy / e2.fingerprint.txt
        e3.npy / e3.fingerprint.txt

Retrieval API:
    retrieve_e2(query, top_k=5)               # paper §A
    retrieve_e1(query, threshold=0.8, top_k=K)  # paper §A
    retrieve_strategic(query, ...)            # E2 or E3, picks per config
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hiexp.experience.embedder import E5Embedder, Embedder
from hiexp.experience.schema import ExperienceEntry, ExperienceHit, ExperienceLevel

logger = logging.getLogger(__name__)

_FILES_BY_LEVEL: dict[int, str] = {1: "e1.jsonl", 2: "e2.jsonl", 3: "e3.jsonl"}


def _read_jsonl(path: Path) -> list[ExperienceEntry]:
    if not path.exists():
        return []
    entries: list[ExperienceEntry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        entries.append(ExperienceEntry.from_jsonl(line))
    return entries


def _write_jsonl(path: Path, entries: list[ExperienceEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(e.to_jsonl() for e in entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )


def _file_fingerprint(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha1(path.read_bytes()).hexdigest()


@dataclass
class _LevelIndex:
    entries: list[ExperienceEntry]
    embeddings: np.ndarray  # (n, d), L2-normalized; (0, 0) when empty


class HEKStore:
    """In-memory hierarchical experience knowledge store.

    Construct via :meth:`load` (reads JSONL + cached embeddings) or by
    instantiating empty and calling :meth:`add_entries`.

    The store does not own the embedder; pass one in or use the default
    :class:`E5Embedder` (lazy-loaded multilingual-e5-base).
    """

    def __init__(
        self,
        root: str | Path,
        embedder: Embedder | None = None,
    ) -> None:
        self.root = Path(root)
        self.embedder: Embedder = embedder or E5Embedder()
        self._levels: dict[int, _LevelIndex] = {
            1: _LevelIndex(entries=[], embeddings=np.zeros((0, 0), dtype=np.float32)),
            2: _LevelIndex(entries=[], embeddings=np.zeros((0, 0), dtype=np.float32)),
            3: _LevelIndex(entries=[], embeddings=np.zeros((0, 0), dtype=np.float32)),
        }

    # ── Loading ─────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        root: str | Path,
        embedder: Embedder | None = None,
        *,
        rebuild_embeddings: bool = False,
    ) -> "HEKStore":
        store = cls(root, embedder=embedder)
        for level in (1, 2, 3):
            store._load_level(level, rebuild_embeddings=rebuild_embeddings)
        return store

    def _load_level(self, level: int, *, rebuild_embeddings: bool) -> None:
        jsonl_path = self.root / _FILES_BY_LEVEL[level]
        emb_dir = self.root / ".embeddings"
        emb_path = emb_dir / f"e{level}.npy"
        fp_path = emb_dir / f"e{level}.fingerprint.txt"

        entries = _read_jsonl(jsonl_path)
        if not entries:
            return

        current_fp = _file_fingerprint(jsonl_path)

        if (
            not rebuild_embeddings
            and emb_path.exists()
            and fp_path.exists()
            and fp_path.read_text(encoding="utf-8").strip() == current_fp
        ):
            embeddings = np.load(emb_path)
            if embeddings.shape[0] != len(entries):
                logger.warning(
                    "Cached embeddings shape mismatch for level %d (cached=%d, entries=%d); "
                    "rebuilding.",
                    level,
                    embeddings.shape[0],
                    len(entries),
                )
                embeddings = self._encode_titles(entries)
                emb_dir.mkdir(parents=True, exist_ok=True)
                np.save(emb_path, embeddings)
                fp_path.write_text(current_fp, encoding="utf-8")
        else:
            embeddings = self._encode_titles(entries)
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, embeddings)
            fp_path.write_text(current_fp, encoding="utf-8")

        self._levels[level] = _LevelIndex(entries=entries, embeddings=embeddings)

    def _encode_titles(self, entries: list[ExperienceEntry]) -> np.ndarray:
        titles = [e.title for e in entries]
        return self.embedder.encode_passages(titles)

    # ── Mutation ────────────────────────────────────────────────────────────

    def add_entries(self, entries: list[ExperienceEntry]) -> None:
        """Append entries (any mix of levels). Re-encodes affected levels.

        Persists JSONL and embeddings cache to disk as a side effect so a
        subsequent :meth:`load` call sees the same state.
        """
        if not entries:
            return
        by_level: dict[int, list[ExperienceEntry]] = {}
        for e in entries:
            by_level.setdefault(int(e.level), []).append(e)

        for level, new_entries in by_level.items():
            existing = self._levels[level].entries
            merged = existing + new_entries
            embeddings = self._encode_titles(merged)
            self._levels[level] = _LevelIndex(entries=merged, embeddings=embeddings)
            jsonl_path = self.root / _FILES_BY_LEVEL[level]
            _write_jsonl(jsonl_path, merged)
            emb_dir = self.root / ".embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(emb_dir / f"e{level}.npy", embeddings)
            (emb_dir / f"e{level}.fingerprint.txt").write_text(
                _file_fingerprint(jsonl_path), encoding="utf-8"
            )

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve_e1(
        self,
        query: str,
        threshold: float = 0.8,
        top_k: int = 3,
    ) -> list[ExperienceHit]:
        """Threshold-then-top-k retrieval over E1.

        Paper §A:
            > "We apply a 0.8 similarity threshold for case-based
            > experiences (E1) to ensure high precision."

        ``top_k`` defaults to 3 — the paper does not pin a number; see
        plan §10.
        """
        return self._retrieve(level=1, query=query, threshold=threshold, top_k=top_k)

    def retrieve_e2(self, query: str, top_k: int = 5) -> list[ExperienceHit]:
        """Top-k retrieval over E2 (no threshold).

        Paper §A: ``"For strategy-based experiences (E2 or E3), we
        select the top-5 candidates"``.
        """
        return self._retrieve(level=2, query=query, threshold=None, top_k=top_k)

    def retrieve_e3(self, query: str, top_k: int = 5) -> list[ExperienceHit]:
        return self._retrieve(level=3, query=query, threshold=None, top_k=top_k)

    def retrieve_strategic(
        self,
        query: str,
        top_k: int = 5,
        *,
        use_e3: bool = False,
    ) -> list[ExperienceHit]:
        """Pick E2 (default) or E3 according to ``use_e3``.

        Paper Table 2 shows E2+E1 outperforms E3+E1, so default is E2.
        """
        return self.retrieve_e3(query, top_k) if use_e3 else self.retrieve_e2(query, top_k)

    def _retrieve(
        self,
        *,
        level: ExperienceLevel,
        query: str,
        threshold: float | None,
        top_k: int,
    ) -> list[ExperienceHit]:
        idx = self._levels[level]
        if not idx.entries or top_k <= 0:
            return []
        q_vec = self.embedder.encode_query(query)
        # Embeddings are L2-normalized → cosine similarity = dot product.
        scores = idx.embeddings @ q_vec
        order = np.argsort(-scores)  # descending
        hits: list[ExperienceHit] = []
        for i in order:
            score = float(scores[i])
            if threshold is not None and score < threshold:
                # Sorted descending → no later element can pass the threshold.
                break
            hits.append(ExperienceHit(entry=idx.entries[int(i)], score=score))
            if len(hits) >= top_k:
                break
        return hits

    # ── Introspection ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return sum(len(idx.entries) for idx in self._levels.values())

    def count(self, level: int) -> int:
        return len(self._levels[level].entries)

    def __repr__(self) -> str:
        sizes = {f"E{lvl}": self.count(lvl) for lvl in (1, 2, 3)}
        return f"HEKStore(root={self.root!s}, sizes={sizes})"

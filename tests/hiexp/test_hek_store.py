from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hiexp.experience import ExperienceEntry, ExperienceHit, HEKStore


class _DeterministicEmbedder:
    """Stub embedder for tests.

    Each input string is hashed deterministically into a unit vector in a
    small dimensional space so cosine similarity is a pure function of
    the input. We boost similarity between strings that share whole
    words by encoding word-level shingles into a fixed lexicon.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._lexicon: dict[str, int] = {}

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().replace("query: ", "").replace("passage: ", "").split():
            idx = self._lexicon.setdefault(word, len(self._lexicon) % self.dim)
            v[idx] += 1.0
        norm = float(np.linalg.norm(v))
        if norm == 0:
            v[0] = 1.0
            norm = 1.0
        return v / norm

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(f"passage: {t}") for t in texts]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(f"query: {text}").astype(np.float32)


def _make_entry(level: int, title: str, description: str, **extra) -> ExperienceEntry:
    return ExperienceEntry(level=level, title=title, description=description, **extra)


# ── Schema ────────────────────────────────────────────────────────────────


def test_entry_auto_id():
    e = _make_entry(1, "Identify the poet first", "...")
    assert e.id.startswith("E1-")
    assert len(e.id) == 3 + 12  # "E1-" + 12 hex chars


def test_entry_explicit_id_preserved():
    e = _make_entry(2, "Multi-hop pattern", "...", id="custom-id")
    assert e.id == "custom-id"


def test_entry_jsonl_roundtrip():
    e1 = _make_entry(1, "alpha", "alpha desc", tags=["x", "y"])
    line = e1.to_jsonl()
    e2 = ExperienceEntry.from_jsonl(line)
    assert e2.level == 1
    assert e2.title == "alpha"
    assert e2.tags == ["x", "y"]
    assert e2.id == e1.id


# ── Loading ───────────────────────────────────────────────────────────────


def _seed_store(root: Path) -> None:
    e1_entries = [
        _make_entry(1, "Verify temporal anchor before searching", "Resolve the year first."),
        _make_entry(1, "Confirm director identity uniquely", "Cross-check across 2 sources."),
        _make_entry(1, "Avoid premiere date confusion", "Premiere ≠ composition month."),
    ]
    e2_entries = [
        _make_entry(2, "Multi-hop constraint decomposition", "Break into sequential sub-queries."),
        _make_entry(2, "Temporal-then-attribute pattern", "Resolve time first, then attribute."),
    ]
    (root / "e1.jsonl").write_text(
        "\n".join(e.to_jsonl() for e in e1_entries) + "\n", encoding="utf-8"
    )
    (root / "e2.jsonl").write_text(
        "\n".join(e.to_jsonl() for e in e2_entries) + "\n", encoding="utf-8"
    )


def test_load_empty_dir(tmp_path):
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    assert len(store) == 0
    assert store.count(1) == 0
    assert store.count(2) == 0


def test_load_populated_dir(tmp_path):
    _seed_store(tmp_path)
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    assert store.count(1) == 3
    assert store.count(2) == 2
    assert store.count(3) == 0
    # Cache files written
    assert (tmp_path / ".embeddings" / "e1.npy").exists()
    assert (tmp_path / ".embeddings" / "e2.npy").exists()
    assert (tmp_path / ".embeddings" / "e1.fingerprint.txt").exists()


def test_load_uses_cached_embeddings(tmp_path):
    _seed_store(tmp_path)
    embedder = _DeterministicEmbedder()
    store1 = HEKStore.load(tmp_path, embedder=embedder)
    cached_e1 = np.load(tmp_path / ".embeddings" / "e1.npy")

    # Mutate the cache to a sentinel so we can detect re-encoding.
    sentinel = np.full_like(cached_e1, 0.123, dtype=np.float32)
    np.save(tmp_path / ".embeddings" / "e1.npy", sentinel)

    store2 = HEKStore.load(tmp_path, embedder=embedder)
    assert np.allclose(store2._levels[1].embeddings, sentinel)


def test_load_rebuild_when_jsonl_changes(tmp_path):
    _seed_store(tmp_path)
    embedder = _DeterministicEmbedder()
    HEKStore.load(tmp_path, embedder=embedder)

    # Change JSONL → fingerprint changes → cache must be rebuilt.
    extra = _make_entry(1, "New entry", "new desc")
    with (tmp_path / "e1.jsonl").open("a", encoding="utf-8") as f:
        f.write(extra.to_jsonl() + "\n")

    store = HEKStore.load(tmp_path, embedder=embedder)
    assert store.count(1) == 4


# ── Retrieval ─────────────────────────────────────────────────────────────


def test_retrieve_e2_top_k(tmp_path):
    _seed_store(tmp_path)
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    hits = store.retrieve_e2("How to handle multi-hop questions with constraints", top_k=2)
    assert len(hits) <= 2
    assert all(isinstance(h, ExperienceHit) for h in hits)
    # Top hit should mention "multi-hop" or "constraint" via lexicon overlap.
    titles = [h.entry.title for h in hits]
    assert any("Multi-hop" in t or "constraint" in t.lower() for t in titles)


def test_retrieve_e1_threshold_filters_below(tmp_path):
    _seed_store(tmp_path)
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    # Threshold 0.99 → almost nothing passes.
    high = store.retrieve_e1("entirely unrelated content xyz123", threshold=0.99, top_k=3)
    assert high == []
    # Threshold 0.0 → top_k bound dominates.
    low = store.retrieve_e1("temporal anchor", threshold=0.0, top_k=2)
    assert len(low) == 2
    assert low[0].score >= low[1].score


def test_retrieve_e1_default_threshold_keeps_relevant(tmp_path):
    _seed_store(tmp_path)
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    # The query shares all tokens with the entry → high cosine.
    hits = store.retrieve_e1("Verify temporal anchor before searching", threshold=0.8, top_k=3)
    assert len(hits) >= 1
    assert hits[0].score >= 0.8
    assert hits[0].entry.title == "Verify temporal anchor before searching"


def test_retrieve_strategic_uses_e3_when_flag_set(tmp_path):
    _seed_store(tmp_path)
    # Add E3 entries.
    e3 = [
        _make_entry(3, "Global decomposition strategy", "..."),
    ]
    (tmp_path / "e3.jsonl").write_text(e3[0].to_jsonl() + "\n", encoding="utf-8")
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    hits_e2 = store.retrieve_strategic("multi-hop", top_k=5, use_e3=False)
    hits_e3 = store.retrieve_strategic("global decomposition", top_k=5, use_e3=True)
    assert all(h.entry.level == 2 for h in hits_e2)
    assert all(h.entry.level == 3 for h in hits_e3)


def test_retrieve_returns_empty_on_empty_level(tmp_path):
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    assert store.retrieve_e2("anything", top_k=5) == []
    assert store.retrieve_e1("anything", top_k=3) == []


# ── Mutation ──────────────────────────────────────────────────────────────


def test_add_entries_persists_to_disk(tmp_path):
    embedder = _DeterministicEmbedder()
    store = HEKStore.load(tmp_path, embedder=embedder)
    entries = [
        _make_entry(1, "fresh case", "case desc"),
        _make_entry(2, "fresh strategy", "strategy desc"),
    ]
    store.add_entries(entries)

    jsonl = (tmp_path / "e1.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl) == 1
    assert json.loads(jsonl[0])["title"] == "fresh case"

    reopened = HEKStore.load(tmp_path, embedder=embedder)
    assert reopened.count(1) == 1
    assert reopened.count(2) == 1
    hits = reopened.retrieve_e1("fresh case", threshold=0.5, top_k=3)
    assert len(hits) == 1
    assert hits[0].entry.title == "fresh case"


# ── Repr / introspection ──────────────────────────────────────────────────


def test_repr(tmp_path):
    _seed_store(tmp_path)
    store = HEKStore.load(tmp_path, embedder=_DeterministicEmbedder())
    text = repr(store)
    assert "E1" in text and "E2" in text
    assert "3" in text  # 3 E1 entries

"""HEK (Hierarchical Experience Knowledge) data layer.

Public:
    ExperienceEntry / ExperienceHit / ExperienceLevel — paper Table 11
        schema (``schema.py``).
    Embedder protocol + E5Embedder — embedding interface used by the
        store (``embedder.py``).
    HEKStore — JSONL + npy disk layout, e5-prefix encoding, parent-child
        retrieval (``store.py``).

Filled in by Milestone 4: ``construction.py`` (offline contrastive
distillation) and ``clustering.py`` (Ward-linkage agglomerative
clustering, paper §C).
"""

from hiexp.experience.embedder import E5Embedder, Embedder
from hiexp.experience.schema import ExperienceEntry, ExperienceHit, ExperienceLevel
from hiexp.experience.store import HEKStore

__all__ = [
    "ExperienceEntry",
    "ExperienceHit",
    "ExperienceLevel",
    "Embedder",
    "E5Embedder",
    "HEKStore",
]

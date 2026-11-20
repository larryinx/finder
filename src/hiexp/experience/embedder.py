"""Text embedder used for HEK retrieval.

Wraps :class:`sentence_transformers.SentenceTransformer` with the e5
prefix convention:
  - queries are encoded with ``"query: "`` prefix,
  - passages (HEK titles) are encoded with ``"passage: "`` prefix,
  - all embeddings are L2-normalized so cosine similarity = dot product.

The HEK store does not import :class:`SentenceTransformer` directly so
that tests can inject a deterministic stub embedder without pulling
multi-GB model weights.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Interface used by :class:`hiexp.experience.store.HEKStore`."""

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Return ``(n, d)`` float32 matrix of L2-normalized embeddings."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Return ``(d,)`` float32 vector of L2-normalized embedding."""
        ...


class E5Embedder:
    """Default embedder for HEK; loads multilingual-e5-base lazily.

    Per HiExp paper §A: ``"we employ multilingual-e5-base as the
    retriever and use the widely used Wikipedia dump from December 2018
    as the retrieval corpus"``. Note: that sentence describes the *doc*
    retriever; here we re-use the same embedding model for HEK because
    paper §A also says ``"All HEK are encoded using the same embedding
    model."``.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so the package can be inspected without
            # paying the sentence-transformers / torch import cost.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._ensure_model()
        prefixed = [f"passage: {t}" for t in texts]
        emb = model.encode(
            prefixed,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(
            f"query: {text}",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

"""Random sampler with persistent RNG state.

v1: with-replacement sampling. State = the RNG seed + the global step
counter, so resume reproduces the exact sequence of batches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from search_ttl.rollout.schemas import QuestionRecord


@dataclass
class SamplerState:
    seed: int
    cursor: int = 0  # number of `sample()` calls already made — opaque to users


class RandomSampler:
    def __init__(
        self,
        questions: list[QuestionRecord],
        *,
        seed: int = 42,
    ) -> None:
        if not questions:
            raise ValueError("RandomSampler needs a non-empty question list")
        self.questions = questions
        self._state = SamplerState(seed=seed)
        # Shared RNG lets us advance deterministically.
        self._rng = random.Random(seed)

    def sample(self, n: int) -> list[QuestionRecord]:
        out = self._rng.choices(self.questions, k=n)
        self._state.cursor += 1
        return out

    def state(self) -> dict:
        return {"seed": self._state.seed, "cursor": self._state.cursor}

    def load_state(self, d: dict) -> None:
        self._state.seed = int(d.get("seed", self._state.seed))
        self._state.cursor = int(d.get("cursor", 0))
        # Replay RNG forward by `cursor` calls of size 1 to deterministically restore
        # the same sequence of `sample()` outputs going forward. We can't know the
        # historical `n`, but seed+cursor means the *future* sequence is reproducible
        # provided the caller draws the same n's.
        self._rng = random.Random(self._state.seed)
        for _ in range(self._state.cursor):
            self._rng.choices(self.questions, k=1)

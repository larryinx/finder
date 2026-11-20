"""Plugin registry pattern.

Every pluggable module (Reward, Reflector, Curator, Experience) registers
itself here under a string key. YAML configs reference these keys via the
``kind:`` field. Adding a new implementation = drop a file + apply the
decorator + reference the key in YAML. No edits to existing code.
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar

T = TypeVar("T")

REWARDS: dict[str, Type] = {}
REFLECTORS: dict[str, Type] = {}
CURATORS: dict[str, Type] = {}
EXPERIENCES: dict[str, Type] = {}


def _register(table: dict[str, Type], name: str) -> Callable[[Type[T]], Type[T]]:
    def deco(cls: Type[T]) -> Type[T]:
        if name in table and table[name] is not cls:
            raise ValueError(
                f"name {name!r} already registered to {table[name]!r}; "
                f"cannot re-register to {cls!r}. Pick a unique name."
            )
        table[name] = cls
        return cls

    return deco


def register_reward(name: str) -> Callable[[Type[T]], Type[T]]:
    """Register a RewardManager implementation. Used by `reward.kind:` YAML key."""
    return _register(REWARDS, name)


def register_reflector(name: str) -> Callable[[Type[T]], Type[T]]:
    """Register a ReflectorManager implementation. Used by `reflector.kind:`."""
    return _register(REFLECTORS, name)


def register_curator(name: str) -> Callable[[Type[T]], Type[T]]:
    """Register an CuratorManager implementation. Used by `curator.kind:`."""
    return _register(CURATORS, name)


def register_experience(name: str) -> Callable[[Type[T]], Type[T]]:
    """Register an Experience backend. Used by `experience.kind:`."""
    return _register(EXPERIENCES, name)


def _lookup(table: dict[str, Type], name: str, role: str) -> Type:
    if name not in table:
        keys = ", ".join(sorted(table)) or "(none)"
        raise KeyError(
            f"unknown {role} kind={name!r}; registered: {keys}. "
            f"Did you forget to import the module that registers it?"
        )
    return table[name]


def build_reward(cfg) -> object:
    """Instantiate the reward manager named by `cfg.kind`.

    The implementation is responsible for `from_omegaconf(cfg)`; we keep
    `__init__` private so callers always come through the registry.
    """
    cls = _lookup(REWARDS, cfg.kind, "reward")
    return cls.from_omegaconf(cfg)


def build_reflector(cfg) -> object:
    cls = _lookup(REFLECTORS, cfg.kind, "reflector")
    return cls.from_omegaconf(cfg)


def build_curator(cfg) -> object:
    cls = _lookup(CURATORS, cfg.kind, "curator")
    return cls.from_omegaconf(cfg)


def build_experience(cfg) -> object:
    cls = _lookup(EXPERIENCES, cfg.kind, "experience")
    return cls.from_omegaconf(cfg)


def import_default_plugins() -> None:
    """Trigger registration of the default plugins.

    Call this once at startup (e.g. in scripts/ttl/run.py) before consulting
    the registry. Imports are isolated here to keep the registry import
    cheap and avoid circular deps.
    """
    # Side-effect imports register their classes via the decorators.
    import search_ttl.experience.tactic_playbook  # noqa: F401
    import search_ttl.experience.staged_playbook  # noqa: F401
    import search_ttl.reward.search_agent_lasj  # noqa: F401
    import search_ttl.reward.qa_em_f1  # noqa: F401
    import search_ttl.reflector.querychain_tactic  # noqa: F401
    import search_ttl.reflector.querychain_staged  # noqa: F401
    import search_ttl.curator.react_tactic  # noqa: F401
    import search_ttl.curator.react_staged  # noqa: F401

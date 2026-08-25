# Adding an experience backend

The experience is the playbook — the only state that learns. The agent
sees `render()` inside the `<experience>` block of its system prompt; the
curator edits it through the `Experience` API; the trainer persists
`to_text()` and rebuilds it with `from_text()` on resume and on rollback.

Existing backends in `src/search_ttl/experience/`:

| kind | Sections | Bullet ids | Pairs with |
|---|---|---|---|
| `staged_playbook` | `FIND`, `VERIFY`, `DECIDE` (fixed; each with a preamble) | `F-001`, `V-002`, `D-003` | `querychain_staged` + `react_staged` |
| `tactic_playbook` | `TACTIC_A`, `TACTIC_B`, `TACTIC_C` (fixed) | `A-001`, … | `querychain_tactic` + `react_tactic` |

## Interface

```python
# src/search_ttl/experience/base.py
class Experience(ABC):
    @classmethod
    def from_omegaconf(cls, cfg) -> "Experience": ...        # honours cfg.init_from
    @classmethod
    def from_text(cls, text: str, *, cfg=None) -> "Experience": ...   # inverse of to_text()

    def render(self) -> str: ...                               # what the agent sees
    def to_text(self) -> str: ...                              # what is persisted (may equal render())
    def list_sections(self) -> list[SectionInfo]: ...          # SectionInfo(name, bullet_count, char_count)
    def read_section(self, section: str) -> list[Bullet]: ...
    def read_bullet(self, bullet_id: str) -> Bullet | None: ...
    def search_bullets(self, keyword: str, top_k: int = 5) -> list[Bullet]: ...
    def char_count(self) -> int: ...                           # len(render()) by convention

    # writes return a NEW instance — never mutate self
    def add_bullet(self, section, content, *, metadata=None) -> tuple["Experience", Bullet]: ...
    def modify_bullet(self, bullet_id, new_content, *, metadata_patch=None) -> tuple["Experience", Bullet | None]: ...
    def delete_bullet(self, bullet_id, *, reason="") -> tuple["Experience", bool]: ...
```

Contract details:

- **Value semantics.** Curators keep the previous instance around to roll
  back, so writes must copy.
- **Round trip.** `from_text(to_text())` must reproduce the playbook
  exactly — it is how checkpoints resume, how `eval_experience.py` loads a
  frozen `best.md`, and how curators roll back a rejected step
  (`type(experience).from_text(pre_text)`).
- **`render()` is in every agent prompt.** Keep it compact and stable;
  the empty playbook should render something the agent can ignore
  (the agent treats the sentinel `"(none yet)"` as "no experience").
- **`char_count()` is the budget unit.** Return `len(self.render())`;
  curators compare it with `curator.char_budget`.
- **Section names are the contract with the reflector and the curator**:
  the reflector's `section_hint` values and the curator's routing must be
  members of `list_sections()`.
- `search_bullets` backs the curator's `search_bullets` tool (used to
  find near-duplicates before adding) — a simple token-overlap ranking is
  enough.

## Skeleton

Copy `staged_playbook.py` and adjust `_SECTION_ORDER`, the per-section
preamble, and the bullet-id prefixes. The important parts:

```python
_SECTION_ORDER = ["PLAN", "SEARCH", "ANSWER"]
_SLUG = {"PLAN": "P", "SEARCH": "S", "ANSWER": "A"}

@register_experience("my_playbook")
class MyPlaybook(Experience):
    def __init__(self, bullets: dict[str, list[Bullet]]) -> None: ...

    @classmethod
    def from_omegaconf(cls, cfg) -> "MyPlaybook":
        init_from = cfg.get("init_from", None) if cfg is not None else None
        if init_from:
            return cls.from_text(Path(init_from).read_text(encoding="utf-8"), cfg=cfg)
        return cls(bullets={sec: [] for sec in _SECTION_ORDER})

    @classmethod
    def from_text(cls, text: str, *, cfg=None) -> "MyPlaybook":
        ...   # parse "## <section>" headers and "[<id>] :: <content>" bullets (multi-line content
              #  accumulates until the next bullet / header)

    def render(self) -> str: ...        # "# Search Agent Experience" + one "## <section>" block each
    def to_text(self) -> str: return self.render()
    def char_count(self) -> int: return len(self.render())
    def add_bullet(self, section, content, *, metadata=None):
        new = self._copy(); bullet = Bullet(bullet_id=self._next_id(section), section=section, content=content)
        new._bullets[section].append(bullet); return new, bullet
```

## Register and configure

1. Add the import to `import_default_plugins()` in `src/search_ttl/trainer/registry.py`.
2. Create `config/ttl/experience/my_playbook.yaml`:

```yaml
experience:
  kind: my_playbook
  init_from: null      # or a path to seed the playbook from an existing file
```

3. Provide (or adapt) a reflector and a curator that use your section
   names, then run `python scripts/ttl/run.py --experience my_playbook
   --reflector … --curator … `.

## Tests to add or update

- `tests/search_ttl/test_registry.py` — the kind registers.
- A round-trip test: build a playbook, `add`/`modify`/`delete`, assert
  `from_text(to_text())` equals it and that the original instance is
  unchanged (value semantics).
- `tests/search_ttl/test_curator_char_budget.py` and
  `tests/search_ttl/test_step_schedule.py` construct `StagedPlaybook`
  directly; parametrise or copy them if your backend should be covered
  by the same guards.

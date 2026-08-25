# Adding a curator

The curator is the "optimizer" of the loop: it receives a mini-batch of
`ReflectionOutput`s and the current `Experience`, and returns a **new**
`Experience` plus an audit trail of the edits it made. The built-in
curators are ReAct loops in which the LLM inspects the playbook through
read tools and edits it through `add_bullet` / `modify_bullet` /
`delete_bullet` tools that are budget-guarded by a `_Ledger`.

Existing implementations in `src/search_ttl/curator/`: `react_staged`
(routes bullets to `FIND` / `VERIFY` / `DECIDE`; `DECIDE` bullets are
directives, the others are query patterns) and `react_tactic` (routes to
`TACTIC_A/B/C` with tactic-conditional query validation).

## Interface

```python
# src/search_ttl/curator/manager.py
class CuratorManager(ABC):
    @classmethod
    def from_omegaconf(cls, cfg) -> "CuratorManager": ...
    async def update(
        self,
        experience: Experience,
        reflections: list[ReflectionOutput],   # one mini-batch (actor.mini_batch_size)
        *,
        step: int,
        save_dir=None,                          # steps/step_NNNN
    ) -> tuple[Experience, CuratorStepResult]: ...
```

Contract details:

- **The experience is value-typed.** Never mutate the object you were
  given; every `experience.add_bullet(...)` etc. returns a new instance —
  return the latest one. The trainer assigns whatever you return, so a
  curator that *rejects* a step must return the original experience
  together with `rejected=True`.
- `CuratorStepResult` (`curator/schemas.py`): `step`, `ops: list[OpRecord]`
  (one per ADD / MODIFY / DELETE with `content_before` / `content_after` and
  `source_reflection_ids`), `final_summary`, `leftover_reflection_ids`
  (reflections you did not use — informational only, the trainer does
  **not** re-queue them), `react_turns_used`, `rejected`, `rejection_reason`.
  The trainer writes it to `steps/step_NNNN/curator/summary_pass_K.json`.
- Only `status == "ok"` reflections carry lessons; skip the rest.
- Route each bullet by the reflection's `structured_payload["section_hint"]`
  into a section that exists in the experience backend you pair with.
- `update()` runs `actor.epochs` times per mini-batch, so it must be
  idempotent-ish: re-reading the playbook first and preferring `modify`
  over `add` avoids piling up near-duplicates.

## Configuration keys the rest of the system reads

| Key | Read by | Notes |
|---|---|---|
| `curator.kind` | registry | |
| `curator.conservativeness.{max_add,max_modify,max_delete,max_total_edits}_per_step` | `compose.validate` | **mandatory** — validation reads it unconditionally |
| `curator.max_leftover_size` | `TTLTrainer` | cap on reflections carried to the next step (default 200) |
| `curator.char_budget`, `curator.reject_if_over_budget_ratio` | the curator | size guard on the rendered playbook, in **characters** (`Experience.char_count()` = `len(render())`): edits past `char_budget` are applied with a warning, edits past `char_budget × ratio` are refused; a whole step is rolled back if it ends over the hard cap. `0` disables |
| `curator.max_bullets_per_section` | `react_staged` | per-section bullet cap (0 disables) |
| `CURATOR_LLM_*` env vars | YAML | fall back to `AGENT_LLM_*` |

## Skeleton

The shortest path is to copy `react_staged.py` and change the prompt and
the section routing. Its pieces, in order:

```python
@dataclass
class _Conserv: max_add_per_step: int = 5; ...      # per-step op caps
@dataclass
class _OptCfg: llm_model: str; ...; char_budget: int = 48000; reject_if_over_budget_ratio: float = 1.2

@register_curator("my_curator")
class MyCurator(CuratorManager):
    @classmethod
    def from_omegaconf(cls, cfg) -> "MyCurator": ...      # build _OptCfg with cfg.get(...) defaults

    async def update(self, experience, reflections, *, step, save_dir=None):
        ledger = _Ledger(experience=experience, conserv=self.cfg.conservativeness, step=step,
                         char_budget=self.cfg.char_budget, reject_ratio=self.cfg.reject_if_over_budget_ratio)
        tools = self._build_tools(ledger)     # list_sections / read_section / read_bullet / search_bullets /
                                              # add_bullet / modify_bullet / delete_bullet / remaining_budget / finalize
        ...                                   # ReAct loop: build_chat_llm(...).bind_tools(tools), ≤ max_react_turns
        if ledger.experience.char_count() > self.cfg.char_budget * self.cfg.reject_if_over_budget_ratio:
            return experience, CuratorStepResult(step=step, ops=[], final_summary="", leftover_reflection_ids=[...],
                                                 react_turns_used=turns, rejected=True, rejection_reason="over hard cap")
        return ledger.experience, CuratorStepResult(step=step, ops=ledger.ops, final_summary=summary,
                                                    leftover_reflection_ids=uncited, react_turns_used=turns)
```

`_Ledger` (also in `react_staged.py`) is worth reusing verbatim: it
enforces the per-op caps, the char budget (soft warning / hard refusal,
never for deletes), the per-section bullet cap, and records `OpRecord`s.
`tests/search_ttl/test_curator_char_budget.py` pins its behaviour.

A non-LLM curator is also fine — e.g. one that applies every `ok`
reflection's `principle` as a bullet — as long as it returns the new
experience and a `CuratorStepResult`.

### If your curator runs its own tool-calling loop

Same three contracts as for reflectors (see
[adding-a-reflector.md](adding-a-reflector.md)): promote `mcp_call` blocks
before sanitizing, recover text-form tool calls through
`recover_text_tool_calls`, and add your module to the `paths` lists in
`tests/search_ttl/test_tool_call_recovery.py`.

## Register and configure

1. Add the import to `import_default_plugins()` in `src/search_ttl/trainer/registry.py`.
2. Create `config/ttl/curator/my_curator.yaml` — start from
   `react_staged.yaml`; keep the `CURATOR_LLM_*` fallback chains, the four
   provider knobs, `conservativeness`, `char_budget`,
   `reject_if_over_budget_ratio`, `max_leftover_size`.
3. `python scripts/ttl/run.py --curator my_curator --experience <matching backend> …`

## Tests to add or update

- `tests/search_ttl/test_registry.py`, `tests/search_ttl/test_llm_builder.py`
  (`from_omegaconf` reads the knobs; `test_reflectors_and_curators_read_llm_provider`;
  the YAML in `test_yaml_configs_expose_thinking_knobs`).
- `tests/search_ttl/test_curator_char_budget.py` if you change or reuse the
  ledger; `tests/search_ttl/test_step_schedule.py` shows how the trainer
  drives `update()` with a mock curator.

# Adding a reflector

The reflector is called once per question with that question's `G`
trajectories and their reward reports, and returns one `ReflectionOutput`
per trajectory. Its job is to distil *what to do differently* into a
structured payload the curator can turn into playbook bullets.

Existing implementations in `src/search_ttl/reflector/`:

| kind | Sections it targets | Idea |
|---|---|---|
| `querychain_staged` | `FIND` / `VERIFY` / `DECIDE` (`staged_playbook`) | contrastive: only incorrect trajectories run the LLM, contrasted against the shortest *correct* peer; lessons are routed to a search stage |
| `querychain_tactic` | `TACTIC_A/B/C` (`tactic_playbook`) | same loop, lessons routed to one of three query tactics |

Both run a ReAct loop with live access to the search tool (MCP) so the
reflector can *verify* an alternative query before recommending it.

## Interface

```python
# src/search_ttl/reflector/manager.py
class ReflectorManager(ABC):
    @classmethod
    def from_omegaconf(cls, cfg) -> "ReflectorManager": ...
    async def reflect(
        self,
        question_record: QuestionRecord,
        trajectories: list[Trajectory],      # the G rollouts of ONE question
        rewards: list[RewardReport],         # aligned with trajectories
        experience_text: str,                # current playbook render()
        *,
        save_dir=None,                       # steps/step_NNNN — write transcripts here
    ) -> list[ReflectionOutput]: ...
```

`ReflectionOutput` (`reflector/schemas.py`):

| Field | Contract |
|---|---|
| `qid`, `r_idx`, `reflector_kind` | identity; `reflector_kind` is your registry name |
| `status` | `ok` (has a lesson), `no_useful_insight`, `already_covered`, `error`. Curators skip everything but `ok` |
| `structured_payload` | `dataclasses.asdict(ReflectionPayload)`; the curator reads `principle`, `trigger`, `anti_pattern`, `verified_alternative_query`, `verification_evidence`, `applies_to_future_questions_when`, `confidence`, and **`section_hint`** |
| `text_for_curator` | the human-readable rendering the curator prompt shows |
| `used_tool_calls`, `used_react_turns`, `transcript_path` | bookkeeping (logged to `reflections.jsonl`) |

**`section_hint` must be a section name of the experience backend you
pair with**, because the curator routes bullets by it. If you introduce
new section names, pair the reflector with a matching experience backend
and curator (see [adding-an-experience.md](adding-an-experience.md)).

The trainer appends every output to `steps/step_NNNN/reflections.jsonl`
and queues `ok` ones for the curator; a mini-batch fires once
`actor.mini_batch_size` have accumulated.

## Skeleton

```python
# src/search_ttl/reflector/my_reflector.py
import dataclasses
from dataclasses import dataclass
from search_ttl.reflector.manager import ReflectorManager
from search_ttl.reflector.schemas import ReflectionOutput, ReflectionPayload
from search_ttl.trainer.registry import register_reflector
from search_ttl.utils.concurrency import bounded_gather
from search_ttl.utils.llm_builder import build_chat_llm
from search_ttl.utils.tracing import traceable


@dataclass
class _Cfg:
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_provider: str = "openai_compat"
    llm_api_flavor: str = "chat_completions"
    enable_thinking: bool = False
    preserve_thinking: bool = True
    concurrency: int = 4


@register_reflector("my_reflector")
class MyReflector(ReflectorManager):
    def __init__(self, cfg: _Cfg) -> None:
        self.cfg = cfg

    @classmethod
    def from_omegaconf(cls, cfg) -> "MyReflector":
        return cls(_Cfg(
            llm_model=cfg.llm_model, llm_base_url=cfg.llm_base_url, llm_api_key=cfg.llm_api_key,
            llm_provider=str(cfg.get("llm_provider", "openai_compat")),
            llm_api_flavor=str(cfg.get("llm_api_flavor", "chat_completions")),
            enable_thinking=bool(cfg.get("enable_thinking", False)),
            preserve_thinking=bool(cfg.get("preserve_thinking", True)),
            concurrency=int(cfg.get("concurrency", 4)),
        ))

    def _llm(self):
        c = self.cfg
        return build_chat_llm(model=c.llm_model, base_url=c.llm_base_url, api_key=c.llm_api_key,
                              llm_provider=c.llm_provider, llm_api_flavor=c.llm_api_flavor,
                              enable_thinking=c.enable_thinking, preserve_thinking=c.preserve_thinking)

    @traceable(name="Reflector.reflect", run_type="chain")
    async def reflect(self, question_record, trajectories, rewards, experience_text, *, save_dir=None):
        factories = [
            (lambda i=i: self._reflect_one(i, question_record, trajectories, rewards, experience_text))
            for i in range(len(trajectories))
        ]
        return await bounded_gather(factories, concurrency=self.cfg.concurrency)

    async def _reflect_one(self, i, q, trajs, rewards, experience_text) -> ReflectionOutput:
        if rewards[i].correctness:                      # nothing to fix
            payload = ReflectionPayload(status="no_useful_insight")
        else:
            ...  # prompt the LLM with trajs[i], rewards[i].text_for_reflector, a correct peer, …
            payload = ReflectionPayload(status="ok", section_hint="FIND", principle="…", trigger="…")
        return ReflectionOutput(
            qid=q.qid, r_idx=trajs[i].r_idx, reflector_kind="my_reflector",
            status=payload.status, structured_payload=dataclasses.asdict(payload),
            text_for_curator=f"[{q.qid} r{i}] {payload.principle}",
        )
```

Rendering helpers you will want: `search_ttl.utils.tool_render.render_tool_result`
/ `describe_visibility_for_prompt` for showing search results at a chosen
verbosity (`none` / `title` / `clipped` / `full`), and the
`traj_*` / `live_*` knob pattern in `querychain_staged.py`.

### If your reflector runs its own tool-calling loop

Copy the loop skeleton from `querychain_staged.py:_reflect_one` rather
than writing one from scratch. Three contracts are pinned by
`tests/search_ttl/test_tool_call_recovery.py::test_all_react_sites_*` and
you must add your file to their `paths` lists:

1. after every `ainvoke`, call `recover_mcp_call_tool_calls(ai)` **before**
   `sanitize_responses_message_for_round_trip(ai)` (Responses-API servers
   return tool calls as `mcp_call` blocks that the sanitizer would strip);
2. route text-form tool calls through `recover_text_tool_calls` (one
   parser for all model families);
3. accept a flattened `submit_reflection` payload (`{"payload": args}`
   wrapping when the LLM omits the `payload` key —
   `tests/search_ttl/test_submit_reflection_dispatch.py`).

## Register and configure

1. Add the import to `import_default_plugins()` in `src/search_ttl/trainer/registry.py`.
2. Create `config/ttl/reflector/my_reflector.yaml`. Keep the `REFLECTOR_LLM_*`
   → `AGENT_LLM_*` fallback chains and the four provider knobs
   (`llm_provider`, `llm_api_flavor`, `enable_thinking`, `preserve_thinking`);
   `test_yaml_configs_expose_thinking_knobs` will fail otherwise. Include
   `mcp_url` / `mcp_transport` if the loop searches live.
3. `python scripts/ttl/run.py --reflector my_reflector --experience <matching backend> …`

## Tests to add or update

- `tests/search_ttl/test_registry.py` — the kind registers.
- `tests/search_ttl/test_llm_builder.py` — a `from_omegaconf` case reading the
  four knobs (copy `test_reflector_querychain_staged_reads_*`) and add the
  class to `test_reflectors_and_curators_read_llm_provider`; add the YAML
  to `test_yaml_configs_expose_thinking_knobs`.
- `tests/search_ttl/test_tool_call_recovery.py` — add the module path to the
  three `paths` lists if it has a ReAct loop.
- A unit test for your prompt rendering that builds `Trajectory` /
  `RewardReport` objects by hand (no LLM).

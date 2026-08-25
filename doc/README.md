# FINDER developer wiki

This wiki documents `src/search_ttl/` (TTL-Search) for people who want to
change or extend it. If you only want to *run* the experiments, the
top-level [README](../README.md) is enough.

| Page | Read it when you want to… |
|---|---|
| [architecture.md](architecture.md) | understand the training loop, the data that flows between the parts, and what a run writes to disk |
| [configuration.md](configuration.md) | know how `config/ttl/` is composed, which env vars feed it, and how to override anything |
| [adding-a-reward.md](adding-a-reward.md) | score trajectories differently (new judge, new metric, no judge) |
| [adding-a-reflector.md](adding-a-reflector.md) | change how rollouts are turned into structured lessons |
| [adding-a-curator.md](adding-a-curator.md) | change how lessons are written into the playbook |
| [adding-an-experience.md](adding-an-experience.md) | change the playbook format / sections / what the agent sees |
| [adding-a-benchmark.md](adding-a-benchmark.md) | train or evaluate on a new dataset |
| [adding-a-tool-provider.md](adding-a-tool-provider.md) | give the agent a different search backend |

## The one-paragraph version

TTL-Search is a training-free loop. Each step it (1) **rolls out** a
search agent `G` times per question with the current *experience
playbook* in its system prompt, (2) **rewards** every trajectory with an
LLM judge plus rule-based efficiency stats, (3) **reflects** on each
question's `G` trajectories to produce structured lessons, and (4) lets a
**curator** LLM apply those lessons to the playbook through a small edit
API (`add_bullet` / `modify_bullet` / `delete_bullet`). The playbook is the
only thing that "learns"; model weights never change.

Every one of those four parts — plus the playbook backend itself — is a
plugin selected by a string `kind` in YAML:

```
reward      → src/search_ttl/reward/      config/ttl/reward/
reflector   → src/search_ttl/reflector/   config/ttl/reflector/
curator     → src/search_ttl/curator/     config/ttl/curator/
experience  → src/search_ttl/experience/  config/ttl/experience/
```

## Conventions every plugin follows

1. **One module per implementation**, registered with the matching
   decorator from `search_ttl.trainer.registry`:
   `@register_reward("name")`, `@register_reflector("name")`,
   `@register_curator("name")`, `@register_experience("name")`.
   Registration is import-time, so add the module to
   `import_default_plugins()` in `src/search_ttl/trainer/registry.py`.
2. **Construction goes through `from_omegaconf(cfg)`** — a classmethod
   that receives the plugin's YAML block. Read every knob with
   `cfg.get("key", default)` so partial configs (tests, `-o` overrides)
   keep working.
3. **One YAML per implementation** at `config/ttl/<group>/<name>.yaml`,
   whose top-level key is the group and whose `kind:` is the registry
   name. The file stem is what you pass on the command line
   (`--reward <stem>`), so keep stem == kind.
4. **LLM access goes through `search_ttl.utils.llm_builder.build_chat_llm`**
   so the provider / API-flavor / thinking knobs (`llm_provider`,
   `llm_api_flavor`, `enable_thinking`, `preserve_thinking`) behave the
   same everywhere. Fan out with `search_ttl.utils.concurrency.bounded_gather`
   and decorate entry points with `@traceable` from `search_ttl.utils.tracing`.
5. **Tests live in `tests/search_ttl/`.** At minimum, extend
   `test_registry.py` (the kind registers) and — for anything that builds
   an LLM — `test_llm_builder.py` (the knobs are read). The per-part pages
   list the other tests that pin contracts you must keep.

Run the suite with `uv run pytest tests/search_ttl -q --no-cov`.

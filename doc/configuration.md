# Configuration

Everything under `config/ttl/` is plain [OmegaConf](https://omegaconf.readthedocs.io/)
YAML; there is no Hydra. `search_ttl.config.compose()` merges layers in a
fixed order and `validate()` fails fast on inconsistent settings.

## Groups and composition order

```
base.yaml → benchmark/<x> → tools/<x> → reward/<x> → reflector/<x> → curator/<x> → experience/<x> → mode/<x> → -o overrides
```

Later layers win. Each group is a directory under `config/ttl/`; the CLI
flag of `scripts/ttl/run.py` / `eval_experience.py` names the file stem:

| Group | CLI flag | Top-level YAML key | Options today |
|---|---|---|---|
| `benchmark` | `--benchmark` | `data:` (+ `agent.user_prompt_template`) | `bcp`, `browsecomp` |
| `tools` | `--retriever` (legacy name) | `agent:` (+ `retriever:` metadata) | `qwen3-8b`, `mcp`, `serper`, `ddg` |
| `reward` | `--reward` | `reward:` | `search_agent_lasj`, `qa_em_f1` |
| `reflector` | `--reflector` | `reflector:` | `querychain_staged`, `querychain_tactic` |
| `curator` | `--curator` | `curator:` | `react_staged`, `react_tactic` |
| `experience` | `--experience` | `experience:` | `staged_playbook`, `tactic_playbook` |
| `mode` | `--mode` | any keys (overrides) | `smoke`, `eval_only` |

`list_options(group)` (used for `--help`) is just the YAML stems in the
directory, so **adding a file is all it takes to add an option**. The
defaults of `compose()` and of `run.py` are `search_agent_lasj` /
`querychain_staged` / `react_staged` / `staged_playbook`
(`tests/search_ttl/test_compose.py` pins them).

## `base.yaml` sections

| Section | What it controls |
|---|---|
| `exp_name` | Run name; `trainer.save_path` defaults to `runs/ttl/${exp_name}` |
| `data` | `benchmark`, `decrypted_jsonl`, `splits.{train,val,test}`, `train_batch_size` (questions per step) |
| `agent` | The SearchAgent: LLM endpoint + provider/flavor/thinking knobs, `max_search_limit`, `max_turn`, `timeout_s`, `tool_call_timeout_s`, `reasoning_aware_retrieval`. Tool-backend keys (`tool_provider`, `mcp_url`, …) are layered in by the `tools` group |
| `rollout` | `n` (= G rollouts per question), `concurrency`, `seed` |
| `actor` | `mini_batch_size` (reflections per curator call), `micro_batch_size` (trajectories per reward+reflector chunk), `epochs` (curator passes per mini-batch) |
| `eval` | `eval_every_n_steps` (0 = never), `eval_split`, `final_eval_split`, `save_best`, `concurrency` (null → `rollout.concurrency`) |
| `trainer` | `save_path`, `save_freq`, `keep_last_n_checkpoints`, `resume_mode`, `resume_from_path`, `val_before_train`, `total_epochs`, `total_training_steps` |
| `tracing` | Weave / W&B project names and tags |

Plugin blocks (`reward:`, `reflector:`, `curator:`, `experience:`) come
entirely from their group file — `base.yaml` does not define them.

## Environment variables

YAML values reference the environment with `${oc.env:VAR,default}`. The
convention is a **fallback chain** so one variable moves everything and
component-specific variables let you mix endpoints:

```
agent      AGENT_LLM_MODEL / AGENT_LLM_BASE_URL / AGENT_LLM_API_KEY / AGENT_LLM_PROVIDER
reflector  REFLECTOR_LLM_*   → falls back to AGENT_LLM_*
curator    CURATOR_LLM_*     → falls back to AGENT_LLM_*
judge      EVAL_JUDGE_MODEL / _BASE_URL / _API_KEY / _PROVIDER → falls back to AGENT_LLM_*
search     AGENT_MCP_URL, AGENT_MCP_TRANSPORT (mcp) · SERPER_API_KEY (serper)
```

`scripts/ttl/run.py` loads `.env` from the repo root before composing, so
`.env` is the normal place to set these (see `.env.example`).

Provider knobs every LLM-using block exposes (and
`tests/search_ttl/test_llm_builder.py::test_yaml_configs_expose_thinking_knobs`
checks for reflector / curator YAMLs):

| Key | Values | Meaning |
|---|---|---|
| `llm_provider` | `openai_compat`, `openrouter` | `ChatOpenAI` against any OpenAI-compatible URL vs `langchain-openrouter` |
| `llm_api_flavor` | `chat_completions`, `responses` | Chat Completions (with a reasoning-preserving wrapper) vs the OpenAI Responses API (needs vLLM ≥ 0.19 or SGLang) |
| `enable_thinking` | bool | Ask the model to think (`chat_template_kwargs`) |
| `preserve_thinking` | bool | Re-inject the previous turn's reasoning on multi-turn loops (only with thinking on + `chat_completions`) |

## Overrides

`-o key=value [key=value …]` is an OmegaConf dotlist applied last, e.g.

```bash
python scripts/ttl/run.py ... -o rollout.n=2 curator.char_budget=16000 reflector.enable_thinking=true
```

`trainer.save_path` is always injected from `--save-path`.

## `validate()` rules

Run automatically by `run.py` (the rejudge scripts skip it):

- `curator.conservativeness.max_total_edits_per_step ≥ max(max_add, max_modify, max_delete)` — so **every curator YAML must carry a `conservativeness` block**;
- `actor.micro_batch_size % rollout.n == 0`, `actor.mini_batch_size % actor.micro_batch_size == 0`, `mini_batch_size ≤ train_batch_size × rollout.n`, `rollout.concurrency > 0`;
- `data.splits.*` and `data.decrypted_jsonl` exist on disk;
- `agent.llm_base_url` is set; `agent.mcp_url` when `tool_provider=mcp`; `agent.serper.api_key` when `tool_provider=serper`.

## Modes

`mode/smoke.yaml` shrinks everything for a five-minute end-to-end run
(2 questions × 2 rollouts, 2 steps, tiny ReAct budgets);
`mode/eval_only.yaml` loads a checkpoint (`trainer.resume_mode=resume_path`)
and runs zero training steps. Add your own mode file for any recurring
override set.

## Adding a config option

1. Copy the closest existing file in `config/ttl/<group>/` to `<name>.yaml`.
2. Keep the top-level key (`reward:` etc.) and set `kind:` to the registry name.
3. Keep the env-var fallback chains for endpoints and the four provider knobs.
4. Run `python scripts/ttl/run.py --help` — the new stem is listed — and
   `uv run pytest tests/search_ttl/test_compose.py tests/search_ttl/test_llm_builder.py -q --no-cov`.

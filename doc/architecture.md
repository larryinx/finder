# Architecture

## Package map

```
src/search_ttl/
  agent/        SearchAgent (LangChain ReAct loop), prompts, tool providers, MCP helpers
  rollout/      RolloutManager (G rollouts per question, bounded concurrency) + schemas
  reward/       RewardManager ABC, LLM judge, reward implementations
  reflector/    ReflectorManager ABC + implementations
  curator/      CuratorManager ABC + implementations
  experience/   Experience ABC (the playbook) + backends
  evaluator/    Evaluator (n=1 rollouts + scoring → metrics.json) and per-benchmark loaders
  trainer/      TTLTrainer (the loop), plugin registry, checkpointing, sampler
  config/       compose() — layered YAML composition — and validate()
  utils/        llm_builder, concurrency, tracing, persistence, tool_render, tool_call_recovery, logging
```

Entry points: `scripts/ttl/run.py` (train), `scripts/ttl/eval_experience.py`
(evaluate a frozen playbook), `scripts/ttl/rejudge_*.py` (re-score existing
run directories with a different reward).

## One training step

`TTLTrainer.step()` (`src/search_ttl/trainer/ttl_trainer.py`):

```
Phase 1  rollout      sample data.train_batch_size questions; SearchAgent runs
                      rollout.n (= G) times each, all concurrently (rollout.concurrency)
                      → flat list of Trajectory, grouped by question

Phase 2  reward       for each micro-batch (actor.micro_batch_size trajectories):
                        rewards = reward.score(chunk)            # one RewardReport per traj

Phase 3  reflect        for each question in the chunk:
                          reflections = reflector.reflect(question, its G trajs,
                                                          its G rewards, playbook text)

Phase 4  curate       whenever ≥ actor.mini_batch_size reflections have accumulated:
                        playbook, result = curator.update(playbook, mini_batch)  × actor.epochs
```

Batch-sizing constraints (checked by `compose.validate`):
`micro_batch_size % rollout.n == 0` (a chunk never splits a question's
trajectories) and `mini_batch_size % micro_batch_size == 0`. Reflections
that never fill a mini-batch by the end of a step are carried to the next
step (`curator.max_leftover_size`); reflections the curator considered
but did not cite are **not** re-queued.

After each step the trainer writes the playbook, maybe saves a checkpoint
(`trainer.save_freq`), and maybe evaluates on `eval.eval_split`
(`eval.eval_every_n_steps`). `trainer.total_training_steps` caps the loop
(derived from `total_epochs × len(train) // train_batch_size` when null).

## Data contracts

All schemas are plain dataclasses (JSON-serialisable with `asdict`).

| Stage | Type (module) | Key fields |
|---|---|---|
| input | `QuestionRecord` (`rollout/schemas.py`) | `qid`, `question`, `target`, `golden_answers` |
| rollout | `Trajectory` (`rollout/schemas.py`) | `steps` (reasoning / tool_call / tool_result / output_text), `queries: list[QueryRecord]`, `extracted_answer`, `raw_response`, `status` ∈ {completed, max_turn, max_search, error}, `retrieved_docids`, `tool_call_counts`, `metrics` (`turns`, `search_count`), `experience_snapshot` |
| reward | `RewardReport` (`reward/schemas.py`) | `correctness`, `judge: JudgeResult`, `efficiency: EfficiencyStats`, `failure_mode`, `text_for_reflector`, `per_query_analysis`, `extra` |
| reflect | `ReflectionOutput` (`reflector/schemas.py`) | `status` ∈ {ok, no_useful_insight, already_covered, error}, `structured_payload` (a `ReflectionPayload` as dict — `principle`, `trigger`, `anti_pattern`, `verified_alternative_query`, `section_hint`, …), `text_for_curator` |
| curate | `CuratorStepResult` (`curator/schemas.py`) | `ops: list[OpRecord]` (ADD / MODIFY / DELETE with before/after content and cited reflection ids), `rejected`, `final_summary`, `leftover_reflection_ids` |
| state | `Experience` (`experience/base.py`) | value-typed playbook: `render()`, `to_text()`, `list_sections()`, `read_/search_bullets()`, `char_count()`, `add_/modify_/delete_bullet()` (each returns a **new** instance) |

The reflector's `section_hint` and the curator's routing must agree on
the experience backend's section names (e.g. `FIND` / `VERIFY` / `DECIDE`
for `staged_playbook`, `TACTIC_A/B/C` for `tactic_playbook`). That is the
main coupling between plugins — see the per-part pages.

## The agent

`SearchAgent` (`agent/search_agent.py`) is a LangChain tool-calling loop:
system prompt = a one-line body plus an `<experience>…</experience>` block
holding `Experience.render()` (omitted when the playbook is empty, so the
baseline prompt is reproduced byte-for-byte); first user message = the
benchmark's question template (`agent.user_prompt_template`:
`bcp` / `browsecomp` / `boxed_answer`, see `agent/prompts.py`). The agent
stops at `agent.max_turn` turns or `agent.max_search_limit` searches and
extracts the answer (`Exact Answer:` / `\boxed{}`) into
`Trajectory.extracted_answer`.

Search tools come from a `ToolBuilder` (`agent/tool_builders.py`) chosen by
`agent.tool_provider`: `mcp` (BrowseComp-Plus server), `serper`, `ddg`.

## Evaluation

`Evaluator.run_split()` (`evaluator/runner.py`) rolls the agent out once
per question (`n=1`, `eval.concurrency`), scores with the same reward, and
writes `eval/<tag>/{rollouts.jsonl, rewards.jsonl, metrics.json}`
(`accuracy`, `avg_n_search`, `avg_n_turn`, plus `f1/cem/em` when the reward
puts them in `RewardReport.extra`). With `eval.save_best` it maintains
`experience/best.md` + `best.meta.json`. `--resume` keeps clean rollouts and
re-runs only `status == "error"` ones. Benchmarks are dispatched through
`evaluator/loaders.py:get_loader`.

## Checkpoints and resume

`CheckpointManager` saves `checkpoints/global_step_N/{experience.md,
sampler_state.json, trainer_state.json}` every `trainer.save_freq` steps
(and on the last step). `trainer.resume_mode` ∈ {`auto`, `disable`,
`resume_path`}; on resume the playbook is rebuilt with
`type(experience).from_text(...)`, which is why every backend must
round-trip its own `to_text()`.

## Run directory

```
runs/ttl/<exp>/
  experience/    step_0000_initial.md, step_NNNN_after_curator.md, best.md, best.meta.json, final.md
  steps/step_NNNN/
                 rollouts.jsonl        one BCP-shaped record per trajectory
                 rewards.jsonl         one RewardReport per trajectory
                 reflections.jsonl     one ReflectionOutput per trajectory
                 reflector/…           per-reflection transcripts (if the reflector saves them)
                 curator/summary_pass_K.json
  checkpoints/global_step_N/
  eval/<tag>/    rollouts.jsonl, rewards.jsonl, metrics.json
```

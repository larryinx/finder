# Adding a benchmark

A benchmark is (1) data on disk, (2) a loader that turns a split into
`QuestionRecord` rows, (3) a `benchmark/<name>.yaml` that points at the
data and picks the agent's question template, and (4) a choice of reward
and search tool that fit the data.

Existing: `bcp` (BrowseComp-Plus: TSV query files + a decrypted JSONL,
LLM judge, MCP retriever) and `browsecomp` (open-web, Serper / DuckDuckGo,
eval-only). A `hotpotqa` loader also exists for multi-gold QA with the
`qa_em_f1` reward.

## 1. Data preparation

Put a script under `scripts/data_prep/<name>/prepare.sh` (or a Python
builder it calls) that writes into `data/<name>/`, and register it in the
`ENTRIES` table of `scripts/data_prep/prepare.sh` as
`"<name>|scripts/data_prep/<name>/prepare.sh|data/<name>"`. The dispatcher
`cd`s into the data dir before running your script. Make it idempotent.

## 2. Loader

Add a function to `src/search_ttl/evaluator/loaders.py` that returns plain
dicts and register it in `get_loader`:

```python
def load_my_split(eval_jsonl: str | Path) -> list[dict[str, Any]]:
    rows = []
    for d in read_jsonl(eval_jsonl):
        golds = list(d.get("golden_answers") or [])
        rows.append({
            "query_id": str(d["query_id"]),
            "question": str(d["question"]),
            "target": str(d.get("target") or (golds[0] if golds else "")),
            "golden_answers": golds,      # optional; needed by qa_em_f1
            "context": "",
        })
    return rows

def get_loader(benchmark):
    table = {..., "my_bench": load_my_split}
```

`QuestionRecord.from_bcp_dict` consumes exactly these keys (`target`
falls back to the first gold answer and vice versa). Both the trainer
(`scripts/ttl/run.py`, train split) and the Evaluator go through
`load_split(benchmark, path, decrypted_jsonl=…)`, which dispatches via
`get_loader` — registering the loader is all it takes for training and
evaluation alike (only `browsecomp_plus` receives the extra
`decrypted_jsonl` argument).

## 3. Benchmark YAML

```yaml
# config/ttl/benchmark/my_bench.yaml
data:
  benchmark: my_bench                 # the get_loader key
  decrypted_jsonl: data/my_bench/eval.jsonl   # must exist (validate checks it); BCP-specific, point it at any real file
  splits:
    train: data/my_bench/train.jsonl
    val:   data/my_bench/val.jsonl
    test:  data/my_bench/test.jsonl

agent:
  user_prompt_template: boxed_answer  # bcp | browsecomp | boxed_answer (agent/prompts.py)
```

If the benchmark needs a different answer format, add a template to
`_USER_PROMPT_TEMPLATES` in `src/search_ttl/agent/prompts.py` and make sure
`agent/extract.py` can pull the final answer out of it.

## 4. Reward and tools

- Single canonical answer graded by an LLM → `--reward search_agent_lasj`
  (the BCP grader template; `Trajectory.target` is the reference).
- A list of acceptable answers, no judge → `--reward qa_em_f1`
  (max-over-golds EM / cover-EM / F1, written into `metrics.json`).
- A vetted corpus served over MCP → `--retriever mcp` / `qwen3-8b`; open web →
  `--retriever serper` or `ddg` (see [adding-a-tool-provider.md](adding-a-tool-provider.md)).

## 5. Run

```bash
bash scripts/data_prep/prepare.sh my_bench
python scripts/ttl/run.py --benchmark my_bench --retriever ddg --reward qa_em_f1 --save-path runs/ttl/my_bench
```

## Tests to add or update

- A loader test with a tiny JSONL fixture in `tmp_path` (shape of the
  returned dicts, `QuestionRecord.from_bcp_dict` round trip) — see
  `tests/search_ttl/test_loaders.py`.
- `tests/search_ttl/test_compose.py::test_lists_kinds` style assertion that
  `"my_bench" in list_options("benchmark")`.

# Adding a tool provider

A tool provider is what the agent's `search` tool talks to. The provider
hands the agent a list of LangChain tools and a post-processor that turns
a raw tool result into what the trajectory stores.

Existing providers in `src/search_ttl/agent/tool_builders.py`:
`mcp` (the BrowseComp-Plus MCP server — returns hits with `docid`s),
`serper` (Google Serper), `ddg` (DuckDuckGo). The last two return flat
text and no docids.

## Interface

```python
# src/search_ttl/agent/tool_builders.py
class ToolBuilder(ABC):
    search_tool_name: str = "search"        # the agent resolves the search tool by this name

    async def aload(self) -> list: ...      # LangChain tools to bind to the LLM
    async def reconnect(self) -> list: ...  # default: aload() again; MCP rebuilds its session
    def postprocess(self, raw_result) -> ToolPostprocess: ...
        # ToolPostprocess(tool_str=<what the LLM sees>, docids=[...], persisted_output=<what rollouts.jsonl stores>)
```

`docids` feed the efficiency stats (`retrieved_docids`, `novel_doc_ratio`)
— return `[]` when your backend has no stable document ids. `tool_str` is
what goes into the `ToolMessage`; keep it bounded in size.

## Steps

1. **Implement the builder** next to the existing ones, e.g.
   `class MySearchToolBuilder(ToolBuilder)` whose `aload()` returns one
   `@tool async def search(query: str) -> str` (name it `self.search_tool_name`).
2. **Wire the factory**: add an `if provider == "my_search":` branch to
   `build_tool_builder()` that reads its knobs from the agent config.
3. **Expose the knobs on `SearchAgentConfig`** (`agent/search_agent.py`):
   add a `my_search: dict = field(default_factory=dict)` sub-dict and read
   it in `from_omegaconf` with the existing `_subdict("my_search")`
   helper (this is how `serper` and `ddg` are done).
4. **Add `config/ttl/tools/my_search.yaml`**:

```yaml
agent:
  tool_provider: my_search
  my_search:
    api_key: ${oc.env:MY_SEARCH_API_KEY,""}
    num_results: 10

retriever:            # metadata only (some prompts mention the retriever type)
  name: my_search
  expected_top_k: 10
```

5. **Validation** (optional): if a key is required, add a branch in
   `compose.validate()` next to the `serper` one so a missing key fails at
   startup rather than mid-rollout. Document the env var in `.env.example`.
6. Select it with `--retriever my_search` (the flag keeps its legacy name;
   it resolves under `config/ttl/tools/`).

Notes:

- The built-in reflectors make their *own* live search calls over MCP
  (`reflector.mcp_url`); a non-MCP provider only affects the agent unless
  you also teach the reflector about it.
- `SearchAgent` wraps each call with `agent.tool_call_timeout_s`; on
  timeout the MCP builder reconnects once and retries, other builders just
  propagate the error — implement `reconnect()` if your backend has
  sessions.

## Tests to add or update

- A unit test for `postprocess()` with a canned raw result (the
  `tool_str` / `docids` split).
- `tests/search_ttl/test_llm_builder.py::test_search_agent_config_*` style
  case that `SearchAgentConfig.from_omegaconf` picks up your sub-dict.

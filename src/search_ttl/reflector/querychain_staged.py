"""QuerychainStagedReflector — staged variant of ``querychain_tactic``.

Pairs with the ``staged_playbook`` experience backend (FIND / VERIFY /
DECIDE) and the ``react_staged`` curator. Same contrastive ReAct loop
as ``querychain_tactic`` (skip correct trajectories with no LLM call;
for an incorrect one, state the judged result and contrast it against
the single shortest correct trajectory; all-incorrect questions yield
``no_useful_insight``).

Behavioural changes vs ``querychain_tactic`` — driven by the analyses in
``docs/stage2/analysis/gptoss/*``:

1. **The current playbook is NOT shown to the reflector.** It used to be
   dumped into the prompt and the model was told to return
   ``already_covered`` if its idea was present. That made the model
   second-guess and produce weaker insights. We now let it always
   produce its single best, transferable insight; deduplication is the
   curator's job (it sees the live playbook).
2. **Diagnose the failure STAGE, then teach that stage.** Instead of
   "pick a query tactic", the model first classifies WHERE the failed
   trajectory went wrong — FIND (never located the entity), VERIFY
   (had/saw a candidate but mis-verified or looped), or DECIDE (had the
   evidence but committed the wrong answer / drifted to a famous
   look-alike / never committed). ``section_hint`` ∈ {FIND, VERIFY,
   DECIDE}.
3. **Dense-retriever-correct query style.** The retriever is dense
   (qwen3-8b): quoting is inert and shard/boolean chains hurt. FIND /
   VERIFY ``verified_alternative_query`` MUST be ONE coherent
   natural-language sentence / plain words — never quoted shards,
   never AND/OR.
4. **DECIDE produces a directive, not a query.** For DECIDE the model
   leaves ``verified_alternative_query`` empty and fills ``principle``
   (the rule), ``anti_pattern`` (the do-not), ``applies_to_future_
   questions_when`` (when to apply) and ``what_went_wrong_with_my_traj``
   / ``verification_evidence`` (why). The ``react_staged`` curator
   assembles these into a Rule/Why/Apply-when/Do-NOT bullet.
5. **Transferable, no answer leak.** The insight must describe the
   reusable MOVE, not bake the specific answer entity into the query;
   the curator enforces this too.

``ReflectionPayload`` shape is unchanged (no schema edit) — the stage is
carried in ``section_hint`` and the directive vs query distinction is
implied by ``section_hint == 'DECIDE'``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search_ttl.reflector.manager import ReflectorManager
from search_ttl.reflector.schemas import ReflectionOutput, ReflectionPayload
from search_ttl.reward.schemas import RewardReport
from search_ttl.rollout.schemas import QuestionRecord, Trajectory
from search_ttl.trainer.registry import register_reflector
from search_ttl.utils.concurrency import bounded_gather
from search_ttl.utils.persistence import append_jsonl
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


@dataclass
class _ReflectorCfg:
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    mcp_url: str
    mcp_transport: str = "streamable_http"
    tool_call_timeout_s: float = 90.0
    traj_tool_result_visibility: str = "title"
    traj_tool_result_max_chars_total: int = 1500
    traj_tool_result_max_chars_per_hit: int = 500
    traj_reasoning_chars_per_step: int = 600
    live_tool_result_visibility: str = "clipped"
    live_tool_result_max_chars_total: int = 1500
    live_tool_result_max_chars_per_hit: int = 500
    max_react_turns: int = 10
    max_search_calls: int = 6
    concurrency: int = 8
    allow_no_useful_insight: bool = True
    llm_provider: str = "openai_compat"  # openai_compat | openrouter
    llm_api_flavor: str = "chat_completions"
    enable_thinking: bool = False
    preserve_thinking: bool = True
    temperature: float = 0.4
    top_p: float = 0.95
    max_response_tokens: int = 8192
    llm_timeout_s: float = 600.0


@register_reflector("querychain_staged")
class QuerychainStagedReflector(ReflectorManager):
    def __init__(self, cfg: _ReflectorCfg) -> None:
        self.cfg = cfg
        self._mcp_tools: list = []
        self._search_tool = None

    @classmethod
    def from_omegaconf(cls, cfg) -> "QuerychainStagedReflector":
        rcfg = _ReflectorCfg(
            llm_model=str(cfg.get("llm_model", ""))
            or os.environ.get("REFLECTOR_LLM_MODEL", "")
            or os.environ.get("AGENT_LLM_MODEL", ""),
            llm_base_url=str(cfg.get("llm_base_url", ""))
            or os.environ.get("REFLECTOR_LLM_BASE_URL", "")
            or os.environ.get("AGENT_LLM_BASE_URL", ""),
            llm_api_key=str(cfg.get("llm_api_key", ""))
            or os.environ.get("REFLECTOR_LLM_API_KEY", "")
            or os.environ.get("AGENT_LLM_API_KEY", ""),
            mcp_url=str(cfg.get("mcp_url", "")) or os.environ.get("AGENT_MCP_URL", ""),
            mcp_transport=str(cfg.get("mcp_transport", "streamable_http")),
            tool_call_timeout_s=float(cfg.get("tool_call_timeout_s", 90.0)),
            traj_tool_result_visibility=str(
                cfg.get("traj_tool_result_visibility", "title")
            ),
            traj_tool_result_max_chars_total=int(
                cfg.get("traj_tool_result_max_chars_total", 1500)
            ),
            traj_tool_result_max_chars_per_hit=int(
                cfg.get("traj_tool_result_max_chars_per_hit", 500)
            ),
            traj_reasoning_chars_per_step=int(
                cfg.get("traj_reasoning_chars_per_step", 600)
            ),
            live_tool_result_visibility=str(
                cfg.get("live_tool_result_visibility", "clipped")
            ),
            live_tool_result_max_chars_total=int(
                cfg.get("live_tool_result_max_chars_total", 1500)
            ),
            live_tool_result_max_chars_per_hit=int(
                cfg.get("live_tool_result_max_chars_per_hit", 500)
            ),
            max_react_turns=int(cfg.get("max_react_turns", 10)),
            max_search_calls=int(cfg.get("max_search_calls", 6)),
            concurrency=int(cfg.get("concurrency", 8)),
            allow_no_useful_insight=bool(cfg.get("allow_no_useful_insight", True)),
            llm_provider=str(cfg.get("llm_provider", "openai_compat")),
            llm_api_flavor=str(cfg.get("llm_api_flavor", "chat_completions")),
            enable_thinking=bool(cfg.get("enable_thinking", False)),
            preserve_thinking=bool(cfg.get("preserve_thinking", True)),
            temperature=float(cfg.get("temperature", 0.4)),
            top_p=float(cfg.get("top_p", 0.95)),
            max_response_tokens=int(cfg.get("max_response_tokens", 8192)),
            llm_timeout_s=float(cfg.get("llm_timeout_s", 600.0)),
        )
        return cls(rcfg)

    # ── public ────────────────────────────────────────────────────────────
    @traceable(name="Reflector.reflect", run_type="chain")
    async def reflect(
        self,
        question_record: QuestionRecord,
        trajectories: list[Trajectory],
        rewards: list[RewardReport],
        experience_text: str,  # intentionally unused — see module docstring
        *,
        save_dir: Path | None = None,
    ) -> list[ReflectionOutput]:
        qid = question_record.qid
        n = len(trajectories)

        correct_idxs = [i for i in range(n) if rewards[i].correctness]

        shortest_correct: int | None = None
        if correct_idxs:

            def _length(i: int) -> tuple[int, int]:
                eff = rewards[i].efficiency
                n_search = (
                    eff.n_search
                    if eff is not None
                    else len(trajectories[i].queries)
                )
                return (n_search, len(trajectories[i].steps))

            shortest_correct = min(correct_idxs, key=_length)

        will_run_llm = shortest_correct is not None and any(
            not rewards[i].correctness for i in range(n)
        )
        if will_run_llm:
            await self._ensure_tools()

        factories = []
        for r_idx in range(n):
            if rewards[r_idx].correctness:
                factories.append(
                    lambda r=r_idx: self._skip(
                        r, qid, "trajectory correct — reflection skipped"
                    )
                )
            elif shortest_correct is None:
                factories.append(
                    lambda r=r_idx: self._skip(
                        r,
                        qid,
                        "no correct trajectory for this question — "
                        "nothing to contrast against",
                    )
                )
            else:
                factories.append(
                    lambda r=r_idx, c=shortest_correct: self._reflect_one(
                        r,
                        correct_idx=c,
                        question_record=question_record,
                        trajectories=trajectories,
                        rewards=rewards,
                        save_dir=save_dir,
                    )
                )
        return await bounded_gather(factories, concurrency=self.cfg.concurrency)

    async def _skip(
        self, r_idx: int, qid: str, reason: str
    ) -> ReflectionOutput:
        return _no_insight_output(qid=qid, r_idx=r_idx, reason=reason)

    # ── tool init ─────────────────────────────────────────────────────────
    async def _ensure_tools(self) -> None:
        if self._search_tool is not None:
            return
        await self._connect_mcp()

    async def _connect_mcp(self) -> None:
        from search_ttl.agent.tools import build_mcp_loader

        loader = build_mcp_loader(
            mcp_url=self.cfg.mcp_url, transport=self.cfg.mcp_transport
        )
        self._mcp_tools = await loader.aload()
        self._search_tool = None
        for t in self._mcp_tools:
            if t.name == "search":
                self._search_tool = t
                break
        if self._search_tool is None:
            raise RuntimeError(
                f"reflector requires a 'search' tool but none was advertised by MCP at "
                f"{self.cfg.mcp_url!r}; got {[t.name for t in self._mcp_tools]}"
            )

    # ── per-reflector loop ───────────────────────────────────────────────
    @traceable(name="Reflector.reflect_one", run_type="chain")
    async def _reflect_one(
        self,
        r_idx: int,
        *,
        correct_idx: int,
        question_record: QuestionRecord,
        trajectories: list[Trajectory],
        rewards: list[RewardReport],
        save_dir: Path | None,
    ) -> ReflectionOutput:
        from langchain_core.messages import (
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
        from langchain_core.tools import tool as tool_dec

        from search_ttl.utils.llm_builder import build_chat_llm

        llm = build_chat_llm(
            model=self.cfg.llm_model,
            base_url=self.cfg.llm_base_url,
            api_key=self.cfg.llm_api_key,
            llm_provider=self.cfg.llm_provider,
            llm_api_flavor=self.cfg.llm_api_flavor,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_response_tokens,
            enable_thinking=self.cfg.enable_thinking,
            preserve_thinking=self.cfg.preserve_thinking,
            timeout_s=self.cfg.llm_timeout_s,
        )

        submitted_payload: dict[str, Any] = {}

        @tool_dec
        def submit_reflection(payload: dict[str, Any]) -> str:
            """Submit your final ReflectionPayload. Calling this ends reflection."""
            submitted_payload.clear()
            submitted_payload.update(payload)
            return "submitted"

        bound_llm = llm.bind_tools([self._search_tool, submit_reflection])

        sys_prompt = (
            "You are a Reflector for a search agent's experience playbook. "
            "Diagnose which STAGE the failed trajectory broke at, then teach "
            "ONE transferable lesson for that stage. Use the `search` tool to "
            "verify a FIND/VERIFY query before finalizing (DECIDE needs no search)."
        )
        user_msg = _render_user_message(
            r_idx=r_idx,
            correct_idx=correct_idx,
            question=question_record.question,
            target=question_record.target,
            trajectories=trajectories,
            rewards=rewards,
            cfg=self.cfg,
        )
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)]
        transcript: list[dict] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        used_search = 0
        used_turns = 0
        status = "no_useful_insight"
        for turn_idx in range(self.cfg.max_react_turns):
            used_turns += 1
            try:
                ai = await bound_llm.ainvoke(messages)
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "querychain_staged reflector LLM failed qid=%s r_idx=%s: %s",
                    question_record.qid, r_idx, exc,
                )
                status = "error"
                break
            from search_ttl.utils.tool_call_recovery import (
                recover_mcp_call_tool_calls,
                sanitize_responses_message_for_round_trip,
            )
            mcp_recovered = recover_mcp_call_tool_calls(ai)
            if mcp_recovered:
                LOG.info(
                    "querychain_staged qid=%s r_idx=%s: promoted %d mcp_call block(s)",
                    question_record.qid, r_idx, len(mcp_recovered),
                )
            n_stripped = sanitize_responses_message_for_round_trip(ai)
            if n_stripped:
                LOG.info(
                    "querychain_staged qid=%s r_idx=%s: stripped %d "
                    "vLLM-rejected content block(s) before round-trip",
                    question_record.qid, r_idx, n_stripped,
                )
            messages.append(ai)
            transcript.append(
                {"role": "assistant",
                 "tool_calls": _serialise_tool_calls(ai),
                 "content": _safe_text(ai)}
            )

            tcs = getattr(ai, "tool_calls", None) or []

            if not tcs and isinstance(ai.content, str):
                from search_ttl.utils.tool_call_recovery import recover_text_tool_calls

                recovered = recover_text_tool_calls(ai.content)
                if recovered:
                    LOG.info(
                        "querychain_staged qid=%s r_idx=%s turn=%s: "
                        "recovered %d tool_call(s) from text",
                        question_record.qid, r_idx, turn_idx, len(recovered),
                    )
                    ai.tool_calls = recovered
                    tcs = recovered

            if not tcs:
                if turn_idx + 1 >= self.cfg.max_react_turns:
                    break
                messages.append(
                    HumanMessage(
                        content=(
                            "You did not call a tool. Call `submit_reflection` "
                            "with your structured payload, even if status='no_useful_insight'."
                        )
                    )
                )
                transcript.append(
                    {"role": "user", "content": "(nudge: call submit_reflection)"}
                )
                continue

            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                tcid = (
                    tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "") or ""
                )

                if name == "submit_reflection":
                    if isinstance(args, dict) and "payload" not in args:
                        normalised_args = {"payload": args}
                    else:
                        normalised_args = args
                    try:
                        submit_reflection.invoke(normalised_args)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(
                            "submit_reflection failed qid=%s r_idx=%s: %s; "
                            "treating as no_useful_insight",
                            question_record.qid, r_idx, exc,
                        )
                        submitted_payload.clear()
                        submitted_payload.update(
                            {"status": "no_useful_insight",
                             "notes": f"submit_reflection arg-validation: {exc}"}
                        )
                    status = str(submitted_payload.get("status", "ok"))
                    transcript.append(
                        {"tool": "submit_reflection", "args": args, "result": "submitted"}
                    )
                    messages.append(ToolMessage(content="submitted", tool_call_id=tcid))
                    break

                if name == "search":
                    if isinstance(args, dict) and "query" not in args and any(
                        k in args
                        for k in (
                            "status", "principle", "trigger", "anti_pattern",
                            "section_hint", "verified_alternative_query",
                        )
                    ):
                        messages.append(
                            ToolMessage(
                                content=(
                                    "ERROR: you called `search` but passed "
                                    "ReflectionPayload fields. Use "
                                    "search(query='...') for retrieval, and "
                                    "submit_reflection(payload={...}) when done."
                                ),
                                tool_call_id=tcid,
                            )
                        )
                        transcript.append(
                            {"tool": "search", "args": args,
                             "result": "(redirected: payload kwargs misrouted)"}
                        )
                        continue
                    if used_search >= self.cfg.max_search_calls:
                        messages.append(
                            ToolMessage(
                                content="(search budget exhausted; finalize now)",
                                tool_call_id=tcid,
                            )
                        )
                        transcript.append(
                            {"tool": "search", "args": args, "result": "(budget exhausted)"}
                        )
                        continue
                    try:
                        from search_ttl.agent.mcp_call import (
                            call_with_timeout_and_reconnect,
                        )

                        result = await call_with_timeout_and_reconnect(
                            invoke_fn=lambda: self._search_tool.ainvoke(args),
                            reconnect_fn=self._connect_mcp,
                            timeout_s=self.cfg.tool_call_timeout_s,
                            label=f"querychain_staged.search[qid={question_record.qid} r={r_idx}]",
                        )
                    except Exception as exc:  # noqa: BLE001
                        result = f"[tool_error] {type(exc).__name__}: {exc}"
                    used_search += 1
                    from hiexp.graph import _unwrap_mcp_text  # type: ignore[import-not-found]
                    from search_ttl.utils.tool_render import render_tool_result

                    if isinstance(result, str) and result.startswith("[tool_error]"):
                        rendered = result
                    else:
                        unwrapped = _unwrap_mcp_text(result)
                        rendered = render_tool_result(
                            unwrapped,
                            visibility=self.cfg.live_tool_result_visibility,  # type: ignore[arg-type]
                            max_chars_total=self.cfg.live_tool_result_max_chars_total,
                            max_chars_per_hit=self.cfg.live_tool_result_max_chars_per_hit,
                        )
                    messages.append(ToolMessage(content=rendered, tool_call_id=tcid))
                    transcript.append(
                        {"tool": "search", "args": args,
                         "result_preview": rendered[:400]}
                    )
                else:
                    messages.append(
                        ToolMessage(content=f"unknown tool {name!r}", tool_call_id=tcid)
                    )

            if submitted_payload:
                break

        if not submitted_payload:
            payload = ReflectionPayload(status="no_useful_insight")
            structured = dataclasses.asdict(payload)
        else:
            structured = _coerce_payload_dict(submitted_payload)
            status = str(structured.get("status", "no_useful_insight"))

        text = _render_text_for_curator(
            qid=question_record.qid, r_idx=r_idx, payload=structured
        )
        transcript_path: Path | None = None
        if save_dir is not None:
            transcript_path = (
                save_dir / f"reflector_transcripts/q{question_record.qid}__r{r_idx}.jsonl"
            )
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            await append_jsonl(
                transcript_path,
                {"qid": question_record.qid, "r_idx": r_idx, "transcript": transcript},
            )

        return ReflectionOutput(
            qid=question_record.qid,
            r_idx=r_idx,
            reflector_kind="querychain_staged",
            status=status,
            structured_payload=structured,
            text_for_curator=text,
            used_tool_calls=used_search,
            used_react_turns=used_turns,
            transcript_path=transcript_path,
        )


# ── rendering helpers ────────────────────────────────────────────────────


_KIND_LABEL = {
    "reasoning": "Reasoning",
    "search": "Search",
    "results": "Results",
    "assistant": "Assistant",
}


def _format_turns(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    out: list[str] = []
    turn = 0
    seen_action = False
    for kind, text in pairs:
        if kind == "reasoning":
            turn += 1
            seen_action = False
        elif kind == "search":
            if turn == 0 or seen_action:
                turn += 1
                seen_action = False
            seen_action = True
        else:
            if turn == 0:
                turn = 1
            seen_action = True
        out.append(f"Step {turn} {_KIND_LABEL.get(kind, kind)}: {text}")
    return "\n".join(out)


def _no_insight_output(*, qid: str, r_idx: int, reason: str) -> ReflectionOutput:
    payload = ReflectionPayload(status="no_useful_insight", notes=reason)
    structured = dataclasses.asdict(payload)
    text = _render_text_for_curator(qid=qid, r_idx=r_idx, payload=structured)
    return ReflectionOutput(
        qid=qid,
        r_idx=r_idx,
        reflector_kind="querychain_staged",
        status="no_useful_insight",
        structured_payload=structured,
        text_for_curator=text,
        used_tool_calls=0,
        used_react_turns=0,
        transcript_path=None,
    )


def _render_traj_steps(traj: Trajectory, cfg: _ReflectorCfg) -> str:
    from search_ttl.utils.tool_render import (
        has_substantive_text,
        render_tool_result,
        strip_tool_call_blocks,
    )

    def _truncate(text: str) -> str:
        cap = cfg.traj_reasoning_chars_per_step
        if cap and len(text) > cap:
            return text[:cap] + " …(truncated)"
        return text

    pairs: list[tuple[str, str]] = []
    for step in traj.steps:
        kind = step.get("type")
        if kind == "reasoning":
            chunks = step.get("output") or []
            text = " ".join(str(c) for c in chunks) if isinstance(chunks, list) else str(chunks)
            if text.strip():
                pairs.append(("reasoning", _truncate(text.strip())))
        elif kind == "tool_call":
            args = step.get("arguments") or {}
            q = args.get("query") or args.get("q") or ""
            pairs.append(("search", str(q)))
        elif kind == "tool_result":
            results_str = render_tool_result(
                step.get("output"),
                visibility=cfg.traj_tool_result_visibility,  # type: ignore[arg-type]
                max_chars_total=cfg.traj_tool_result_max_chars_total,
                max_chars_per_hit=cfg.traj_tool_result_max_chars_per_hit,
            )
            pairs.append(("results", results_str))
        elif kind == "output_text":
            chunks = step.get("output") or []
            text = " ".join(str(c) for c in chunks) if isinstance(chunks, list) else str(chunks)
            text = strip_tool_call_blocks(text)
            if has_substantive_text(text):
                pairs.append(("assistant", _truncate(text.strip())))
    return _format_turns(pairs) or "(no trajectory steps)"


def _render_user_message(
    *,
    r_idx: int,
    correct_idx: int,
    question: str,
    target: str,
    trajectories: list[Trajectory],
    rewards: list[RewardReport],
    cfg: _ReflectorCfg,
) -> str:
    from search_ttl.utils.tool_render import describe_visibility_for_prompt

    own_traj = trajectories[r_idx]
    own_reward = rewards[r_idx]
    own_judge = own_reward.judge
    correct_traj = trajectories[correct_idx]

    own_steps_block = _render_traj_steps(own_traj, cfg)
    correct_steps_block = _render_traj_steps(correct_traj, cfg)

    judge_reason = (getattr(own_judge, "reasoning", None) or "").strip()
    if len(judge_reason) > 500:
        judge_reason = judge_reason[:500].rstrip() + " …"
    judge_lines = [
        "This trajectory was judged **INCORRECT**.",
        f"- Your final extracted answer: {own_traj.extracted_answer!r}",
        f"- Correct answer (target): {target!r}",
    ]
    if judge_reason:
        judge_lines.append(f"- Judge reasoning: {judge_reason}")
    judged_result_block = "\n".join(judge_lines)

    traj_preamble = describe_visibility_for_prompt(
        cfg.traj_tool_result_visibility,  # type: ignore[arg-type]
        max_chars_total=cfg.traj_tool_result_max_chars_total,
        reasoning_chars_per_step=cfg.traj_reasoning_chars_per_step,
    )
    live_preamble = describe_visibility_for_prompt(
        cfg.live_tool_result_visibility,  # type: ignore[arg-type]
        max_chars_total=cfg.live_tool_result_max_chars_total,
    )

    title_mode_note = ""
    if cfg.traj_tool_result_visibility == "title":
        title_mode_note = (
            "\n\nIMPORTANT: results above are document TITLES ONLY — titles "
            "alone may not reveal WHY the correct trajectory's query won. Use "
            "the `search` tool to re-run the correct trajectory's decisive "
            "query (and close variants) and inspect what it retrieves before "
            "you finalize a FIND/VERIFY insight."
        )

    return f"""
You are Reflector #{r_idx} for question {own_reward.qid}.

# Your trajectory's judged result

{judged_result_block}

Your job: (1) diagnose WHICH STAGE this trajectory broke at, (2) study
the correct trajectory, and (3) submit ONE transferable lesson for that
stage that will help on FUTURE, DIFFERENT questions.

# YOUR Trajectory (the incorrect one)

Question: {question}
Target: {target}
Final answer: {own_traj.extracted_answer!r}
Status: {own_traj.status}

How to read the trajectory below: {traj_preamble}
Each step starts with `Step N <Kind>: ` (Reasoning / Search / Results / Assistant).

Steps:
{own_steps_block}

# A CORRECT trajectory for the SAME question — study & contrast

The shortest trajectory (fewest searches) that answered correctly.
Compare its wording, ordering, and pivots to yours; extract the reusable
principle behind why it worked — do NOT just copy its exact query.

Correct trajectory's final answer: {correct_traj.extracted_answer!r}

Steps:
{correct_steps_block}{title_mode_note}

# STEP 1 — Diagnose the failure stage (set `section_hint`)

Pick the ONE stage where your trajectory actually broke:

- **FIND** — it never located the right entity/document at all: queries
  were generic / fragmented / over-loaded and the relevant source never
  surfaced. The correct run found it with a better *first* query.
  → teach a better FIND query.

- **VERIFY** — a candidate name or a needed *bridge* entity DID appear
  (in your run or the correct one) but you mis-verified it, kept
  re-issuing the whole clue bundle, or — for a two-hop question (an
  attribute OF an entity you first had to identify, e.g. "the assistant
  coach OF the losing team") — you never pivoted to look up that
  attribute directly. → teach the VERIFY pivot.

- **DECIDE** — the answer-bearing evidence was actually retrieved, but
  you committed the WRONG answer (often a more *famous* look-alike), or
  you never committed (looped / returned None), or you ignored an
  explicit numeric/temporal constraint. The fix is NOT a query — it is
  a decision rule. → teach a DECIDE directive.

Set `section_hint` to exactly one of: FIND | VERIFY | DECIDE.

# STEP 2 — Write the lesson IN THAT STAGE'S FORM

The retriever is DENSE (a vector model). Quotes are NOT operators,
`AND`/`OR` are just words, and chains of quoted shards FRAGMENT the
embedding and retrieve worse. Therefore:

- **FIND** `verified_alternative_query`: ONE coherent natural-language
  sentence that names the entity type and packs the 3-5 strongest
  constraints inline (proper nouns/numbers as plain words). NO quotes,
  NO AND/OR, NO shard chain. Verify it with `search` before submitting.
  e.g. `the Indian footballer who won gold at the 1962 Asian Games and
  received the Arjuna Award that same year`

- **VERIFY** `verified_alternative_query`: candidate (or bridge) entity
  name + exactly ONE target attribute, plain words. Verify with
  `search`. e.g. `South Korea national team 2022 World Cup assistant
  coach name`

- **DECIDE**: leave `verified_alternative_query` EMPTY. Instead fill:
  · `principle`  = the Rule, an imperative ("Once a candidate matches
     the rarest 2-3 constraints, run at most one confirming query then
     commit").
  · `anti_pattern` = the Do-NOT ("do not keep launching fresh broad
     constraint searches to be sure").
  · `applies_to_future_questions_when` = when this rule fires.
  · `what_went_wrong_with_my_traj` / `verification_evidence` = the WHY,
     grounded in this trajectory.
  No `search` call is required for DECIDE.

# STEP 3 — Make it TRANSFERABLE (hard requirement)

Future questions are about DIFFERENT entities. Describe the reusable
MOVE, not this question's answer. Do NOT bake the specific answer
entity into `verified_alternative_query`/`principle`; use the entity
TYPE or a placeholder ("the novel", "<candidate>", "the losing team").
A bullet whose query only works because it contains the answer is
worthless and will be rejected.

# Tools

1. `search(query: str)` — same dense MCP retriever the agent used. Pass
   ONE field `query`, written in the chosen stage's style above (FIND =
   one sentence; VERIFY = name + one attribute; never quoted shards /
   boolean). Max {cfg.max_search_calls} calls. DECIDE needs no search.
   ▸ How results appear: {live_preamble}

2. `submit_reflection(payload: dict)` — call ONCE. Schema (all strings
   unless noted):

    {{
      "status": "ok" | "no_useful_insight" | "error",
      "scope": "case" | "strategy",
      "section_hint": "FIND" | "VERIFY" | "DECIDE",
      "trigger":                          "<the recurring scenario this fires on (entity-type level, no specific answer)>",
      "what_went_wrong_with_my_traj":     "<the failing move; for DECIDE, the WHY>",
      "what_peer_did_better":             "<the correct run's decisive move, generalized>",
      "verified_alternative_query":       "<FIND/VERIFY: the dense-correct query you verified; DECIDE: leave empty>",
      "verification_evidence":            "<FIND/VERIFY: Ran search() → docid X surfaced ...; DECIDE: why the rule holds>",
      "principle":                        "<FIND/VERIFY: one clause why the good query works; DECIDE: the Rule imperative>",
      "anti_pattern":                     "<FIND/VERIFY: why the bad query fails; DECIDE: the Do-NOT>",
      "applies_to_future_questions_when": "<broad scope: the question shape, not this answer>",
      "do_not_emit_if":                   "<when this lesson should NOT be applied>",
      "confidence": 0.0..1.0
    }}

# Constraints

- Exactly ONE insight, for ONE stage.
- FIND/VERIFY status='ok' MUST cite a `search`-verified query in
  `verification_evidence`. DECIDE status='ok' must ground the rule in
  this trajectory's failure.
- Keep it transferable (STEP 3) — no answer leakage.
- You have at most {cfg.max_react_turns} turns. Always finalize via
  `submit_reflection` (allow no_useful_insight={cfg.allow_no_useful_insight}).
""".strip()


def _safe_text(ai_message) -> str:
    c = getattr(ai_message, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            str(x.get("text") or "") if isinstance(x, dict) else str(x) for x in c
        )
    return ""


def _serialise_tool_calls(ai_message) -> list[dict]:
    out = []
    for tc in getattr(ai_message, "tool_calls", None) or []:
        if isinstance(tc, dict):
            out.append({"name": tc.get("name"), "args": tc.get("args")})
        else:
            out.append({"name": getattr(tc, "name", ""), "args": getattr(tc, "args", {})})
    return out


def _coerce_payload_dict(d: dict) -> dict:
    base = ReflectionPayload(status="no_useful_insight")
    out = dataclasses.asdict(base)
    for k, v in d.items():
        if k in out:
            out[k] = v
    try:
        out["confidence"] = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        out["confidence"] = 0.5
    # Normalize the stage so the curator routes deterministically.
    # Keep this in lock-step with staged_playbook._normalize_section_name
    # and react_staged._stage_of so the rendered text_for_curator
    # always matches where the bullet actually lands.
    sh = str(out.get("section_hint") or "").strip().lower()
    if (
        "decide" in sh or "decision" in sh or "commit" in sh or "evidence" in sh
        or "directive" in sh or "rule" in sh or "anti-drift" in sh
        or "stop" in sh or sh == "d"
    ):
        out["section_hint"] = "DECIDE"
    elif (
        "verify" in sh or "verification" in sh or "candidate" in sh
        or "pivot" in sh or "tactic_c" in sh or "tactic c" in sh
        or sh in ("v", "c")
    ):
        out["section_hint"] = "VERIFY"
    else:
        out["section_hint"] = "FIND"
    return out


def _render_text_for_curator(*, qid: str, r_idx: int, payload: dict) -> str:
    sh = str(payload.get("section_hint") or "FIND")
    head = (
        f"### Reflection from r_idx={r_idx} (qid={qid}) — "
        f"status={payload.get('status')}, stage={sh}, "
        f"confidence={payload.get('confidence')}\n\n"
        f"**Trigger**: {payload.get('trigger') or '(none)'}\n"
        f"**Stage (section_hint)**: {sh}\n"
        f"**Scope**: {payload.get('scope') or 'strategy'}\n\n"
    )
    if sh == "DECIDE":
        return head + (
            f"**Rule (principle)**: {payload.get('principle') or '(none)'}\n"
            f"**Why**: {payload.get('what_went_wrong_with_my_traj') or '(n/a)'}"
            f" | {payload.get('verification_evidence') or ''}\n"
            f"**Apply when**: "
            f"{payload.get('applies_to_future_questions_when') or '(general)'}\n"
            f"**Do NOT (anti_pattern)**: {payload.get('anti_pattern') or '(none)'}\n"
            f"**Skip if**: {payload.get('do_not_emit_if') or '(no constraint)'}\n"
        )
    return head + (
        f"**What went wrong (mine)**: "
        f"{payload.get('what_went_wrong_with_my_traj') or '(n/a)'}\n"
        f"**What peer did better**: {payload.get('what_peer_did_better') or '(n/a)'}\n"
        f"**Verified query**: `{payload.get('verified_alternative_query') or ''}`\n"
        f"**Evidence**: {payload.get('verification_evidence') or '(none)'}\n\n"
        f"**Principle**: {payload.get('principle') or '(none)'}\n\n"
        f"**Anti-pattern**: {payload.get('anti_pattern') or '(none)'}\n\n"
        f"**Applies when**: "
        f"{payload.get('applies_to_future_questions_when') or '(general)'}\n"
        f"**Skip if**: {payload.get('do_not_emit_if') or '(no constraint)'}\n"
    )

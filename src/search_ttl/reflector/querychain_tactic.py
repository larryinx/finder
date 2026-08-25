"""QuerychainTacticReflector — tactic-selecting contrastive reflector.

Contrastive loop: correct trajectories are skipped with no LLM call;
for an incorrect one, the prompt states the judged result directly and
contrasts it against the single shortest correct trajectory;
all-incorrect questions yield ``no_useful_insight``. The reflector is
shown the three search TACTICS
and must (a) pick exactly which tactic its insight teaches, set
``section_hint`` to ``TACTIC_A`` / ``TACTIC_B`` / ``TACTIC_C``, and
(b) write ``verified_alternative_query`` in THAT tactic's query style:

- TACTIC_A — exact anchor probe: short, quoted, high precision.
- TACTIC_B — constraint fingerprint: 3-5 strongest constraints only.
- TACTIC_C — candidate verification: candidate name + one attribute.

``ReflectionPayload`` shape is unchanged, so it pairs with the
``react_tactic`` curator (which routes by ``section_hint``). Pairs
naturally with the judge-only ``search_agent_lasj`` reward.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
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
    # Trajectory rendering (the agent's PAST steps shown in user prompt).
    traj_tool_result_visibility: str = "title"
    traj_tool_result_max_chars_total: int = 1500
    traj_tool_result_max_chars_per_hit: int = 500
    traj_reasoning_chars_per_step: int = 600
    # Live tool calls (ToolMessage the reflector itself sees).
    live_tool_result_visibility: str = "clipped"
    live_tool_result_max_chars_total: int = 1500
    live_tool_result_max_chars_per_hit: int = 500
    max_react_turns: int = 10
    max_search_calls: int = 6
    concurrency: int = 8
    allow_no_useful_insight: bool = True
    # ── LLM sampling for the reflector ReAct loop ────────────────────
    # ``llm_provider`` / ``llm_api_flavor`` / ``preserve_thinking``
    # mirror the rollout agent so a thinking-on reflector loop preserves
    # reasoning across turns.
    llm_provider: str = "openai_compat"  # openai_compat | openrouter
    llm_api_flavor: str = "chat_completions"
    enable_thinking: bool = False
    preserve_thinking: bool = True
    temperature: float = 0.4
    top_p: float = 0.95
    max_response_tokens: int = 8192
    llm_timeout_s: float = 600.0


@register_reflector("querychain_tactic")
class QuerychainTacticReflector(ReflectorManager):
    def __init__(self, cfg: _ReflectorCfg) -> None:
        self.cfg = cfg
        self._mcp_tools: list = []
        self._search_tool = None

    @classmethod
    def from_omegaconf(cls, cfg) -> "QuerychainTacticReflector":
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
        experience_text: str,
        *,
        save_dir: Path | None = None,
    ) -> list[ReflectionOutput]:
        qid = question_record.qid
        n = len(trajectories)

        # Correctness gating. ``rewards[i]`` is aligned with
        # ``trajectories[i]`` (the trainer groups by qid then zips).
        correct_idxs = [i for i in range(n) if rewards[i].correctness]

        # Pick the SHORTEST correct trajectory (fewest searches; ties
        # broken by fewer steps) as the single contrast example.
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

        # Only connect MCP if ≥1 incorrect trajectory will actually run
        # the reflector LLM (it needs a correct peer to contrast
        # against). All-correct or all-incorrect groups skip the LLM
        # entirely, so we never pay the MCP connection there.
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
                        experience_text=experience_text,
                        save_dir=save_dir,
                    )
                )
        return await bounded_gather(factories, concurrency=self.cfg.concurrency)

    async def _skip(
        self, r_idx: int, qid: str, reason: str
    ) -> ReflectionOutput:
        """Build a no-LLM ``no_useful_insight`` output (correct traj, or
        no correct peer to contrast against)."""
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
        experience_text: str,
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
            "Use the `search` tool to verify alternative queries before finalizing."
        )
        user_msg = _render_user_message(
            r_idx=r_idx,
            correct_idx=correct_idx,
            question=question_record.question,
            target=question_record.target,
            trajectories=trajectories,
            rewards=rewards,
            experience_text=experience_text,
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
                    "querychain reflector LLM failed qid=%s r_idx=%s: %s",
                    question_record.qid, r_idx, exc,
                )
                status = "error"
                break
            # Promote mcp_call blocks → ai.tool_calls, THEN strip them
            # before round-trip (same two-step fix as the rollout agent).
            from search_ttl.utils.tool_call_recovery import (
                recover_mcp_call_tool_calls,
                sanitize_responses_message_for_round_trip,
            )
            mcp_recovered = recover_mcp_call_tool_calls(ai)
            if mcp_recovered:
                LOG.info(
                    "querychain reflector qid=%s r_idx=%s: promoted %d "
                    "mcp_call block(s) into ai.tool_calls",
                    question_record.qid, r_idx, len(mcp_recovered),
                )
            n_stripped = sanitize_responses_message_for_round_trip(ai)
            if n_stripped:
                LOG.info(
                    "querychain reflector qid=%s r_idx=%s: stripped %d "
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

            # Recover text-form tool calls when the inference server
            # isn't wired with a tool-call parser. Handles both Qwen3
            # (<tool_call>...) and gpt-oss Harmony shapes. Mirrors the
            # same fallback in agent/search_agent.py.
            if not tcs and isinstance(ai.content, str):
                from search_ttl.utils.tool_call_recovery import recover_text_tool_calls

                recovered = recover_text_tool_calls(ai.content)
                if recovered:
                    LOG.info(
                        "querychain reflector qid=%s r_idx=%s turn=%s: "
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
                            "You did not call a tool. Please call `submit_reflection` "
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
                            label=f"querychain_reflector.search[qid={question_record.qid} r={r_idx}]",
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
            reflector_kind="querychain_tactic",
            status=status,
            structured_payload=structured,
            text_for_curator=text,
            used_tool_calls=used_search,
            used_react_turns=used_turns,
            transcript_path=transcript_path,
        )


# ── rendering helpers ────────────────────────────────────────────────────


# Display label per step kind, used by the turn-grouped renderer.
_KIND_LABEL = {
    "reasoning": "Reasoning",
    "search": "Search",
    "results": "Results",
    "assistant": "Assistant",
}


def _format_turns(pairs: list[tuple[str, str]]) -> str:
    """Render ``(kind, text)`` pairs grouped by ReAct turn.

    One turn = the model's reasoning + the search action it took +
    the results that came back, e.g.::

        Step 1 Reasoning: ...
        Step 1 Search: '...'
        Step 1 Results: ...
        Step 2 Reasoning: ...
        Step 2 Search: '...'
        Step 2 Results: ...

    A new turn opens at a ``reasoning`` step, or at a ``search`` step
    that has no preceding reasoning in the current turn / follows a
    prior action (results, another search). ``results`` / ``assistant``
    stay attached to the current turn. With thinking OFF (no reasoning
    steps) turns are delimited by ``search`` instead, yielding
    ``Step 1 Search / Step 1 Results / Step 2 Search / ...``.
    """
    if not pairs:
        return ""
    out: list[str] = []
    turn = 0
    seen_action = False  # search/results/assistant already in this turn
    for kind, text in pairs:
        if kind == "reasoning":
            turn += 1
            seen_action = False
        elif kind == "search":
            if turn == 0 or seen_action:
                turn += 1
                seen_action = False
            seen_action = True
        else:  # results / assistant
            if turn == 0:
                turn = 1
            seen_action = True
        out.append(f"Step {turn} {_KIND_LABEL.get(kind, kind)}: {text}")
    return "\n".join(out)


def _no_insight_output(*, qid: str, r_idx: int, reason: str) -> ReflectionOutput:
    """Build a ``no_useful_insight`` ReflectionOutput WITHOUT any LLM
    call. Used when the trajectory is correct (skipped) or when there
    is no correct peer to contrast an incorrect trajectory against.
    """
    payload = ReflectionPayload(status="no_useful_insight", notes=reason)
    structured = dataclasses.asdict(payload)
    text = _render_text_for_curator(qid=qid, r_idx=r_idx, payload=structured)
    return ReflectionOutput(
        qid=qid,
        r_idx=r_idx,
        reflector_kind="querychain_tactic",
        status="no_useful_insight",
        structured_payload=structured,
        text_for_curator=text,
        used_tool_calls=0,
        used_react_turns=0,
        transcript_path=None,
    )


def _render_traj_steps(traj: Trajectory, cfg: _ReflectorCfg) -> str:
    """Render a trajectory's steps using the ``traj_tool_result_*``
    config. Used for BOTH the agent's own (incorrect) trajectory and
    the contrast (shortest correct) trajectory, so the same
    ``traj_tool_result_visibility`` controls how much of the search
    results the agent sees in either one.
    """
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
    experience_text: str,
    cfg: _ReflectorCfg,
) -> str:
    from search_ttl.utils.tool_render import describe_visibility_for_prompt

    own_traj = trajectories[r_idx]
    own_reward = rewards[r_idx]
    own_judge = own_reward.judge
    correct_traj = trajectories[correct_idx]

    own_steps_block = _render_traj_steps(own_traj, cfg)
    correct_steps_block = _render_traj_steps(correct_traj, cfg)

    # Judged result, stated directly (replaces the v2 analyzer summary).
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

    # NOTE: This is a hardcoded additional note for only this reflector's user prompt.
    # In `title` visibility the rendered trajectories show document
    # titles only (no snippet bodies), so you can see WHICH query the
    # correct run issued but not WHY its hits were right. Tell the model
    # to actively re-run the winning query with the live `search` tool
    # to recover that. Empty for other visibility modes.
    title_mode_note = ""
    if cfg.traj_tool_result_visibility == "title":
        title_mode_note = (
            "\n\nIMPORTANT: the search results in both trajectories above "
            "are shown as document TITLES ONLY — you cannot see snippet "
            "bodies, so titles alone may not reveal why the correct "
            "trajectory's query won. Use the `search` tool to re-run the "
            "correct trajectory's decisive query (and close variants) and "
            "inspect what it actually retrieves, so you can pin down the "
            "exact query wording that leads to the right document before "
            "you finalize."
        )

    return f"""
You are Reflector #{r_idx} for question {own_reward.qid}.

# Your trajectory's judged result

{judged_result_block}

Your job: work out WHY this query approach failed, study the correct
trajectory shown further below, and propose ONE playbook improvement
focused on QUERY DESIGN.

# YOUR Trajectory (the incorrect one)

Question: {question}
Target: {target}
Final answer: {own_traj.extracted_answer!r}
Status: {own_traj.status}

How to read the trajectory below: {traj_preamble}
Each step starts with `Step N: ` followed by its kind
(search / results / reasoning / assistant).

Steps:
{own_steps_block}

# A CORRECT trajectory for the SAME question — study & contrast

Below is the shortest trajectory (fewest searches) that answered this
question correctly. Compare its query wording, ordering, and pivots to
yours, then lift the transferable lesson into `what_peer_did_better` /
`verified_alternative_query`. Do NOT just copy its exact query verbatim
— extract the reusable query-design principle behind why it worked.

Correct trajectory's final answer: {correct_traj.extracted_answer!r}

Steps:
{correct_steps_block}{title_mode_note}

# Current Experience playbook

```
{experience_text}
```

# Tactics — pick the ONE tactic your insight teaches

The playbook has exactly THREE tactic sections. Decide which tactic the
correct trajectory's decisive move exemplifies (the move your failed
trajectory should have used), set `section_hint` to that tactic's key,
and write `verified_alternative_query` in THAT tactic's query style.
Verify it with the `search` tool in the same style before submitting.

- TACTIC_A — exact anchor probe.
  Use for: rare phrases, titles, exact numbers, document names.
  Query style: short, quoted, high precision (a verbatim title /
  number / distinctive phrase, optionally quoted).
  e.g.  "Assessing Entrepreneurial Orientation of Young Graduates" 1979

- TACTIC_B — constraint fingerprint query.
  Use when: no single clue is distinctive.
  Query style: combine the 3-5 STRONGEST constraints into one natural
  sentence — NOT every clue, NOT quoted keyword shards.
  e.g.  Indian footballer who won gold at the 1962 Asian Games and
        received the Arjuna Award the same year

- TACTIC_C — candidate verification query.
  Use after: a candidate name has appeared (yours or the correct run's).
  Query style: candidate name + exactly ONE target attribute.
  e.g.  Tulsidas Balaram birth year

Pick exactly one. Set `section_hint` to one of: TACTIC_A | TACTIC_B |
TACTIC_C. The `verified_alternative_query` you submit MUST be written
in the chosen tactic's style above.

# Tools

Use the tools to OPTIMIZE the experience playbook. Pick whichever
direction the trajectories support and verify it against the live
retriever before submitting:

- If you are PROPOSING a successful pattern (drawn from a peer or
  retry), run `search` to confirm the alternative query actually
  surfaces the relevant doc.
- If you are FLAGGING a recurring failure mode, run `search` to confirm
  the failing query shape really returns nothing useful (or returns the
  wrong thing) so the anti-pattern is grounded, not speculative.

Then submit a structured insight via `submit_reflection`.

1. `search(query: str)` — same MCP retriever the agent used.
   ▸ Pass ONE field only: `query`. Write `query` in the chosen tactic's
     query style (see "# Tactics" above) — TACTIC_A short/quoted,
     TACTIC_B a 3-5 constraint sentence, TACTIC_C name + one attribute.
   ▸ Use it to verify alternative queries (max {cfg.max_search_calls} calls).
   ▸ DO NOT pass ReflectionPayload fields here.
   ▸ How `search` results will appear back to you: {live_preamble}

2. `submit_reflection(payload: dict)` — call ONCE with your final structured payload.
   ▸ Pass a SINGLE keyword `payload` whose value is the JSON object below.
     Example:
       submit_reflection(payload={{"status": "ok", "principle": "...", ...}})

   Schema. Each field maps to ONE slot of the multi-line playbook
   bullet the curator will write (Task / Pre-search state / Good
   query / Bad query / Caveat). All fields are strings unless noted.

    {{
      "status": "ok" | "no_useful_insight" | "already_covered" | "error",
      "scope": "case" | "strategy",
      "section_hint": "TACTIC_A" | "TACTIC_B" | "TACTIC_C"  (which tactic this insight teaches — see # Tactics),
      "trigger":                          "<→ Task: the recurring search scenario this bullet fires on>",
      "what_went_wrong_with_my_traj":     "<→ Pre-search state + Bad query source; include the failing query VERBATIM if recoverable>",
      "what_peer_did_better":             "<peer's working approach, or empty>",
      "verified_alternative_query":       "<→ Good query: written in the chosen tactic's style (see # Tactics above)>",
      "verification_evidence":            "Ran search() → docid X showed ...",
      "principle":                        "<→ the '— why it works' clause next to Good query: one short clause>",
      "anti_pattern":                     "<→ the '— why it fails' clause next to Bad query: one short clause>",
      "applies_to_future_questions_when": "<broad scope hint, narrows the Caveat>",
      "do_not_emit_if":                   "<→ Caveat: when this bullet should NOT fire>",
      "confidence": 0.0..1.0
    }}

# Constraints

- Stay focused on QUERY DESIGN, not snippet content.
- Every `status='ok'` payload MUST cite a verified alternative or peer
  pattern in `verification_evidence`.
- If your insight is already covered by the current experience, return
  `status='already_covered'` and an empty principle.
- You currently have at most {cfg.max_react_turns} turns remaining before the reflector is forced to finalize.
- Finalize via `submit_reflection` (always call it, even on
  no_useful_insight={cfg.allow_no_useful_insight}).
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
    return out


def _render_text_for_curator(*, qid: str, r_idx: int, payload: dict) -> str:
    return (
        f"### Reflection from r_idx={r_idx} (qid={qid}) — "
        f"status={payload.get('status')}, confidence={payload.get('confidence')}\n\n"
        f"**Trigger**: {payload.get('trigger') or '(none)'}\n"
        f"**Section hint**: {payload.get('section_hint') or 'TACTIC_B'}\n"
        f"**Scope**: {payload.get('scope') or 'strategy'}\n\n"
        f"**What went wrong (mine)**: "
        f"{payload.get('what_went_wrong_with_my_traj') or '(n/a)'}\n"
        f"**What peer did better**: {payload.get('what_peer_did_better') or '(n/a)'}\n"
        f"**Verified alternative**: `{payload.get('verified_alternative_query') or ''}`\n"
        f"**Evidence**: {payload.get('verification_evidence') or '(none)'}\n\n"
        f"**Principle**: {payload.get('principle') or '(none)'}\n\n"
        f"**Anti-pattern**: {payload.get('anti_pattern') or '(none)'}\n\n"
        f"**Applies when**: "
        f"{payload.get('applies_to_future_questions_when') or '(general)'}\n"
        f"**Skip if**: {payload.get('do_not_emit_if') or '(no constraint)'}\n"
    )

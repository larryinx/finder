"""ReActTacticCurator — ReAct curator for the 3-tactic ``tactic_playbook`` backend.

Key behaviours:

1. Bullets are routed into one of three FIXED tactic sections —
   ``TACTIC_A`` (exact anchor probe), ``TACTIC_B`` (constraint
   fingerprint), ``TACTIC_C`` (candidate verification) — taken from the
   reflection's ``section_hint``.
2. ``good_query`` style is TACTIC-CONDITIONAL: the no-quoted-shard
   validation is enforced for TACTIC_B / TACTIC_C, but RELAXED for
   TACTIC_A (whose entire point is a short, quoted, high-precision
   anchor probe).
3. The char-budget rollback is backend-agnostic
   (``experience.__class__.from_text``).

Tools implemented:
- list_sections, read_section, read_bullet, search_bullets
- add_bullet, modify_bullet, delete_bullet
- remaining_budget
- finalize  (ends the loop)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from search_ttl.experience.base import Experience
from search_ttl.curator.manager import CuratorManager
from search_ttl.curator.schemas import OpRecord, CuratorStepResult
from search_ttl.reflector.schemas import ReflectionOutput
from search_ttl.trainer.registry import register_curator
from search_ttl.utils.persistence import append_jsonl, write_json
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


@dataclass
class _Conserv:
    max_add_per_step: int = 5
    max_modify_per_step: int = 3
    max_delete_per_step: int = 2
    max_total_edits_per_step: int = 8


@dataclass
class _OptCfg:
    """Per-curator LLM + budget config.

    Note: mini/micro batch sizing and epochs live under ``cfg.actor`` —
    the trainer drives the schedule, this manager just executes one
    update() over the mini-batch it's handed.
    """

    llm_model: str
    llm_base_url: str
    llm_api_key: str
    max_react_turns: int = 24
    conservativeness: _Conserv = field(default_factory=_Conserv)
    char_budget: int = 48000
    reject_if_over_budget_ratio: float = 1.2
    # ── Provider + thinking / reasoning preservation. Mirrors the
    # rollout agent's four toggles so a thinking-on curator ReAct
    # loop preserves reasoning across its ≤max_react_turns turns.
    llm_provider: str = "openai_compat"  # openai_compat | openrouter
    llm_api_flavor: str = "chat_completions"
    enable_thinking: bool = False
    preserve_thinking: bool = True
    temperature: float = 0.2
    top_p: float = 0.95
    # Curator's ReAct loop reads/writes playbook over up to
    # max_react_turns turns; each turn's response includes tool_calls
    # plus a short rationale. 8192 is plenty unless you bump
    # max_react_turns or playbook size dramatically.
    max_response_tokens: int = 8192
    llm_timeout_s: float = 600.0


@register_curator("react_tactic")
class ReActTacticCurator(CuratorManager):
    def __init__(self, cfg: _OptCfg) -> None:
        self.cfg = cfg

    @classmethod
    def from_omegaconf(cls, cfg) -> "ReActTacticCurator":
        c = cfg.get("conservativeness", {}) or {}
        ocfg = _OptCfg(
            llm_model=str(cfg.get("llm_model", ""))
            or os.environ.get("CURATOR_LLM_MODEL", "")
            or os.environ.get("AGENT_LLM_MODEL", ""),
            llm_base_url=str(cfg.get("llm_base_url", ""))
            or os.environ.get("CURATOR_LLM_BASE_URL", "")
            or os.environ.get("AGENT_LLM_BASE_URL", ""),
            llm_api_key=str(cfg.get("llm_api_key", ""))
            or os.environ.get("CURATOR_LLM_API_KEY", "")
            or os.environ.get("AGENT_LLM_API_KEY", ""),
            max_react_turns=int(cfg.get("max_react_turns", 24)),
            conservativeness=_Conserv(
                max_add_per_step=int(c.get("max_add_per_step", 5)),
                max_modify_per_step=int(c.get("max_modify_per_step", 3)),
                max_delete_per_step=int(c.get("max_delete_per_step", 2)),
                max_total_edits_per_step=int(c.get("max_total_edits_per_step", 8)),
            ),
            char_budget=int(cfg.get("char_budget", 48000)),
            reject_if_over_budget_ratio=float(cfg.get("reject_if_over_budget_ratio", 1.2)),
            llm_provider=str(cfg.get("llm_provider", "openai_compat")),
            llm_api_flavor=str(cfg.get("llm_api_flavor", "chat_completions")),
            enable_thinking=bool(cfg.get("enable_thinking", False)),
            preserve_thinking=bool(cfg.get("preserve_thinking", True)),
            temperature=float(cfg.get("temperature", 0.2)),
            top_p=float(cfg.get("top_p", 0.95)),
            max_response_tokens=int(cfg.get("max_response_tokens", 8192)),
            llm_timeout_s=float(cfg.get("llm_timeout_s", 600.0)),
        )
        return cls(ocfg)

    @traceable(name="Curator.update", run_type="chain")
    async def update(
        self,
        experience: Experience,
        reflections: list[ReflectionOutput],
        *,
        step: int,
        save_dir: Path | None = None,
    ) -> tuple[Experience, CuratorStepResult]:
        # Snapshot the pre-edit state for rollback.
        pre_text = experience.to_text()
        # Mutable container we wire into tool closures.
        ledger = _Ledger(
            experience=experience,
            conserv=self.cfg.conservativeness,
            step=step,
            char_budget=self.cfg.char_budget,
            reject_ratio=self.cfg.reject_if_over_budget_ratio,
        )
        finalize_summary: list[str] = []

        # Persist input (audit).
        if save_dir is not None:
            write_json(save_dir / "curator/input.json",
                       [_reflection_brief(r) for r in reflections])

        # Bind tools.
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_core.tools import tool as tool_dec

        from search_ttl.utils.llm_builder import build_chat_llm

        @tool_dec
        def list_sections() -> str:
            """List sections + bullet counts."""
            sections = ledger.experience.list_sections()
            return json.dumps([dataclasses.asdict(s) for s in sections])

        @tool_dec
        def read_section(section: str) -> str:
            """Read all bullets in a section."""
            bullets = ledger.experience.read_section(section)
            return json.dumps(
                [
                    {"bullet_id": b.bullet_id, "section": b.section, "content": b.content}
                    for b in bullets
                ]
            )

        @tool_dec
        def read_bullet(bullet_id: str) -> str:
            """Read one bullet."""
            b = ledger.experience.read_bullet(bullet_id)
            if b is None:
                return f"(no bullet {bullet_id!r})"
            return json.dumps({"bullet_id": b.bullet_id, "section": b.section, "content": b.content})

        @tool_dec
        def search_bullets(keyword: str, top_k: int = 5) -> str:
            """Find bullets containing a keyword."""
            bs = ledger.experience.search_bullets(keyword, top_k=top_k)
            return json.dumps(
                [
                    {"bullet_id": b.bullet_id, "section": b.section, "content": b.content}
                    for b in bs
                ]
            )

        # ── Structured-bullet helpers (shared by add_bullet / modify_bullet) ──
        #
        # The Experience layer still stores a single content string per
        # bullet; the multi-field surface lives at the tool boundary so the
        # LLM is structurally guided to fill each slot. ``_assemble_bullet``
        # joins the slots into the rendered markdown that Experience sees;
        # ``_validate_good_query`` enforces the "no quoted-shard fragments"
        # rule at dispatch time so the curator can't silently regress.

        def _assemble_bullet(
            task: str,
            pre_search_state: str,
            good_query: str,
            good_rationale: str,
            bad_query: str,
            bad_rationale: str,
            caveat: str,
        ) -> str:
            lines = [
                f"**Task**: {task.strip()}",
                f"**Pre-search state**: {pre_search_state.strip()}",
                f"**Good query**: `{good_query.strip()}` — {good_rationale.strip()}",
            ]
            if bad_query.strip():
                tail = f" — {bad_rationale.strip()}" if bad_rationale.strip() else ""
                lines.append(f"**Bad query**: `{bad_query.strip()}`{tail}")
            if caveat.strip():
                lines.append(f"**Caveat**: {caveat.strip()}")
            return "\n".join(lines)

        def _is_anchor_tactic(section: str) -> bool:
            """True iff ``section`` routes to TACTIC_A (exact anchor
            probe). TACTIC_A queries are SUPPOSED to be short, quoted,
            high-precision anchors, so the no-quoted-shard rule does NOT
            apply there. TACTIC_B / TACTIC_C still get the strict check.
            """
            s = (section or "").strip().lower()
            return (
                "tactic_a" in s
                or "tactic a" in s
                or "exact anchor" in s
                or s == "a"
            )

        def _validate_good_query(section: str, good_query: str) -> str:
            # TACTIC_A is an exact anchor probe — quoting is the point.
            if _is_anchor_tactic(section):
                return ""
            import re as _re
            n_quoted = len(_re.findall(r'"[^"]+"', good_query))
            if n_quoted >= 3:
                return (
                    f"rejected: `good_query` contains {n_quoted} separately "
                    "quoted phrases. For TACTIC_B / TACTIC_C that is the "
                    "fragmented form which belongs in `bad_query`, not in "
                    "`good_query`. Rewrite `good_query` per the chosen "
                    "tactic's style: TACTIC_B = ONE sentence combining the "
                    "3-5 strongest constraints (proper nouns inline, not "
                    "quoted shards); TACTIC_C = candidate name + ONE target "
                    "attribute. (If this insight is really an exact-anchor "
                    "probe, route it to TACTIC_A instead, where quoting is "
                    "allowed.)"
                )
            return ""

        @tool_dec
        def add_bullet(
            section: str,
            task: str,
            pre_search_state: str,
            good_query: str,
            good_rationale: str,
            bad_query: str = "",
            bad_rationale: str = "",
            caveat: str = "",
            source_reflection_ids: list[str] | None = None,
        ) -> str:
            """Add a new bullet to ``section``.

            A bullet has five slots; supply them as separate fields. The
            tool assembles them into a multi-line markdown bullet of the
            form

                **Task**: <task>
                **Pre-search state**: <pre_search_state>
                **Good query**: `<good_query>` — <good_rationale>
                **Bad query**: `<bad_query>` — <bad_rationale>     (omitted if bad_query empty)
                **Caveat**: <caveat>                                (omitted if caveat empty)

            Field semantics (and where each one is sourced from the
            reflection payload — see the reflector's submit_reflection
            schema for the source-side names):

            - ``section``: which TACTIC this bullet belongs to. MUST be
              exactly one of ``TACTIC_A`` / ``TACTIC_B`` / ``TACTIC_C``,
              taken from the reflection's ``section_hint``:
                · TACTIC_A — exact anchor probe (rare phrase / title /
                  exact number / document name).
                · TACTIC_B — constraint fingerprint (no single clue is
                  distinctive).
                · TACTIC_C — candidate verification (a candidate name
                  has appeared).
            - ``task``: one short clause naming the recurring search
              scenario this bullet fires on. Source: ``trigger``.
            - ``pre_search_state``: one short clause describing what
              the agent had already established before its failing
              query. Source: extract from ``what_went_wrong_with_my_traj``.
            - ``good_query``: written in the chosen tactic's STYLE
              (source: reflection's ``verified_alternative_query``):
                · TACTIC_A — short, quoted, high precision: a verbatim
                  title / number / distinctive phrase, quoting allowed.
                · TACTIC_B — ONE natural sentence combining the 3-5
                  STRONGEST constraints (proper nouns inline, NOT a
                  chain of quoted keyword shards, NOT every clue).
                · TACTIC_C — candidate name + exactly ONE target
                  attribute (e.g. ``Tulsidas Balaram birth year``).
              The tool rejects ``good_query`` with 3+ quoted phrases for
              TACTIC_B / TACTIC_C only; TACTIC_A is exempt.
            - ``good_rationale``: one short clause naming WHY the Good
              query lands. Source: ``principle``.
            - ``bad_query``: the fragmented / quote-heavy / clue-dropping
              form of the same content, for contrast. Often the failing
              query verbatim from ``what_went_wrong_with_my_traj``.
              Leave empty ("") if no concrete failure shape is
              recoverable; the Bad line is then omitted from the bullet.
            - ``bad_rationale``: one short clause naming WHY the Bad
              query fails. Source: ``anti_pattern``. Ignored if
              ``bad_query`` is empty.
            - ``caveat``: one short clause naming when this bullet is
              NOT safe to apply (the over-application trap). Source:
              ``do_not_emit_if`` (use the converse of
              ``applies_to_future_questions_when`` as a fallback).
              Leave empty if no constraint applies.
            """
            err = _validate_good_query(section, good_query)
            if err:
                return err
            content = _assemble_bullet(
                task, pre_search_state, good_query, good_rationale,
                bad_query, bad_rationale, caveat,
            )
            return ledger.add(section, content, source_reflection_ids or [])

        @tool_dec
        def modify_bullet(
            bullet_id: str,
            task: str,
            pre_search_state: str,
            good_query: str,
            good_rationale: str,
            bad_query: str = "",
            bad_rationale: str = "",
            caveat: str = "",
            source_reflection_ids: list[str] | None = None,
        ) -> str:
            """Modify an existing bullet by supplying the new full set of fields.

            Same field semantics, formatting, and validation as
            ``add_bullet`` (read its docstring for the per-field guidance
            and the ``good_query`` quoted-shard check). The bullet ID and
            section are preserved; everything else is replaced.

            Call ``read_bullet(bullet_id)`` first to see the existing
            content before deciding what to change. The tactic is taken
            from the existing bullet's section, so the ``good_query``
            quoted-shard check is relaxed iff that bullet lives in
            TACTIC_A.
            """
            _existing = ledger.experience.read_bullet(bullet_id)
            _sec = _existing.section if _existing is not None else ""
            err = _validate_good_query(_sec, good_query)
            if err:
                return err
            content = _assemble_bullet(
                task, pre_search_state, good_query, good_rationale,
                bad_query, bad_rationale, caveat,
            )
            return ledger.modify(bullet_id, content, source_reflection_ids or [])

        @tool_dec
        def delete_bullet(
            bullet_id: str,
            justification: str,
            source_reflection_ids: list[str] | None = None,
        ) -> str:
            """Delete a bullet (only with justification)."""
            return ledger.delete(bullet_id, justification, source_reflection_ids or [])

        @tool_dec
        def remaining_budget() -> str:
            """How many ADD/MODIFY/DELETE you have left this step."""
            return json.dumps(ledger.remaining())

        @tool_dec
        def finalize(summary: str) -> str:
            """End optimization for this step."""
            finalize_summary.append(summary)
            return "finalized"

        tools = [
            list_sections,
            read_section,
            read_bullet,
            search_bullets,
            add_bullet,
            modify_bullet,
            delete_bullet,
            remaining_budget,
            finalize,
        ]

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
        bound = llm.bind_tools(tools)

        sys_msg = "You are the ExperienceCurator. Apply the reflections to the playbook."
        user_msg = _render_curator_user_msg(
            reflections=reflections,
            conserv=self.cfg.conservativeness,
            max_react_turns=self.cfg.max_react_turns,
            char_budget=self.cfg.char_budget,
        )
        messages = [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]

        tool_calls_log: list[dict] = []
        used_turns = 0
        for turn_idx in range(self.cfg.max_react_turns):
            used_turns += 1
            if finalize_summary:
                break
            try:
                ai = await bound.ainvoke(messages)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("curator LLM error step=%s: %s", step, exc)
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
                    "curator step=%s: promoted %d mcp_call block(s) "
                    "into ai.tool_calls",
                    step, len(mcp_recovered),
                )
            n_stripped = sanitize_responses_message_for_round_trip(ai)
            if n_stripped:
                LOG.info(
                    "curator step=%s: stripped %d vLLM-rejected "
                    "content block(s) before round-trip",
                    step, n_stripped,
                )
            messages.append(ai)

            tcs = getattr(ai, "tool_calls", None) or []

            # Same text-form tool-call recovery the rollout agent uses,
            # covering both Qwen3 ``<tool_call>...`` and gpt-oss Harmony
            # ``<|channel|>...to=functions.X<|message|>...<|call|>``
            # shapes. Without this branch the curator would loop
            # forever on the "please call finalize" nudge.
            if not tcs and isinstance(ai.content, str):
                from search_ttl.utils.tool_call_recovery import recover_text_tool_calls

                recovered = recover_text_tool_calls(ai.content)
                if recovered:
                    LOG.info(
                        "curator step=%s: recovered %d tool_call(s) from text",
                        step, len(recovered),
                    )
                    ai.tool_calls = recovered
                    tcs = recovered

            if not tcs:
                # Force a finalize.
                messages.append(
                    HumanMessage(
                        content="Please call `finalize(summary='...')` to end the step."
                    )
                )
                continue

            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                tcid = (
                    tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "") or ""
                )
                tool_map = {t.name: t for t in tools}
                tool = tool_map.get(name)
                if tool is None:
                    result = f"unknown tool {name!r}"
                else:
                    try:
                        result = tool.invoke(args)
                    except Exception as exc:  # noqa: BLE001
                        result = f"[tool_error] {type(exc).__name__}: {exc}"
                tool_calls_log.append(
                    {"turn": turn_idx, "tool": name, "args": args, "result": result}
                )
                messages.append(ToolMessage(content=str(result), tool_call_id=tcid))
                if name == "finalize":
                    break  # finalize_summary already populated

        # Char budget check — reject the step if catastrophically over.
        new_chars = ledger.experience.char_count()
        rejected = False
        rejection_reason: str | None = None
        if new_chars > self.cfg.char_budget * self.cfg.reject_if_over_budget_ratio:
            rejected = True
            rejection_reason = (
                f"new playbook chars={new_chars} exceeds "
                f"{self.cfg.char_budget} × {self.cfg.reject_if_over_budget_ratio} = "
                f"{int(self.cfg.char_budget * self.cfg.reject_if_over_budget_ratio)}"
            )
            # Roll back to pre-edit state. Reconstruct via the SAME
            # backend class as the live experience (e.g. TacticPlaybook)
            # so the playbook is not mis-parsed on rollback.
            ledger.experience = type(experience).from_text(pre_text)
            ledger.ops = []

        # Persist tool_calls log + ops_applied + final summary.
        if save_dir is not None:
            base = save_dir / "curator"
            base.mkdir(parents=True, exist_ok=True)
            for entry in tool_calls_log:
                await append_jsonl(base / "tool_calls.jsonl", entry)
            for op in ledger.ops:
                await append_jsonl(base / "ops_applied.jsonl", dataclasses.asdict(op))
            (base / "final_summary.txt").write_text(
                "\n".join(finalize_summary) or "(no finalize summary)",
                encoding="utf-8",
            )
            if rejected:
                (base / "STEP_REJECTED.txt").write_text(rejection_reason or "", encoding="utf-8")

        consumed_ids = {
            sid
            for op in ledger.ops
            for sid in (op.source_reflection_ids or [])
        }
        leftover = [
            r.qid + ":" + str(r.r_idx)
            for r in reflections
            if r.status == "ok"
            and (r.qid + ":" + str(r.r_idx)) not in consumed_ids
        ]
        return ledger.experience, CuratorStepResult(
            step=step,
            ops=list(ledger.ops),
            final_summary="\n".join(finalize_summary),
            leftover_reflection_ids=leftover,
            react_turns_used=used_turns,
            rejected=rejected,
            rejection_reason=rejection_reason,
        )


# ── helpers ──────────────────────────────────────────────────────────────


@dataclass
class _Ledger:
    experience: Experience
    conserv: _Conserv
    step: int
    ops: list[OpRecord] = field(default_factory=list)
    # Per-call char-budget guard. ``0`` disables the check (useful
    # for tests that don't care about playbook size). The two
    # thresholds correspond to the soft warning (``char_budget``) and
    # the hard refusal (``char_budget × reject_ratio``).
    char_budget: int = 0
    reject_ratio: float = 1.2

    # ── budget thresholds + tentative-apply helper ────────────────────
    @property
    def _hard_cap(self) -> int:
        return int(self.char_budget * self.reject_ratio)

    def _char_check(self, projected: int, op: str) -> tuple[str, str]:
        """Classify a projected playbook char count.

        Returns ``(kind, msg)`` where ``kind`` is one of:
          - ``"ok"``     — under soft budget, commit silently
          - ``"warn"``   — between soft budget and hard cap; commit
                           but tell the LLM to start consolidating
          - ``"error"``  — over hard cap; refuse the edit and tell
                           the LLM to delete + merge duplicates first
        Disabled when ``char_budget <= 0`` (always returns ``"ok"``).
        """
        if self.char_budget <= 0:
            return "ok", ""
        hard = self._hard_cap
        if projected > hard:
            return "error", (
                f"refused: applying this {op} would push the playbook to "
                f"~{projected} chars, over the hard cap of {hard} "
                f"(char_budget {self.char_budget} × reject_ratio "
                f"{self.reject_ratio}). The playbook is saturated. "
                "DO NOT add more bullets right now. Instead: "
                "(1) call `list_sections` / `search_bullets` to find "
                "duplicate or near-duplicate bullets, "
                "(2) `delete_bullet` the redundant ones, "
                "(3) `modify_bullet` a remaining bullet to absorb the new "
                "insight. Re-check headroom with `remaining_budget`."
            )
        if projected > self.char_budget:
            return "warn", (
                f"playbook would grow to ~{projected} chars, past the "
                f"soft budget of {self.char_budget} (hard cap "
                f"{hard}). The edit was applied, but consider "
                "consolidating duplicates before issuing further ADDs — "
                "subsequent edits will be REFUSED once the hard cap is "
                "reached."
            )
        return "ok", ""

    def remaining(self) -> dict[str, int]:
        used = self._counts()
        out: dict[str, int] = {
            "add_left": max(0, self.conserv.max_add_per_step - used["ADD"]),
            "modify_left": max(0, self.conserv.max_modify_per_step - used["MODIFY"]),
            "delete_left": max(0, self.conserv.max_delete_per_step - used["DELETE"]),
            "total_left": max(
                0,
                self.conserv.max_total_edits_per_step
                - sum(used.values()),
            ),
        }
        if self.char_budget > 0:
            current = self.experience.char_count()
            hard = self._hard_cap
            out["playbook_chars_current"] = current
            out["playbook_chars_soft_budget"] = self.char_budget
            out["playbook_chars_hard_cap"] = hard
            # Negative when already over → tells the LLM how much it
            # has to free up before the next ADD will be accepted.
            out["playbook_chars_until_warning"] = self.char_budget - current
            out["playbook_chars_until_refused"] = hard - current
        return out

    def _counts(self) -> dict[str, int]:
        out = {"ADD": 0, "MODIFY": 0, "DELETE": 0}
        for op in self.ops:
            out[op.op] += 1
        return out

    def _check(self, kind: str) -> tuple[bool, str]:
        used = self._counts()
        budgets = {
            "ADD": self.conserv.max_add_per_step,
            "MODIFY": self.conserv.max_modify_per_step,
            "DELETE": self.conserv.max_delete_per_step,
        }
        if used[kind] >= budgets[kind]:
            return False, f"budget exhausted for {kind}"
        if sum(used.values()) >= self.conserv.max_total_edits_per_step:
            return False, "total edit budget exhausted"
        return True, ""

    def add(self, section: str, content: str, srcs: list[str]) -> str:
        ok, why = self._check("ADD")
        if not ok:
            return f"refused: {why}"
        # Apply tentatively against a value-typed copy so we can roll
        # back without mutation if the char check fails.
        new_exp, bullet = self.experience.add_bullet(section, content)
        kind, msg = self._char_check(new_exp.char_count(), op="ADD")
        if kind == "error":
            return msg  # don't commit — playbook saturated
        self.experience = new_exp
        self.ops.append(
            OpRecord(
                step=self.step,
                op="ADD",
                section=bullet.section,
                bullet_id_before=None,
                bullet_id_after=bullet.bullet_id,
                content_before=None,
                content_after=bullet.content,
                source_reflection_ids=srcs,
            )
        )
        result: dict[str, Any] = {
            "bullet_id": bullet.bullet_id,
            "section": bullet.section,
        }
        if kind == "warn":
            result["warning"] = msg
        return json.dumps(result)

    def modify(self, bullet_id: str, new_content: str, srcs: list[str]) -> str:
        ok, why = self._check("MODIFY")
        if not ok:
            return f"refused: {why}"
        before = self.experience.read_bullet(bullet_id)
        if before is None:
            return f"refused: bullet {bullet_id!r} not found"
        # Tentative apply, then char-check.
        new_exp, after = self.experience.modify_bullet(bullet_id, new_content)
        kind, msg = self._char_check(new_exp.char_count(), op="MODIFY")
        if kind == "error":
            return msg
        self.experience = new_exp
        self.ops.append(
            OpRecord(
                step=self.step,
                op="MODIFY",
                section=before.section,
                bullet_id_before=before.bullet_id,
                bullet_id_after=before.bullet_id,
                content_before=before.content,
                content_after=after.content if after else new_content,
                source_reflection_ids=srcs,
            )
        )
        result: dict[str, Any] = {
            "bullet_id": bullet_id,
            "section": before.section,
        }
        if kind == "warn":
            result["warning"] = msg
        return json.dumps(result)

    def delete(self, bullet_id: str, justification: str, srcs: list[str]) -> str:
        ok, why = self._check("DELETE")
        if not ok:
            return f"refused: {why}"
        before = self.experience.read_bullet(bullet_id)
        if before is None:
            return f"refused: bullet {bullet_id!r} not found"
        new_exp, deleted = self.experience.delete_bullet(bullet_id, reason=justification)
        if not deleted:
            return f"refused: bullet {bullet_id!r} not found (race?)"
        self.experience = new_exp
        self.ops.append(
            OpRecord(
                step=self.step,
                op="DELETE",
                section=before.section,
                bullet_id_before=before.bullet_id,
                bullet_id_after=None,
                content_before=before.content,
                content_after=None,
                source_reflection_ids=srcs,
                justification=justification,
            )
        )
        return json.dumps({"deleted": True})


def _reflection_brief(r: ReflectionOutput) -> dict:
    return {
        "qid": r.qid,
        "r_idx": r.r_idx,
        "status": r.status,
        "section_hint": r.structured_payload.get("section_hint"),
        "trigger": r.structured_payload.get("trigger"),
        "principle": r.structured_payload.get("principle"),
        "confidence": r.structured_payload.get("confidence"),
    }


def _render_curator_user_msg(
    *,
    reflections: list[ReflectionOutput],
    conserv: _Conserv,
    max_react_turns: int,
    char_budget: int,
) -> str:
    ok = [r for r in reflections if r.status == "ok"]
    skip = [r for r in reflections if r.status != "ok"]
    body_parts = [
        "# Reflections",
        "",
        f"## OK reflections (n={len(ok)})",
        "",
    ]
    for r in ok:
        body_parts.append(r.text_for_curator)
        body_parts.append("---")
    if skip:
        body_parts.append(f"## Skipped (n={len(skip)})")
        for r in skip:
            body_parts.append(
                f"- qid={r.qid} r_idx={r.r_idx} status={r.status}"
            )

    return f"""\
You are the ExperienceCurator. Apply the reflections below to the playbook.

# Tactics (the playbook has exactly THREE sections — route every bullet)

Every bullet belongs to exactly one tactic section. Route it using the
reflection's `section_hint`. The `good_query` you write MUST follow that
tactic's query style:

- TACTIC_A — exact anchor probe.
  When: rare phrases, titles, exact numbers, document names.
  good_query style: short, quoted, high precision (a verbatim title /
  number / distinctive phrase; quoting IS allowed and expected here).
- TACTIC_B — constraint fingerprint query.
  When: no single clue is distinctive.
  good_query style: ONE natural sentence combining the 3-5 STRONGEST
  constraints — proper nouns inline, NOT quoted keyword shards, NOT
  every clue.
- TACTIC_C — candidate verification query.
  When: a candidate name has appeared.
  good_query style: candidate name + exactly ONE target attribute.

# Per-step budget
- ADD: up to {conserv.max_add_per_step}
- MODIFY: up to {conserv.max_modify_per_step}
- DELETE: up to {conserv.max_delete_per_step}
- TOTAL: up to {conserv.max_total_edits_per_step}
- React turns: up to {max_react_turns}
- Final playbook char cap: {char_budget}

# How to work
1. Use `list_sections` / `read_section` / `search_bullets` to ground.
2. Prefer `modify_bullet` when an existing bullet covers the same trigger
   AND lives in the same tactic section.
3. Skip reflections with status != 'ok' OR confidence < 0.4.
4. Pick highest-confidence + broadest-applicability when over budget.
5. Each bullet has five slots — **Task**, **Pre-search state**,
   **Good query** (+ rationale), **Bad query** (+ rationale), **Caveat**
   — supplied as separate fields to `add_bullet` / `modify_bullet`,
   plus a `section` ∈ {{TACTIC_A, TACTIC_B, TACTIC_C}} from the
   reflection's `section_hint`. Read the tool docstrings before
   calling: they describe each field's source and the per-tactic
   `good_query` style. The tool assembles the fields into the rendered
   bullet and rejects 3+ quoted-shard `good_query` for TACTIC_B /
   TACTIC_C (TACTIC_A is exempt — quoting is its whole point).
6. **Watch the playbook char budget.** Every `add_bullet` /
   `modify_bullet` call checks the projected playbook size:
   - If the result would exceed the SOFT budget ({char_budget} chars),
     the edit is applied but the response carries a `warning` field.
     When you see this warning, switch to consolidation mode — favour
     `modify_bullet` over further `add_bullet`s and look for
     duplicates to delete.
   - If the result would exceed the HARD cap (soft × reject_ratio), the
     edit is REFUSED and you must use `list_sections` / `search_bullets`
     to find duplicates, then `delete_bullet` the redundant ones and
     `modify_bullet` a remaining bullet to absorb the new insight.
   Call `remaining_budget()` any time to see current playbook chars
   vs. soft budget vs. hard cap.
7. After your edits, call `finalize(summary='...')`.
8. Deduplicate aggressively: merge overlapping or repeated reflections into one broader,
    reusable bullet whenever possible, and avoid adding many near-duplicate bullets that
    express the same lesson.

{chr(10).join(body_parts)}
"""

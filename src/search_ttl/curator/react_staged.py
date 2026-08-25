"""ReActStagedCurator — staged variant of ``react_tactic``.

Pairs with the ``staged_playbook`` backend (FIND / VERIFY / DECIDE) and
the ``querychain_staged`` reflector. Same ReAct loop / ledger / budget
machinery as ``react_tactic``; the targeted differences (all motivated
by ``docs/stage2/analysis/gptoss/*``):

1. Bullets route into FIND / VERIFY / DECIDE (taken from the reflection's
   ``section_hint``; the backend normalizes spellings).
2. ``good_query`` validation is now dense-retriever-correct for the two
   QUERY stages (FIND, VERIFY): reject ANY chain of 2+ quoted phrases
   AND reject boolean ``AND``/``OR`` operators. The old TACTIC_A
   "quoting is the point" exemption is GONE (it was the v2-5 regression
   cause).
3. DECIDE is a DIRECTIVE stage: no ``good_query``/``bad_query``. The
   curator assembles a Rule / Why / Apply-when / Do-NOT bullet from
   dedicated fields, so the highest-value lessons (stop-and-commit,
   anti-fame-drift) survive verbatim instead of being mangled into a
   fake query example.
4. Soft answer-leak guard on QUERY bullets (warns, does not hard-reject,
   to avoid breaking the ReAct loop).
5. Per-section bullet cap to stop the monotone-bloat anti-learning
   (v2-5 grew 3→21 bullets over 25 steps with 0 DELETE; accuracy fell
   as it grew).

Char-budget rollback stays backend-agnostic
(``type(experience).from_text``) so it works with ``staged_playbook``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
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


def _stage_of(section: str) -> str:
    """Normalize any section spelling to FIND / VERIFY / DECIDE — kept
    in sync with ``staged_playbook._normalize_section_name`` so the
    curator's per-stage logic matches where the bullet will land."""
    s = (section or "").strip().lower()
    if (
        "decide" in s or "decision" in s or "commit" in s or "evidence" in s
        or "directive" in s or "anti-drift" in s or "stop" in s or s == "d"
    ):
        return "DECIDE"
    if (
        "verify" in s or "verification" in s or "candidate" in s or "pivot" in s
        or "tactic_c" in s or "tactic c" in s or s in ("v", "c")
    ):
        return "VERIFY"
    return "FIND"


@dataclass
class _Conserv:
    max_add_per_step: int = 5
    max_modify_per_step: int = 3
    max_delete_per_step: int = 2
    max_total_edits_per_step: int = 8


@dataclass
class _OptCfg:
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    max_react_turns: int = 24
    conservativeness: _Conserv = field(default_factory=_Conserv)
    char_budget: int = 48000
    reject_if_over_budget_ratio: float = 1.2
    # Hard per-section bullet cap (0 disables). Stops the monotone-bloat
    # anti-learning observed in v2-5: once a stage is full the curator
    # is forced to MODIFY/DELETE instead of piling on near-duplicates.
    max_bullets_per_section: int = 8
    llm_provider: str = "openai_compat"  # openai_compat | openrouter
    llm_api_flavor: str = "chat_completions"
    enable_thinking: bool = False
    preserve_thinking: bool = True
    temperature: float = 0.2
    top_p: float = 0.95
    max_response_tokens: int = 8192
    llm_timeout_s: float = 600.0


@register_curator("react_staged")
class ReActStagedCurator(CuratorManager):
    def __init__(self, cfg: _OptCfg) -> None:
        self.cfg = cfg

    @classmethod
    def from_omegaconf(cls, cfg) -> "ReActStagedCurator":
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
            max_bullets_per_section=int(cfg.get("max_bullets_per_section", 8)),
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
        pre_text = experience.to_text()
        ledger = _Ledger(
            experience=experience,
            conserv=self.cfg.conservativeness,
            step=step,
            char_budget=self.cfg.char_budget,
            reject_ratio=self.cfg.reject_if_over_budget_ratio,
            max_bullets_per_section=self.cfg.max_bullets_per_section,
        )
        finalize_summary: list[str] = []

        if save_dir is not None:
            write_json(save_dir / "curator/input.json",
                       [_reflection_brief(r) for r in reflections])

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
            """Read all bullets in a section (FIND / VERIFY / DECIDE)."""
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
            """Find bullets containing a keyword (use before ADD to dedup)."""
            bs = ledger.experience.search_bullets(keyword, top_k=top_k)
            return json.dumps(
                [
                    {"bullet_id": b.bullet_id, "section": b.section, "content": b.content}
                    for b in bs
                ]
            )

        # ── assembly + validation helpers ─────────────────────────────────
        def _assemble_query_bullet(
            task: str, pre_search_state: str, good_query: str,
            good_rationale: str, bad_query: str, bad_rationale: str, caveat: str,
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

        def _assemble_directive_bullet(
            rule: str, why: str, apply_when: str, do_not: str,
        ) -> str:
            lines = [f"**Rule**: {rule.strip()}"]
            if why.strip():
                lines.append(f"**Why**: {why.strip()}")
            if apply_when.strip():
                lines.append(f"**Apply when**: {apply_when.strip()}")
            if do_not.strip():
                lines.append(f"**Do NOT**: {do_not.strip()}")
            return "\n".join(lines)

        def _validate_query(stage: str, good_query: str) -> str:
            """Dense-retriever correctness check for FIND / VERIFY. The
            retriever embeds the whole string: quotes are not operators
            and AND/OR are literal tokens, so quoted-shard / boolean
            queries retrieve WORSE. Hard-reject them."""
            gq = good_query or ""
            n_quoted = len(re.findall(r'"[^"]+"', gq))
            if n_quoted >= 2:
                return (
                    f"rejected: `good_query` has {n_quoted} separately quoted "
                    "phrases. The retriever is DENSE — quoting is inert and "
                    "shard chains fragment the embedding. Rewrite as ONE "
                    "coherent natural-language "
                    + ("sentence (FIND) " if stage == "FIND" else
                       "phrase = candidate name + ONE attribute (VERIFY) ")
                    + "with names/numbers as plain words, no quotes."
                )
            if re.search(r"\b(AND|OR)\b", gq):
                return (
                    "rejected: `good_query` uses boolean AND/OR. The dense "
                    "retriever treats them as ordinary words. Remove the "
                    "operators and write a plain coherent query."
                )
            return ""

        def _answer_leak_warn(stage: str, scope: str, good_query: str) -> str:
            """Soft warning (NOT a hard reject — avoids ReAct loop
            breakage) when a strategy-scope query looks like it just
            embeds the answer (mostly capitalized tokens, few relational
            words)."""
            if stage == "VERIFY":
                return ""  # VERIFY is name+attribute by design
            gq = (good_query or "").strip()
            caps = len(re.findall(r"\b[A-Z][a-zA-Z]{2,}", gq))
            words = max(1, len(gq.split()))
            lowers = len(re.findall(r"\b[a-z]{3,}\b", gq))
            if str(scope) == "strategy" and caps >= 4 and lowers < 3 and words <= 10:
                return (
                    " WARNING: this FIND good_query looks like a bare answer "
                    "dump (mostly proper nouns). Rewrite it as the reusable "
                    "MOVE using entity TYPES/placeholders so it transfers to "
                    "different questions."
                )
            return ""

        @tool_dec
        def add_bullet(
            section: str,
            task: str = "",
            pre_search_state: str = "",
            good_query: str = "",
            good_rationale: str = "",
            bad_query: str = "",
            bad_rationale: str = "",
            caveat: str = "",
            rule: str = "",
            why: str = "",
            apply_when: str = "",
            do_not: str = "",
            source_reflection_ids: list[str] | None = None,
        ) -> str:
            """Add a bullet to a stage section.

            ``section`` MUST be one of FIND / VERIFY / DECIDE (from the
            reflection's ``section_hint``).

            - FIND / VERIFY (QUERY stages) — supply the 5 query slots:
              ``task``, ``pre_search_state``, ``good_query`` (+
              ``good_rationale``), ``bad_query`` (+ ``bad_rationale``,
              optional), ``caveat``. ``good_query`` MUST be dense-
              retriever-correct: ONE coherent natural-language query for
              FIND, or `<candidate> <one attribute>` for VERIFY — NO 2+
              quoted phrases, NO AND/OR. Describe the reusable MOVE with
              entity TYPES/placeholders, never the specific answer.

            - DECIDE (DIRECTIVE stage) — supply ``rule`` (the imperative,
              source: reflection ``principle``), ``why`` (source:
              ``what_went_wrong`` / ``verification_evidence``),
              ``apply_when`` (source: ``applies_to_future_questions_when``)
              and ``do_not`` (source: ``anti_pattern``). The query slots
              are ignored for DECIDE.
            """
            stage = _stage_of(section)
            if stage == "DECIDE":
                if not rule.strip():
                    return ("rejected: DECIDE bullets need a `rule` "
                            "(imperative). Map the reflection's `principle` "
                            "→ rule, `anti_pattern` → do_not, "
                            "`applies_to_future_questions_when` → apply_when.")
                content = _assemble_directive_bullet(rule, why, apply_when, do_not)
                return ledger.add(section, content, source_reflection_ids or [])
            # FIND / VERIFY
            if not good_query.strip():
                return ("rejected: FIND/VERIFY bullets need a `good_query`. "
                        "(If this is a decision rule, route it to DECIDE "
                        "with `rule`/`why`/`apply_when`/`do_not`.)")
            err = _validate_query(stage, good_query)
            if err:
                return err
            content = _assemble_query_bullet(
                task, pre_search_state, good_query, good_rationale,
                bad_query, bad_rationale, caveat,
            )
            res = ledger.add(section, content, source_reflection_ids or [])
            return res

        @tool_dec
        def modify_bullet(
            bullet_id: str,
            task: str = "",
            pre_search_state: str = "",
            good_query: str = "",
            good_rationale: str = "",
            bad_query: str = "",
            bad_rationale: str = "",
            caveat: str = "",
            rule: str = "",
            why: str = "",
            apply_when: str = "",
            do_not: str = "",
            source_reflection_ids: list[str] | None = None,
        ) -> str:
            """Replace an existing bullet's content. Same field semantics
            as ``add_bullet``; the stage is taken from the existing
            bullet's section. Call ``read_bullet`` first."""
            existing = ledger.experience.read_bullet(bullet_id)
            stage = _stage_of(existing.section if existing is not None else "")
            if stage == "DECIDE":
                if not rule.strip():
                    return ("rejected: DECIDE bullets need a `rule`.")
                content = _assemble_directive_bullet(rule, why, apply_when, do_not)
                return ledger.modify(bullet_id, content, source_reflection_ids or [])
            if not good_query.strip():
                return "rejected: FIND/VERIFY bullets need a `good_query`."
            err = _validate_query(stage, good_query)
            if err:
                return err
            content = _assemble_query_bullet(
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
            """Delete a bullet (only with justification). Use this to keep
            each stage small and free of near-duplicates."""
            return ledger.delete(bullet_id, justification, source_reflection_ids or [])

        @tool_dec
        def remaining_budget() -> str:
            """ADD/MODIFY/DELETE left + per-section bullet counts vs cap."""
            return json.dumps(ledger.remaining())

        @tool_dec
        def finalize(summary: str) -> str:
            """End optimization for this step."""
            finalize_summary.append(summary)
            return "finalized"

        tools = [
            list_sections, read_section, read_bullet, search_bullets,
            add_bullet, modify_bullet, delete_bullet, remaining_budget, finalize,
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

        sys_msg = "You are the ExperienceCurator. Apply the reflections to the staged playbook."
        user_msg = _render_curator_user_msg(
            reflections=reflections,
            conserv=self.cfg.conservativeness,
            max_react_turns=self.cfg.max_react_turns,
            char_budget=self.cfg.char_budget,
            max_bullets_per_section=self.cfg.max_bullets_per_section,
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
                LOG.warning("staged curator LLM error step=%s: %s", step, exc)
                break
            from search_ttl.utils.tool_call_recovery import (
                recover_mcp_call_tool_calls,
                sanitize_responses_message_for_round_trip,
            )
            mcp_recovered = recover_mcp_call_tool_calls(ai)
            if mcp_recovered:
                LOG.info("staged curator step=%s: promoted %d mcp_call block(s)",
                         step, len(mcp_recovered))
            n_stripped = sanitize_responses_message_for_round_trip(ai)
            if n_stripped:
                LOG.info("staged curator step=%s: stripped %d vLLM-rejected block(s)",
                         step, n_stripped)
            messages.append(ai)

            tcs = getattr(ai, "tool_calls", None) or []
            if not tcs and isinstance(ai.content, str):
                from search_ttl.utils.tool_call_recovery import recover_text_tool_calls

                recovered = recover_text_tool_calls(ai.content)
                if recovered:
                    LOG.info("staged curator step=%s: recovered %d tool_call(s) from text",
                             step, len(recovered))
                    ai.tool_calls = recovered
                    tcs = recovered

            if not tcs:
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
                    break

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
            ledger.experience = type(experience).from_text(pre_text)
            ledger.ops = []

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
            sid for op in ledger.ops for sid in (op.source_reflection_ids or [])
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


# ── ledger ───────────────────────────────────────────────────────────────


@dataclass
class _Ledger:
    experience: Experience
    conserv: _Conserv
    step: int
    ops: list[OpRecord] = field(default_factory=list)
    char_budget: int = 0
    reject_ratio: float = 1.2
    max_bullets_per_section: int = 0

    @property
    def _hard_cap(self) -> int:
        return int(self.char_budget * self.reject_ratio)

    def _char_check(self, projected: int, op: str) -> tuple[str, str]:
        if self.char_budget <= 0:
            return "ok", ""
        hard = self._hard_cap
        if projected > hard:
            return "error", (
                f"refused: applying this {op} would push the playbook to "
                f"~{projected} chars, over the hard cap of {hard}. The "
                "playbook is saturated. DO NOT add more bullets — instead "
                "`search_bullets`/`read_section` to find near-duplicates, "
                "`delete_bullet` the weakest, and `modify_bullet` to absorb "
                "the new insight. Re-check with `remaining_budget`."
            )
        if projected > self.char_budget:
            return "warn", (
                f"playbook would grow to ~{projected} chars, past the soft "
                f"budget of {self.char_budget} (hard cap {hard}). Applied, "
                "but switch to consolidation: prefer `modify_bullet`, delete "
                "duplicates; further ADDs are refused at the hard cap."
            )
        return "ok", ""

    def remaining(self) -> dict[str, int]:
        used = self._counts()
        out: dict[str, Any] = {
            "add_left": max(0, self.conserv.max_add_per_step - used["ADD"]),
            "modify_left": max(0, self.conserv.max_modify_per_step - used["MODIFY"]),
            "delete_left": max(0, self.conserv.max_delete_per_step - used["DELETE"]),
            "total_left": max(
                0, self.conserv.max_total_edits_per_step - sum(used.values())
            ),
        }
        secs = {s.name: s.bullet_count for s in self.experience.list_sections()}
        out["bullets_per_section"] = secs
        if self.max_bullets_per_section > 0:
            out["max_bullets_per_section"] = self.max_bullets_per_section
            out["sections_at_cap"] = [
                k for k, v in secs.items() if v >= self.max_bullets_per_section
            ]
        if self.char_budget > 0:
            current = self.experience.char_count()
            out["playbook_chars_current"] = current
            out["playbook_chars_soft_budget"] = self.char_budget
            out["playbook_chars_hard_cap"] = self._hard_cap
            out["playbook_chars_until_refused"] = self._hard_cap - current
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

    def _section_count(self, section: str) -> int:
        return len(self.experience.read_section(section))

    def add(self, section: str, content: str, srcs: list[str]) -> str:
        ok, why = self._check("ADD")
        if not ok:
            return f"refused: {why}"
        if (
            self.max_bullets_per_section > 0
            and self._section_count(section) >= self.max_bullets_per_section
        ):
            return (
                f"refused: section already has "
                f"{self._section_count(section)} bullets (cap "
                f"{self.max_bullets_per_section}). Keep each stage small: "
                "`delete_bullet` the weakest/most-duplicated bullet here, or "
                "`modify_bullet` an existing one to absorb this insight, "
                "instead of adding another near-duplicate."
            )
        new_exp, bullet = self.experience.add_bullet(section, content)
        kind, msg = self._char_check(new_exp.char_count(), op="ADD")
        if kind == "error":
            return msg
        self.experience = new_exp
        self.ops.append(
            OpRecord(
                step=self.step, op="ADD", section=bullet.section,
                bullet_id_before=None, bullet_id_after=bullet.bullet_id,
                content_before=None, content_after=bullet.content,
                source_reflection_ids=srcs,
            )
        )
        result: dict[str, Any] = {
            "bullet_id": bullet.bullet_id, "section": bullet.section
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
        new_exp, after = self.experience.modify_bullet(bullet_id, new_content)
        kind, msg = self._char_check(new_exp.char_count(), op="MODIFY")
        if kind == "error":
            return msg
        self.experience = new_exp
        self.ops.append(
            OpRecord(
                step=self.step, op="MODIFY", section=before.section,
                bullet_id_before=before.bullet_id, bullet_id_after=before.bullet_id,
                content_before=before.content,
                content_after=after.content if after else new_content,
                source_reflection_ids=srcs,
            )
        )
        result: dict[str, Any] = {
            "bullet_id": bullet_id, "section": before.section
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
                step=self.step, op="DELETE", section=before.section,
                bullet_id_before=before.bullet_id, bullet_id_after=None,
                content_before=before.content, content_after=None,
                source_reflection_ids=srcs, justification=justification,
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
    max_bullets_per_section: int,
) -> str:
    ok = [r for r in reflections if r.status == "ok"]
    skip = [r for r in reflections if r.status != "ok"]
    body_parts = ["# Reflections", "", f"## OK reflections (n={len(ok)})", ""]
    for r in ok:
        body_parts.append(r.text_for_curator)
        body_parts.append("---")
    if skip:
        body_parts.append(f"## Skipped (n={len(skip)})")
        for r in skip:
            body_parts.append(f"- qid={r.qid} r_idx={r.r_idx} status={r.status}")

    cap_line = (
        f"- Per-section bullet cap: {max_bullets_per_section} "
        "(at the cap, ADD is refused — MODIFY/DELETE instead)"
        if max_bullets_per_section > 0 else
        "- Per-section bullet cap: (disabled)"
    )

    return f"""\
You are the ExperienceCurator for a 3-STAGE playbook. Apply the
reflections below.

# Stages (route every bullet by the reflection's `section_hint`)

The retriever is DENSE (a vector model): quotes are NOT operators,
`AND`/`OR` are ordinary words, quoted-shard chains retrieve WORSE.

- **FIND** — locate the entity/document from clues (no candidate yet).
  QUERY bullet. `good_query` = ONE coherent natural-language sentence,
  proper nouns/numbers inline, NO quotes, NO AND/OR. (rejected otherwise)
- **VERIFY** — a candidate / bridge entity has appeared. QUERY bullet.
  `good_query` = `<candidate or bridge entity> <exactly ONE attribute>`,
  plain words. (same quote/boolean rejection)
- **DECIDE** — a DECISION RULE, not a query. DIRECTIVE bullet: fill
  `rule` (imperative; from reflection `principle`), `why` (from
  `what_went_wrong`/`verification_evidence`), `apply_when` (from
  `applies_to_future_questions_when`), `do_not` (from `anti_pattern`).
  Do NOT put a query in DECIDE.

Bullets must teach the reusable MOVE with entity TYPES / placeholders,
never the specific answer (answer-dump `good_query` is warned/rejected).

# Per-step budget
- ADD ≤ {conserv.max_add_per_step}, MODIFY ≤ {conserv.max_modify_per_step}, DELETE ≤ {conserv.max_delete_per_step}, TOTAL ≤ {conserv.max_total_edits_per_step}
- React turns ≤ {max_react_turns}; final playbook char cap {char_budget}
{cap_line}

# How to work
1. `list_sections` / `read_section` / `search_bullets` FIRST to ground
   and find near-duplicates.
2. STRONGLY prefer `modify_bullet` over `add_bullet` when an existing
   bullet in the SAME stage covers the same trigger — merge, don't pile
   up near-duplicates. Each stage should stay small and high-signal.
3. Skip reflections with status != 'ok' OR confidence < 0.4.
4. When a stage is at its bullet cap, `delete_bullet` the weakest /
   most redundant bullet there before adding, or just `modify_bullet`.
5. Read the `add_bullet` docstring for which fields each stage needs.
6. After edits, call `finalize(summary='...')`.
7. Deduplicate aggressively: one broad reusable bullet beats many
   near-duplicates; small playbooks evaluated better than large ones.

{chr(10).join(body_parts)}
"""

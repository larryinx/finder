"""ACE Generator that delegates to the HiExp ReAct loop.

ACE's :class:`ace.core.Generator.generate` wants:

    (response_text, bullet_ids, call_info)

where ``response_text`` parses to a JSON ``{final_answer, ...}`` via
``ace.utils.extract_answer``. The rest of ACE only needs
``extract_answer(response_text) → final_answer`` and ``bullet_ids`` for
reflector tagging; the ``call_info`` is just logged.

This module produces all three by:

1. running a full ReAct cycle through :class:`ace_react.ReActAgent` with
   the playbook injected into the system prompt;
2. extracting the BCP-style ``Exact Answer:`` block from the final
   ``output_text`` item via :func:`hiexp.evaluation.extract_exact_answer`;
3. wrapping that exact-answer string in a small JSON envelope so
   ``ace.utils.extract_answer(response_text) → exact_answer`` works
   without changing ACE;
4. scanning the trajectory for ``[bullet-id]`` references in reasoning
   and final-text blocks (best-effort — ReAct doesn't explicitly
   declare which bullets it used);
5. persisting a BCP-shape ``run_<qid>.json`` to disk if a runs
   directory was configured.

The qid-needed-to-name-the-file gap is bridged by a question→qid
registry the generator is constructed with (questions are unique within
a BCP split, and we know all questions up front).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import weave

from hiexp import HiExpConfig
from hiexp.evaluation.answer_extract import extract_exact_answer

from ace_react import ReActAgent
from ace_react.bcp.persist import trajectory_to_bcp_record, write_bcp_run_json


# Bullet IDs in the playbook look like ``[shr-00009]`` / ``[search-00012]`` —
# letters then a hyphen then digits. We intentionally restrict letters to
# ASCII a-z to avoid catching things like ``[CITATION-99]`` in retrieved
# docs, which aren't ACE bullet ids.
_BULLET_ID_RE = re.compile(r"\[([a-z]+-\d+)\]")


class ReActGenerator:
    """Drop-in replacement for ``ace.core.Generator`` with ReAct semantics.

    Args:
        base_config: HiExpConfig (HEK auto-disabled inside :class:`ReActAgent`).
        runs_dir: optional. When set, every successful ``generate()`` call
            writes ``run_<qid>.json`` under this directory in BCP-shape.
            Pass ``None`` to skip persistence (e.g. during ACE training
            where only the final eval pass needs saved runs).
        task_registry: maps ``question_text → task_dict`` so we can look
            up ``query_id`` per call — ACE's Generator interface doesn't
            pass it through. Build once with
            ``{td["question"]: td for td in samples}``. Falls back to
            an MD5-derived id if a question isn't in the registry.
        run_metadata_extra: dict folded into ``metadata.ace`` of every
            persisted record (useful for tagging with mode / epoch / step).
    """

    def __init__(
        self,
        *,
        base_config: HiExpConfig,
        runs_dir: Path | None = None,
        task_registry: dict[str, dict] | None = None,
        run_metadata_extra: dict | None = None,
    ) -> None:
        self._base_config = base_config
        self.model = base_config.llm_model
        self.max_tokens = base_config.max_response_tokens
        self._runs_dir = Path(runs_dir) if runs_dir is not None else None
        self._registry = dict(task_registry or {})
        self._run_metadata_extra = dict(run_metadata_extra or {})
        # Phase bucket — when set, persisted run_<qid>.json files go under
        # ``runs_dir/<phase>/`` instead of ``runs_dir/``. Set by the run.py
        # orchestrator before each test/training pass so initial-test results
        # don't get clobbered by per-step training generations on the same qid.
        self._phase: str | None = None

    def set_phase(self, phase: str | None) -> None:
        """Change the subdir runs are persisted under (None = top-level)."""
        self._phase = phase

    def _phase_for_call(self, call_id: str) -> str | None:
        """Auto-derive a phase from call_id when none was set externally.

        - ``test_eval_*``  : eval pass (initial / final / window pre-test)
        - ``*_gen_initial``: first generate of a training step → ``training``
        - other            : ``other`` (reflector/curator regenerations)
        """
        if self._phase is not None:
            return self._phase
        if call_id.endswith("_gen_initial"):
            return "training"
        if call_id.startswith("test_eval_"):
            return "eval"
        return None

    # ── ACE-facing API ───────────────────────────────────────────────────

    @weave.op(name="ace.ReActGenerator.generate")
    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: str | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Run a full ReAct cycle; return ACE-compatible tuple.

        Note: ``use_json_mode`` is ignored. Our ReAct generator always
        produces BCP-format final text (``Explanation / Exact Answer /
        Confidence``); we wrap the extracted answer in a JSON envelope
        only because ACE's ``extract_answer`` looks for ``final_answer``.
        """
        composite_playbook = _compose_playbook(playbook, reflection)
        agent = ReActAgent(self._base_config, playbook=composite_playbook)

        # A single sample's ReAct call can fail in many ways — LLM 4xx
        # (context overflow, rate limit), MCP transient disconnect,
        # JSON parse, etc. ACE's training loop (third_party/ace/ace/
        # ace.py:_train_single_sample) has NO try/except around
        # generator.generate, so any of these would otherwise kill the
        # entire run. The eval path already has this robustness
        # (third_party/ace/utils.py:172-199 wraps generator.generate
        # per sample); we add the matching protection here on the
        # adapter side so ACE upstream stays untouched.
        #
        # On failure we return a sentinel envelope whose final_answer
        # matches upstream's own no-answer sentinel ("No final answer
        # found", see utils.py:131), so extract_answer /
        # answer_is_correct / reflector / curator all run unchanged —
        # ACE treats the sample as a wrong answer and advances.
        #
        # Catching ``Exception`` alone is not enough: a TaskGroup that
        # leaks a ``CancelledError`` (e.g. langchain-mcp-adapters' SSE
        # transport dropping mid-rollout) raises a ``BaseExceptionGroup``,
        # which is NOT an ``Exception`` — it would sail through and kill
        # the whole ACE run. Catch groups too, but re-raise any that wrap
        # a Ctrl-C / exit so interrupts still work; bare KeyboardInterrupt
        # / CancelledError keep propagating as before.
        try:
            raw_response, traj = asyncio.run(agent.arun(question))
        except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001 — soft-fail per sample
            if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                if exc.split((KeyboardInterrupt, SystemExit))[0] is not None:
                    raise
                inner: BaseException = exc.exceptions[0]
            else:
                inner = exc
            err_type, err_msg = type(inner).__name__, str(inner)
            # A vLLM 400 echoes the *entire* request body (playbook +
            # prior reflection + full message list). Embedding that
            # verbatim into the reflector's reasoning_trace recursively
            # compounds each round until the reflector prompt blows past
            # the model context and the whole ACE run dies. Cap it.
            err_msg = err_msg if len(err_msg) <= 800 else err_msg[:800] + " …[truncated]"
            print(
                f"[generate] {call_id} failed ({err_type}: {err_msg}); "
                "returning sentinel so training can advance"
            )
            qid_local = str(
                self._registry.get(question, {}).get("query_id")
                or _hash_qid(question)
            )
            envelope = json.dumps(
                {
                    "reasoning": f"(generation failed: {err_type}: {err_msg})",
                    "bullet_ids": [],
                    "final_answer": "No final answer found",
                    "raw_bcp_response": "",
                },
                ensure_ascii=False,
            )
            return envelope, [], {
                "qid": qid_local,
                "call_id": call_id,
                "error": f"{err_type}: {err_msg}",
                "status": "failed",
            }
        exact = extract_exact_answer(raw_response)
        bullet_ids = _extract_bullet_ids(raw_response, traj)

        # Persist BCP-shape JSON if configured. Skip noisy intermediate
        # calls (reflector retries, post-curate regenerations) — we only
        # want the answer that ACE's controller actually scores:
        #   * test_eval_<i>  — eval pass (initial / final / online-eval)
        #   * <step>_gen_initial — first generation of a training step;
        #     this is the "online accuracy" answer in online mode.
        task = self._registry.get(question, {})
        qid = str(task.get("query_id") or _hash_qid(question))
        should_persist = self._runs_dir is not None and (
            call_id.startswith("test_eval_") or call_id.endswith("_gen_initial")
        )
        if should_persist:
            phase = self._phase_for_call(call_id)
            ace_meta = {
                "call_id": call_id,
                "playbook_chars": agent.playbook_chars,
                "phase": phase,
                **self._run_metadata_extra,
            }
            record = trajectory_to_bcp_record(
                query_id=qid,
                question=question,
                raw_response=raw_response,
                trajectory=traj,
                model_name=self._base_config.llm_model,
                ace_metadata=ace_meta,
            )
            # Overlay our own scoring sidecar so HiExp's run_judge.py and
            # BCP's evaluate_with_openai.py both Just Work.
            record["prediction_raw"] = raw_response
            record["prediction_exact"] = exact
            record["benchmark"] = "browsecomp_plus"
            # Bucket by phase so initial-test runs don't get overwritten by
            # window-pre-test or per-step training runs on the same qid.
            target_dir = (
                self._runs_dir / phase if phase is not None else self._runs_dir
            )
            write_bcp_run_json(target_dir, record)

        # ACE expects a JSON-parseable response_text whose ``final_answer``
        # key is what ``extract_answer`` returns.
        envelope = json.dumps(
            {
                "reasoning": "(see trajectory in metadata)",
                "bullet_ids": bullet_ids,
                "final_answer": exact,
                "raw_bcp_response": raw_response,
            },
            ensure_ascii=False,
        )
        call_info = {
            "qid": qid,
            "call_id": call_id,
            "tool_call_counts": traj.tool_call_counts,
            "status": traj.status,
        }
        return envelope, bullet_ids, call_info

    # ── runtime knobs ────────────────────────────────────────────────────

    def update_run_metadata(self, **kw) -> None:
        """Merge into ``metadata.ace`` for subsequent ``generate()`` calls.

        Used to tag persisted runs with mode-specific context (e.g. set
        ``mode='offline_train'`` during training, ``mode='offline_eval'``
        when running the final test pass).
        """
        self._run_metadata_extra.update(kw)

    def set_runs_dir(self, runs_dir: Path | None) -> None:
        self._runs_dir = Path(runs_dir) if runs_dir is not None else None


# ── helpers ──────────────────────────────────────────────────────────────


def _compose_playbook(playbook: str, reflection: str) -> str:
    """Tack the most recent reflection onto the playbook for the next turn.

    Empty / default reflection ("(empty)") is dropped so we don't pollute
    the system prompt with a literal "(empty)" string.
    """
    if reflection and reflection.strip() and reflection.strip() != "(empty)":
        return f"{playbook}\n\n## RECENT REFLECTION\n{reflection.strip()}\n"
    return playbook


def _extract_bullet_ids(raw_response: str, traj) -> list[str]:
    """Best-effort: scan final text + reasoning items for [id-####] mentions."""
    blob = raw_response
    for it in getattr(traj, "items", None) or []:
        if it.type == "reasoning":
            outs = it.output if isinstance(it.output, list) else [it.output]
            blob += "\n" + "\n".join(str(o) for o in outs if o)
    return list({m.group(1) for m in _BULLET_ID_RE.finditer(blob)})


def _hash_qid(question: str) -> str:
    """Stable 8-hex-char id when a question isn't in our registry.

    Should be rare (we always pass a registry built from the loaded
    splits), but we don't want filename collisions if it does happen.
    """
    import hashlib

    return hashlib.md5(question.encode("utf-8")).hexdigest()[:8]


__all__ = ["ReActGenerator"]

"""LLM judge (BCP grader template) reused for TTL.

We import the grader prompt + parser from HiExp's evaluation module so
all three pipelines (HiExp, ACE, TTL) share one judgement contract.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from search_ttl.reward.schemas import JudgeResult
from search_ttl.utils.tracing import traceable

LOG = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    model: str
    base_url: str
    api_key: str
    template: str = "bcp_grader"
    # LLM sampling knobs. Defaults match the historical hardcoded values
    # in `_llm_lazy` — temperature=0.0 / top_p=1.0 / enable_thinking=False
    # are deliberate for grading determinism. Override via yaml only if
    # you know what you're doing.
    max_response_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False
    # ``openai_compat`` (default) or ``openrouter`` — mirrors the agent /
    # reflector / curator provider knob so the judge can run against
    # OpenRouter when the rest of the pipeline does. One-shot grader
    # call, so ``preserve_thinking`` / ``llm_api_flavor`` don't matter.
    llm_provider: str = "openai_compat"


class Judge:
    """Async LLM-as-judge over a single (question, prediction, target) tuple."""

    def __init__(self, cfg: JudgeConfig) -> None:
        self.cfg = cfg
        self._llm = None

    def _llm_lazy(self):
        if self._llm is not None:
            return self._llm
        # Use the shared builder so the judge picks up the same provider
        # (openai_compat / openrouter) as the rest of the pipeline.
        from search_ttl.utils.llm_builder import build_chat_llm

        self._llm = build_chat_llm(
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key or "EMPTY",
            llm_provider=self.cfg.llm_provider,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_response_tokens,
            enable_thinking=self.cfg.enable_thinking,
            # One-shot grading call — there is no prior turn to inject
            # reasoning from, so preserve_thinking is moot here.
            preserve_thinking=False,
        )
        return self._llm

    @traceable(name="Judge.ajudge", run_type="chain")
    async def ajudge(self, question: str, prediction: str, target: str) -> JudgeResult:
        # Fast path — exact match (case-insensitive).
        if (prediction or "").strip().lower() == (target or "").strip().lower() and target:
            return JudgeResult(
                correct=True,
                extracted_answer=prediction.strip(),
                reasoning="exact_match",
                confidence=100.0,
                parse_error=False,
                raw_response="(exact-match shortcut)",
            )
        from hiexp.evaluation.judge import (  # type: ignore[import-not-found]
            render_grader_prompt,
            parse_judge_response,
        )
        from hiexp_prompt._llm_utils import achat_once  # type: ignore[import-not-found]

        prompt = render_grader_prompt(
            question=question, response=prediction or "", correct_answer=target or ""
        )
        try:
            text = await achat_once(self._llm_lazy(), prompt)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("judge LLM error: %s", exc)
            return JudgeResult(
                correct=False,
                extracted_answer=None,
                reasoning=f"[judge_error] {type(exc).__name__}: {exc}",
                confidence=0.0,
                parse_error=True,
                raw_response="",
            )
        parsed = parse_judge_response(text)
        return JudgeResult(
            correct=bool(parsed.get("correct")),
            extracted_answer=parsed.get("extracted_final_answer"),
            reasoning=parsed.get("reasoning"),
            confidence=float(parsed.get("confidence") or 0.0),
            parse_error=bool(parsed.get("parse_error")),
            raw_response=text,
        )

    @classmethod
    def from_env(cls) -> "Judge":
        """Convenience: build a judge from env vars matching ace_react.DataProcessor."""
        return cls(
            JudgeConfig(
                model=os.environ.get("EVAL_JUDGE_MODEL")
                or os.environ.get("AGENT_LLM_MODEL", ""),
                base_url=os.environ.get("EVAL_JUDGE_BASE_URL")
                or os.environ.get("AGENT_LLM_BASE_URL", ""),
                api_key=os.environ.get("EVAL_JUDGE_API_KEY")
                or os.environ.get("AGENT_LLM_API_KEY", ""),
            )
        )

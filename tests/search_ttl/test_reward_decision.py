"""Rule-based efficiency stats of the judge-only reward (no LLM)."""

from __future__ import annotations

from search_ttl.reward.judge import JudgeConfig
from search_ttl.reward.search_agent_lasj import SearchAgentRewardLASJ, _RewardCfg
from search_ttl.rollout.schemas import QueryRecord, Trajectory


def _mgr() -> SearchAgentRewardLASJ:
    return SearchAgentRewardLASJ(
        _RewardCfg(judge=JudgeConfig(model="", base_url="", api_key=""))
    )


def _traj(status: str = "completed", queries: list[QueryRecord] | None = None) -> Trajectory:
    return Trajectory(
        qid="q",
        question="?",
        target="t",
        experience_snapshot="",
        experience_hash="h",
        status=status,
        retrieved_docids=["a", "b"],
        queries=queries or [],
        metrics={"turns": 4, "search_count": len(queries or [])},
        tool_call_counts={"search": len(queries or [])},
        extracted_answer="x",
    )


def test_efficiency_counts_searches_and_turns():
    qs = [
        QueryRecord(idx=i, query=f"q{i}", preceding_thought="", n_results=1, top_doc_ids=["d"])
        for i in range(3)
    ]
    eff = _mgr()._compute_efficiency(_traj(queries=qs))
    assert eff.n_search == 3
    assert eff.n_turn == 4
    assert not eff.hit_max_search and not eff.hit_max_turn


def test_efficiency_flags_max_search_and_max_turn_status():
    mgr = _mgr()
    assert mgr._compute_efficiency(_traj(status="max_search")).hit_max_search
    assert mgr._compute_efficiency(_traj(status="max_turn")).hit_max_turn


def test_efficiency_redundant_query_count():
    qs = [
        QueryRecord(idx=0, query="Foo BAR", preceding_thought="", n_results=3, top_doc_ids=["d1"]),
        QueryRecord(idx=1, query="foo bar", preceding_thought="", n_results=3, top_doc_ids=["d1"]),
        QueryRecord(idx=2, query="baz", preceding_thought="", n_results=2, top_doc_ids=["d2"]),
    ]
    eff = _mgr()._compute_efficiency(_traj(queries=qs))
    assert eff.redundant_query_count == 1

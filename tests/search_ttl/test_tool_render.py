"""Regression tests for ``utils/tool_render.py``.

These pin the contracts:
  - ``tool_result_visibility`` switches the rendering between
    none / title / clipped / full.
  - BCP MCP hits carry their title in the snippet's front matter; the
    title / clipped modes extract it and the clipped body strips it.
  - ``describe_visibility_for_prompt`` produces the visibility-aware
    preamble so the secondary LLM knows what shape of data it's
    looking at.
"""

from __future__ import annotations

from search_ttl.utils.tool_render import (
    describe_visibility_for_prompt,
    render_tool_result,
)


# ── tool_render direct -------------------------------------------------


_HITS = [
    {"docid": "10", "title": "Foo paper", "snippet": "alpha " * 50},
    {"docid": "11", "title": "Bar paper", "snippet": "beta " * 50},
]


# Real BCP MCP shape: title field empty, title encoded in snippet front matter.
_BCP_HITS = [
    {
        "docid": "77234",
        "score": 0.91,
        "snippet": (
            "---\n"
            "title: Romanian Statistical Review 2014 Issue 3\n"
            "author: National Institute of Statistics\n"
            "date: 2014-03-15\n"
            "---\n"
            "Body content goes here."
        ),
    },
    {
        "docid": "12345",
        "score": 0.42,
        "snippet": "---\ntitle: Other paper\n---\nbody",
    },
]


def test_render_none_emits_only_count_and_ids():
    out = render_tool_result(_HITS, visibility="none")
    assert "alpha" not in out
    assert "Foo paper" not in out
    assert "['10', '11']" in out
    assert "2 docs" in out


def test_render_title_emits_docid_and_title_only():
    out = render_tool_result(_HITS, visibility="title")
    assert "docid=10" in out and "docid=11" in out
    assert "Foo paper" in out and "Bar paper" in out
    # No snippet body.
    assert "alpha alpha" not in out
    assert "beta beta" not in out


def test_render_title_extracts_from_snippet_front_matter():
    """Real BCP hits have an empty `title` field; the title lives in
    a YAML-style front matter at the head of `snippet`."""
    out = render_tool_result(_BCP_HITS, visibility="title")
    assert "Romanian Statistical Review 2014 Issue 3" in out
    assert "Other paper" in out
    # The body content must NOT leak into the title-only render.
    assert "Body content" not in out


def test_render_clipped_includes_truncated_snippet():
    out = render_tool_result(_HITS, visibility="clipped", max_chars_total=10_000)
    assert "docid=10" in out and "docid=11" in out
    assert "Foo paper" in out


def test_render_clipped_strips_front_matter_so_body_shows():
    """Clipped mode shows snippet body — but should drop the front
    matter so we don't repeat the title twice in the rendered block."""
    out = render_tool_result(_BCP_HITS, visibility="clipped", max_chars_total=10_000)
    # Title appears in the header line.
    assert "Romanian Statistical Review 2014 Issue 3" in out
    # Body content shows up.
    assert "Body content" in out
    # The literal `---` front matter delimiter shouldn't lead the snippet.
    first_block = out.split("\n\n")[0]
    snippet_part = first_block.split("\n", 1)[1]
    assert not snippet_part.lstrip().startswith("---")


def test_render_clipped_respects_total_budget():
    """When max_chars_total is small, later hits get dropped with a marker."""
    out = render_tool_result(_HITS, visibility="clipped", max_chars_total=120)
    assert "more)" in out  # the "... (N more)" tail


def test_render_full_returns_json_list():
    out = render_tool_result(_HITS, visibility="full")
    assert out.startswith("[") and out.endswith("]")
    assert '"docid"' in out
    assert "Foo paper" in out


def test_render_unwraps_strings_into_text_dict():
    out = render_tool_result("hello world", visibility="clipped", max_chars_total=200)
    assert "hello world" in out


def test_describe_visibility_for_prompt_mentions_mode():
    s = describe_visibility_for_prompt("title", max_chars_total=1500)
    assert "title" in s.lower() or "titles only" in s.lower()


def test_describe_visibility_for_prompt_optional_reasoning_truncation():
    """When ``reasoning_chars_per_step`` is provided (used by the
    reflector), the explainer mentions the truncation. When it's None
    (used by the analyzer, which does NOT truncate), it doesn't."""
    with_trunc = describe_visibility_for_prompt(
        "title", max_chars_total=1500, reasoning_chars_per_step=400
    )
    without = describe_visibility_for_prompt("title", max_chars_total=1500)
    assert "400" in with_trunc and "truncated" in with_trunc
    assert "truncated" not in without

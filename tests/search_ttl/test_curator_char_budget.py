"""Per-call char-budget guards in ReActStagedCurator's ``_Ledger``.

Behavior pinned:

  - ``char_budget <= 0`` disables the guard (backwards compatible).
  - Edit ≤ ``char_budget``: applies silently.
  - Edit between soft budget and ``char_budget × reject_ratio``: applies
    with a ``warning`` field in the response telling the LLM to start
    consolidating.
  - Edit over the hard cap: refused BEFORE being applied (no
    ledger.experience mutation, no entry appended to ledger.ops); the
    response starts with ``refused:`` and tells the LLM to delete +
    merge duplicates.
  - ``remaining()`` exposes the current playbook size + thresholds when
    the guard is enabled, and omits them when disabled.
  - ``delete_bullet`` is never blocked by the char guard (it only
    frees space).

Sizing note: ``char_count()`` is the length of the rendered playbook,
and an empty ``StagedPlaybook`` already contains the fixed section
headers + preambles (``_BASELINE`` chars, measured below). Budgets in
this file are expressed relative to that baseline: the **empty**
playbook must sit under the soft budget, otherwise every ADD would be
refused unconditionally and we couldn't differentiate the cases. A
400-char bullet adds a little over 400 chars (content + id line).
"""

from __future__ import annotations

import json

from omegaconf import OmegaConf

from search_ttl.curator.react_staged import _Conserv, _Ledger
from search_ttl.experience.staged_playbook import StagedPlaybook


def _empty_playbook() -> StagedPlaybook:
    return StagedPlaybook.from_omegaconf(OmegaConf.create({"init_from": None}))


_BASELINE = _empty_playbook().char_count()  # header / preamble overhead
_SECTION = _empty_playbook().list_sections()[0].name


def _ledger(*, char_budget: int = 0, reject_ratio: float = 1.2) -> _Ledger:
    return _Ledger(
        experience=_empty_playbook(),
        conserv=_Conserv(
            max_add_per_step=99, max_modify_per_step=99,
            max_delete_per_step=99, max_total_edits_per_step=999,
        ),
        step=1,
        char_budget=char_budget,
        reject_ratio=reject_ratio,
    )


def test_empty_playbook_baseline_fits_the_budgets_used_here():
    """The budgets below assume the empty playbook is > 160 chars (so a
    hard cap of 160 already refuses) and < 3000 chars (so one 400-char
    bullet crosses the caps). If this fails, the fixed preambles
    changed — revisit the budgets in this file."""
    assert 160 < _BASELINE < 3000


def test_char_count_is_rendered_length():
    pb = _empty_playbook()
    assert pb.char_count() == len(pb.render())


# ── disabled-guard backwards compatibility ───────────────────────────────


def test_char_check_disabled_when_budget_zero():
    """``char_budget=0`` ⇒ no char guard. All adds succeed, no warnings."""
    led = _ledger(char_budget=0)
    for i in range(20):
        out = led.add(_SECTION, f"bullet number {i}" * 5, srcs=[])
        parsed = json.loads(out)
        assert "bullet_id" in parsed
        assert "warning" not in parsed
    assert len(led.ops) == 20


# ── soft warning, edit still applied ─────────────────────────────────────


def test_add_over_soft_budget_warns_but_applies():
    """Push past the soft budget but stay under the hard cap (× 2.0):
    edit lands AND ``warning`` field is in the response."""
    soft = _BASELINE + 160
    led = _ledger(char_budget=soft, reject_ratio=2.0)
    out = led.add(_SECTION, "x" * 400, srcs=[])
    parsed = json.loads(out)
    assert "bullet_id" in parsed
    assert len(led.ops) == 1
    assert "warning" in parsed
    assert "soft budget" in parsed["warning"].lower()
    assert led.experience.char_count() > soft
    assert led.experience.char_count() <= soft * 2


def test_modify_over_soft_budget_warns_but_applies():
    soft = _BASELINE + 160
    led = _ledger(char_budget=soft, reject_ratio=2.0)
    first = led.add(_SECTION, "small", srcs=[])
    bullet_id = json.loads(first)["bullet_id"]
    assert led.experience.char_count() <= soft  # seed didn't trip warning

    # MODIFY to 400 chars → over soft, under hard.
    out = led.modify(bullet_id, "y" * 400, srcs=[])
    parsed = json.loads(out)
    assert parsed.get("bullet_id") == bullet_id
    assert "warning" in parsed
    assert "soft budget" in parsed["warning"].lower()
    assert led.experience.char_count() > soft
    assert led.experience.char_count() <= soft * 2


# ── hard cap, edit refused without mutation ──────────────────────────────


def test_add_over_hard_cap_refused_without_mutation():
    led = _ledger(char_budget=_BASELINE + 36, reject_ratio=1.1)
    pre_chars = led.experience.char_count()
    pre_ops = list(led.ops)
    pre_text = led.experience.to_text()

    out = led.add(_SECTION, "x" * 400, srcs=[])

    assert out.startswith("refused:")
    # Tells LLM how to recover.
    assert "delete" in out.lower() and "modify" in out.lower()
    assert "hard cap" in out.lower()
    # No mutation: experience pointer, char count, and ops list all
    # unchanged.
    assert led.experience.char_count() == pre_chars
    assert led.experience.to_text() == pre_text
    assert led.ops == pre_ops


def test_modify_over_hard_cap_refused_without_mutation():
    led = _ledger(char_budget=_BASELINE + 76, reject_ratio=1.1)
    first = led.add(_SECTION, "x", srcs=[])
    bullet_id = json.loads(first)["bullet_id"]
    snapshot_text = led.experience.to_text()
    snapshot_ops = list(led.ops)

    out = led.modify(bullet_id, "y" * 400, srcs=[])
    assert out.startswith("refused:")
    # Playbook unchanged.
    assert led.experience.to_text() == snapshot_text
    assert led.ops == snapshot_ops


def test_already_saturated_playbook_refuses_all_adds():
    """When the playbook starts ALREADY over the hard cap (e.g., user
    set an unrealistic budget), every ADD must refuse. The LLM should
    be guided to delete first."""
    led = _ledger(char_budget=80, reject_ratio=1.5)  # hard cap 120 < baseline
    out = led.add(_SECTION, "tiny", srcs=[])
    assert out.startswith("refused:")
    assert "delete" in out.lower()


# ── delete never trips the guard ─────────────────────────────────────────


def test_delete_does_not_trigger_char_check():
    """Deletes only free space, so they must always succeed (subject
    only to operation count limits)."""
    led = _ledger(char_budget=_BASELINE + 160, reject_ratio=1.5)
    first = led.add(_SECTION, "fits", srcs=[])
    bullet_id = json.loads(first)["bullet_id"]
    out = led.delete(bullet_id, justification="cleanup", srcs=[])
    parsed = json.loads(out)
    assert parsed.get("deleted") is True


def test_delete_works_even_when_playbook_over_hard_cap():
    """Even when char_budget is unrealistically tiny so the empty
    playbook already exceeds the hard cap, a delete on a non-existent
    bullet must fail with the normal 'not found' message — NOT a budget
    error. Confirms the char guard never intercepts delete."""
    led = _ledger(char_budget=80, reject_ratio=1.5)  # hard cap 120 < baseline
    res = led.delete("nonexistent-id", justification="x", srcs=[])
    assert "not found" in res
    assert "hard cap" not in res.lower()


# ── remaining() surfaces char state ──────────────────────────────────────


def test_remaining_includes_char_state_when_budget_enabled():
    soft = _BASELINE + 476
    led = _ledger(char_budget=soft, reject_ratio=1.5)
    rem = led.remaining()
    for k in ("add_left", "modify_left", "delete_left", "total_left"):
        assert k in rem
    assert rem["playbook_chars_current"] == _BASELINE
    assert rem["playbook_chars_soft_budget"] == soft
    assert rem["playbook_chars_hard_cap"] == int(soft * 1.5)
    # Empty playbook leaves plenty of headroom.
    assert rem["playbook_chars_until_refused"] == int(soft * 1.5) - _BASELINE


def test_remaining_omits_char_state_when_budget_disabled():
    led = _ledger(char_budget=0)
    rem = led.remaining()
    assert "add_left" in rem
    assert "playbook_chars_soft_budget" not in rem
    assert "playbook_chars_hard_cap" not in rem


def test_remaining_until_refused_can_be_negative_when_over_hard_cap():
    """``playbook_chars_until_refused`` goes negative when the playbook
    has already crossed the hard cap — telling the LLM how many chars
    to free up."""
    led = _ledger(char_budget=80, reject_ratio=2.0)  # hard cap 160 < baseline
    rem = led.remaining()
    assert rem["playbook_chars_until_refused"] < 0


# ── prompt states the budgets ────────────────────────────────────────────


def test_curator_prompt_states_char_cap_and_section_cap():
    from search_ttl.curator.react_staged import _render_curator_user_msg

    def _msg(cap: int) -> str:
        return _render_curator_user_msg(
            reflections=[],
            conserv=_Conserv(),
            max_react_turns=24,
            char_budget=20000,
            max_bullets_per_section=cap,
        )

    msg = _msg(8)
    assert "char cap 20000" in msg
    assert "bullet cap: 8" in msg
    assert "refused" in msg.lower()  # at the cap, ADD is refused
    # A disabled section cap is stated explicitly rather than omitted.
    assert "(disabled)" in _msg(0)

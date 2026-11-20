"""BCP-friendly initial playbook for ACE.

Replaces ACE's default finance-flavored sections (FORMULAS &
CALCULATIONS, CODE SNIPPETS & TEMPLATES, …) with sections suited to a
search-agent / deep-research benchmark like BrowseComp-Plus. ACE's
``_initialize_empty_playbook`` is only called when no ``initial_playbook``
arg is given, so passing this constant from our run script keeps ACE's
upstream code untouched.

Section names matter only for human readability and the slug
generator's prefix derivation (`get_section_slug` in
``third_party/ace/utils.py``); they don't affect ACE's behavior. The
slug map there falls back to initials (e.g. "Search Strategies" →
"ss") for unknown sections, so all of these will get sensible
2-character slugs.

If the curator emits a section name that isn't here, it lands under
``OTHERS`` (see ``third_party/ace/playbook_utils.py:148-151``). That's
a safety valve for unforeseen categories — leave it.
"""

from __future__ import annotations


BCP_INITIAL_PLAYBOOK = """## SEARCH STRATEGIES

## QUERY REFORMULATION

## ENTITY DISAMBIGUATION

## EVIDENCE EVALUATION

## DOMAIN HEURISTICS

## COMMON FAILURE MODES

## OTHERS"""


__all__ = ["BCP_INITIAL_PLAYBOOK"]

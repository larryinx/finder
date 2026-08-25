"""BrowseComp-Plus glue for ACE × ace_react.

Contains everything ACE needs to drive on BCP via HiExp's ReAct loop:

* :class:`DataProcessor` — ACE-compatible BCP processor with LLM-as-judge
  ``answer_is_correct``;
* :func:`load_bcp_split` — read a ``queries.tsv`` (path-driven) joined
  against the decrypted gold JSONL;
* :class:`ReActGenerator` — ACE Generator that runs HiExp's ReAct loop
  with the playbook injected and writes BCP-shape ``run_*.json``;
* :data:`BCP_INITIAL_PLAYBOOK` — the markdown skeleton used in place of
  ACE's finance-flavored default sections;
* :func:`trajectory_to_bcp_record` /:func:`write_bcp_run_json` — helpers
  for the BCP-shape run JSONs that ``evaluate_with_openai.py`` and our
  ``run_judge.py`` both read.

Lives under ``src/ace_react/bcp/`` (sibling of ``src/ace_react/agent.py``)
so all ACE-side glue stays in one package — the BCP-specific bits are
not separable from the ReAct adapter since they share the playbook
injection contract and the BCP-shape persistence format.
"""

from ace_react.bcp._playbook import BCP_INITIAL_PLAYBOOK
from ace_react.bcp.data_processor import DataProcessor, load_bcp_split
from ace_react.bcp.eval_progress import (
    evaluate_test_set_with_progress,
    install as install_progress_eval,
)
from ace_react.bcp.generator_react import ReActGenerator
from ace_react.bcp.persist import trajectory_to_bcp_record, write_bcp_run_json

__all__ = [
    "BCP_INITIAL_PLAYBOOK",
    "DataProcessor",
    "load_bcp_split",
    "ReActGenerator",
    "trajectory_to_bcp_record",
    "write_bcp_run_json",
    "evaluate_test_set_with_progress",
    "install_progress_eval",
]

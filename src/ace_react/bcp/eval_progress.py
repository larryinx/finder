"""Rich-progress drop-in for ACE's ``evaluate_test_set``.

Why: upstream ACE's ``evaluate_test_set`` only logs progress every 50
samples (``utils.py`` ~line 268). Our run pins ``test_workers=1`` (the
playbook is shared state in online mode; ``asyncio.run`` reentry is a
risk in offline) AND uses a ReAct generator that takes 30-120s per
call. That combination → 50+ minutes of silence before any feedback.

Drop-in replacement with the same signature and return shape, but:

* a Rich :class:`Progress` bar that ticks every sample;
* per-call latency in the bar so slow tail samples are visible;
* still uses ``ThreadPoolExecutor`` so ``max_workers>1`` works if a
  caller ever wants to flip it back.

Bind into ACE in :func:`install` — it monkey-patches both
``ace.utils`` and ``ace.ace`` so the name resolution path
(``from utils import *`` in ``ace/ace.py``) sees our version.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


_console = Console()


def evaluate_test_set_with_progress(
    data_processor,
    generator,
    playbook: str,
    test_samples: list[dict],
    max_tokens: int = 4096,
    log_dir: str | None = None,
    max_workers: int = 1,
    use_json_mode: bool = False,
) -> tuple[dict, dict]:
    """Drop-in replacement for ``ace.utils.evaluate_test_set`` with Rich progress.

    Same return shape as upstream::

        ({"accuracy": ..., "correct": ..., "total": ..., "no_answer": ...,
          "answers": [...], "targets": [...], "errors": [...]},
         {})    # second dict is for compatibility; upstream always empty
    """
    # Late import: upstream module must already be loaded by ACE itself.
    from utils import evaluate_single_test_sample  # type: ignore[import-not-found]

    n = len(test_samples)
    _console.rule(f"[bold]EVALUATING TEST SET — {n} samples, {max_workers} worker(s)[/bold]")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    results: dict[str, Any] = {
        "correct": 0,
        "total": 0,
        "no_answer": 0,
        "answers": [],
        "targets": [],
        "errors": [],
    }

    def _eval_one(args):
        return evaluate_single_test_sample(args, data_processor)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("acc=[bold green]{task.fields[acc]}[/]"),
        TextColumn("last={task.fields[last]:.1f}s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    )

    last_emit = time.time()
    with progress:
        task = progress.add_task(
            "eval", total=n, acc="0.000", last=0.0,
        )
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = {ex.submit(_eval_one, a): a for a in args_list}
            for fut in as_completed(futures):
                t0 = last_emit
                last_emit = time.time()
                latency = last_emit - t0
                try:
                    result, error = fut.result()
                except Exception as exc:  # noqa: BLE001
                    _console.print(f"[red]exec err[/red]: {exc!r}")
                    progress.advance(task)
                    continue

                if error:
                    _console.print(f"[red]{error}[/red]")
                    progress.advance(task)
                    continue

                if result and result["success"]:
                    is_correct = bool(result["is_correct"])
                    results["correct"] += int(is_correct)
                    results["total"] += 1
                    results["answers"].append(result["final_answer"])
                    results["targets"].append(result["target"])
                    if not is_correct:
                        results["errors"].append({
                            "index": result["index"],
                            "prediction": result["final_answer"],
                            "ground_truth": result["target"],
                        })
                    if result["final_answer"] == "No final answer found":
                        results["no_answer"] += 1

                acc = (
                    results["correct"] / results["total"]
                    if results["total"] > 0 else 0.0
                )
                progress.update(
                    task,
                    advance=1,
                    acc=f"{acc:.3f}",
                    last=latency,
                )

    if results["answers"] and results["targets"]:
        accuracy = data_processor.evaluate_accuracy(
            results["answers"], results["targets"],
        )
        final = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"],
            "answers": results["answers"],
            "targets": results["targets"],
            "errors": results["errors"],
        }
    else:
        final = {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "no_answer": 0,
            "answers": [],
            "targets": [],
            "errors": results["errors"],
        }
    _console.print(
        f"[bold green]✓[/bold green] eval done: "
        f"{final['correct']}/{final['total']} correct  "
        f"acc={final['accuracy']:.3f}  no_answer={final['no_answer']}"
    )
    # Upstream's contract for the second return value — see
    # ``third_party/ace/utils.py:282-285``. ACE's ``_online_train_and_test``
    # iterates ``error_logs['errors']``; returning ``{}`` triggers a
    # KeyError in upstream code.
    error_logs = {
        "accuracy": final["accuracy"],
        "errors": final["errors"],
    }
    return final, error_logs


def install() -> None:
    """Monkey-patch ACE's ``evaluate_test_set`` everywhere it's bound.

    ``ace.ace`` imports via ``from utils import *`` so the name lives in
    *both* ``ace.utils`` and ``ace.ace``. We replace both so the patch
    survives the ``from`` binding.
    """
    import importlib

    for mod_name in ("utils", "ace.ace"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "evaluate_test_set"):
            mod.evaluate_test_set = evaluate_test_set_with_progress  # type: ignore[attr-defined]


__all__ = ["evaluate_test_set_with_progress", "install"]

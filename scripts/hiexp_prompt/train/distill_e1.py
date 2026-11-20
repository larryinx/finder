"""Phase 2 of HEK construction: contrastive distillation → E1 entries.

Implements paper §3.1.1 + Algorithm 1 lines 2-9:

    "We leverage the self-reflection capabilities of either the policy
    model or a superior teacher model to identify two critical features:
    key decision points and reasoning traps. The output of contrastive
    distillation is formalized as case-based experience e_i and its
    corresponding summary description d_i ..."

Reads ``rollouts.jsonl`` from phase 1, keeps only examples with at least
one positive and one negative trajectory (Algorithm 1 line 5), feeds
each (question, +traj, -traj) pair to the LLM via the Table 11 prompt,
parses the resulting JSON, and writes one ``ExperienceEntry(level=1)``
per successful pair into ``<hek_path>/e1.jsonl``.

Self-distillation default (paper Table 6): the same model used by the
agent (Qwen2.5-7B-Instruct) does the reflection, since "self-distillation
slightly outperforms strong-teacher distillation in multi-hop reasoning
tasks".
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from hiexp.experience.schema import ExperienceEntry
from hiexp_prompt._bootstrap import bootstrap
from hiexp_prompt._llm_utils import (
    achat_once,
    make_chat_model,
    parse_json_object,
)
from hiexp_prompt._prompts import build_contrastive_distillation_prompt
from hiexp_prompt._shared import append_jsonl, render_trajectory_for_prompt

bootstrap()

app = typer.Typer(add_completion=False)
console = Console()


def _iter_rollouts(rollouts_path: Path):
    """Yield rollout dicts from either a single file or a sharded directory.

    Accepts:
      * a path to ``rollouts.jsonl`` (single-file output);
      * a path to a directory containing ``rollouts.jsonl`` and/or
        ``rollouts-NNNNN.jsonl`` shards (sample_rollouts.py --shard-size).
    """
    if rollouts_path.is_file():
        files = [rollouts_path]
    elif rollouts_path.is_dir():
        files = []
        single = rollouts_path / "rollouts.jsonl"
        if single.exists():
            files.append(single)
        files.extend(sorted(rollouts_path.glob("rollouts-*.jsonl")))
        if not files:
            raise FileNotFoundError(
                f"no rollouts*.jsonl found under {rollouts_path}"
            )
    else:
        raise FileNotFoundError(rollouts_path)
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


@app.command()
def main(
    rollouts_path: Path = typer.Option(
        ...,
        help=(
            "Phase-1 rollouts source — either ``rollouts.jsonl`` or a "
            "directory containing it / ``rollouts-NNNNN.jsonl`` shards."
        ),
    ),
    hek_path: Path = typer.Option(..., help="HEK directory; e1.jsonl is appended in place."),
    llm_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        envvar="AGENT_LLM_MODEL",
        help="Policy model for self-distillation (paper Table 6). "
        "Defaults from AGENT_LLM_MODEL.",
    ),
    llm_base_url: str = typer.Option(
        "http://localhost:8000/v1",
        envvar="AGENT_LLM_BASE_URL",
        help="OpenAI-compatible endpoint. Defaults from AGENT_LLM_BASE_URL.",
    ),
    llm_api_key: str = typer.Option(
        "EMPTY",
        envvar="AGENT_LLM_API_KEY",
        help="API key. Defaults from AGENT_LLM_API_KEY.",
    ),
    temperature: float = typer.Option(0.3, help="Lower than rollout sampling for stable JSON."),
    max_tokens: int = typer.Option(1024),
    max_chars_per_traj: int = typer.Option(
        2400, help="Trim each trajectory to this many chars before prompting."
    ),
    concurrency: int = typer.Option(
        8,
        envvar=["AGENT_DISTILL_CONCURRENCY", "AGENT_LLM_CONCURRENCY"],
        help=(
            "Concurrent LLM calls for contrastive distillation. "
            "Defaults from AGENT_DISTILL_CONCURRENCY then AGENT_LLM_CONCURRENCY."
        ),
    ),
) -> None:
    hek_path.mkdir(parents=True, exist_ok=True)
    e1_path = hek_path / "e1.jsonl"

    llm = make_chat_model(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    pairs: list[dict] = []
    for obj in _iter_rollouts(rollouts_path):
        if obj.get("n_pos", 0) > 0 and obj.get("n_neg", 0) > 0:
            pairs.append(obj)

    console.print(
        f"[bold]Distillation[/bold] {len(pairs)} usable rollout files; writing → {e1_path}"
    )

    # Skip ids already turned into E1 (resume safety).
    done_titles: set[str] = set()
    if e1_path.exists():
        for line in e1_path.read_text(encoding="utf-8").splitlines():
            try:
                done_titles.add(json.loads(line).get("title", ""))
            except Exception:
                continue

    start = time.time()
    successes = 0
    failures = 0
    write_lock = threading.Lock()

    async def _drive() -> None:
        nonlocal successes, failures
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(pair: dict) -> tuple[dict, str | None, BaseException | None]:
            async with sem:
                pos = pair["pos"][0]
                neg = pair["neg"][0]
                pos_text = render_trajectory_for_prompt(
                    pos, max_chars_per_item=max_chars_per_traj // 4
                )
                neg_text = render_trajectory_for_prompt(
                    neg, max_chars_per_item=max_chars_per_traj // 4
                )
                prompt = build_contrastive_distillation_prompt(
                    question=pair["question"],
                    successful_trajectory_text=pos_text,
                    failed_trajectory_text=neg_text,
                )
                try:
                    raw = await achat_once(llm, prompt)
                    return pair, raw, None
                except Exception as exc:  # noqa: BLE001
                    return pair, None, exc

        tasks = [asyncio.create_task(_one(p)) for p in pairs]
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"distill E1 ({concurrency} in-flight)", total=len(tasks)
            )
            for fut in asyncio.as_completed(tasks):
                pair, raw, err = await fut
                progress.update(task_id, advance=1)
                if err is not None:
                    console.print(
                        f"[red]LLM err[/red] qid={pair['query_id']}: {err!r}"
                    )
                    failures += 1
                    continue
                parsed = parse_json_object(raw or "")
                if (
                    parsed is None
                    or "title" not in parsed
                    or "description" not in parsed
                ):
                    raw_str = (raw or "").strip()
                    if not raw_str:
                        reason = "empty response (likely thinking ate token budget)"
                    elif parsed is None:
                        reason = f"no JSON parse; head={raw_str[:140]!r}"
                    else:
                        reason = (
                            f"missing title/description; keys={list(parsed.keys())}"
                        )
                    console.print(
                        f"[red]parse err[/red] qid={pair['query_id']}: {reason}"
                    )
                    failures += 1
                    continue
                title = str(parsed["title"])
                with write_lock:
                    if title in done_titles:
                        continue
                    try:
                        entry = ExperienceEntry(
                            level=1,
                            type=str(parsed.get("type", "")),
                            title=title,
                            tags=list(parsed.get("tags") or []),
                            description=str(parsed["description"]),
                            thinking=str(parsed.get("thinking", "")),
                        )
                    except Exception as exc:  # noqa: BLE001
                        console.print(
                            f"[red]schema err[/red] qid={pair['query_id']}: {exc!r}"
                        )
                        failures += 1
                        continue
                    append_jsonl(e1_path, [entry])
                    done_titles.add(entry.title)
                    successes += 1

    asyncio.run(_drive())

    console.print(
        f"[bold green]✓[/bold green] {successes} E1 entries written, "
        f"{failures} failed; {round(time.time()-start, 1)}s"
    )


if __name__ == "__main__":
    app()

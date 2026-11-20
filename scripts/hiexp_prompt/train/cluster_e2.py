"""Phase 3 of HEK construction: cluster E1 → E2 strategic experiences.

Implements paper §3.1.2 + §C:

    "we apply agglomerative clustering to v_i for subsequent
    aggregation. ... For each identified cluster C_j^(1), we leverage an
    LLM to distill multiple instance-level experiences into a
    generalized strategic experience."
    "scikit-learn's agglomerative clustering with Ward linkage"

Reads ``<hek_path>/e1.jsonl``, encodes the titles with the same
multilingual-e5-base model used for HEK retrieval, runs
``AgglomerativeClustering(linkage='ward', distance_threshold=tau)``,
and feeds each cluster's entries into the Table 12 prompt to obtain a
generalized E2 entry. Writes ``<hek_path>/e2.jsonl``.

Singleton clusters are skipped (no aggregation possible).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from hiexp.experience.embedder import E5Embedder
from hiexp.experience.schema import ExperienceEntry
from hiexp_prompt._bootstrap import bootstrap
from hiexp_prompt._llm_utils import chat_once, make_chat_model, parse_json_array
from hiexp_prompt._prompts import build_clustering_prompt
from hiexp_prompt._shared import append_jsonl

bootstrap()

app = typer.Typer(add_completion=False)
console = Console()


def _load_entries(path: Path) -> list[ExperienceEntry]:
    out: list[ExperienceEntry] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ExperienceEntry.from_jsonl(line))
    return out


def _render_cluster_input(entries: list[ExperienceEntry]) -> str:
    """Serialize entries as the JSON shape Table 12 specifies."""
    payload = [
        {
            "id": e.id,
            "type": e.type,
            "title": e.title,
            "tags": e.tags,
            "description": e.description,
            "thinking": e.thinking,
        }
        for e in entries
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.command()
def main(
    hek_path: Path = typer.Option(..., help="HEK directory containing e1.jsonl."),
    distance_threshold: float = typer.Option(
        0.6,
        help=(
            "Ward-linkage distance threshold. Paper does not pin a value. "
            "Empirically on L2-normalized e5 embeddings: τ≈0.6 yields ~5-15 "
            "clusters per ~100 entries; τ≥1.2 collapses everything into one "
            "cluster (which then exceeds max_tokens for the LLM aggregator)."
        ),
    ),
    embedding_model: str = typer.Option(
        "intfloat/multilingual-e5-base",
        help="Same e5 model used for HEK retrieval (paper §A).",
    ),
    target_level: int = typer.Option(
        2,
        help="Cluster output level (2 = E2 over E1 input; 3 = E3 over E2 input).",
    ),
    llm_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", envvar="AGENT_LLM_MODEL"
    ),
    llm_base_url: str = typer.Option(
        "http://localhost:8000/v1", envvar="AGENT_LLM_BASE_URL"
    ),
    llm_api_key: str = typer.Option("EMPTY", envvar="AGENT_LLM_API_KEY"),
    temperature: float = typer.Option(0.3),
    max_tokens: int = typer.Option(
        4096,
        help=(
            "LLM max output tokens for the cluster→E2 aggregator. Bumped "
            "from 2048 because clusters >20 members were getting their "
            "JSON arrays truncated mid-output."
        ),
    ),
) -> None:
    from sklearn.cluster import AgglomerativeClustering

    in_level = target_level - 1
    in_path = hek_path / f"e{in_level}.jsonl"
    out_path = hek_path / f"e{target_level}.jsonl"

    entries = _load_entries(in_path)
    if len(entries) < 2:
        console.print(f"[yellow]skip[/yellow] only {len(entries)} entries at e{in_level}; need ≥ 2")
        return
    console.print(
        f"[bold]Clustering[/bold] {len(entries)} e{in_level} entries → e{target_level} "
        f"(τ={distance_threshold})"
    )

    embedder = E5Embedder(model_name=embedding_model)
    embeddings = embedder.encode_passages([e.title for e in entries])

    # Ward linkage requires Euclidean; e5 vectors are L2-normalized so
    # ‖a-b‖² = 2(1 - cos(a,b)) — relative distances are still meaningful.
    cluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward",
    )
    labels = cluster.fit_predict(np.asarray(embeddings, dtype=np.float32))
    groups: dict[int, list[ExperienceEntry]] = defaultdict(list)
    for entry, label in zip(entries, labels):
        groups[int(label)].append(entry)

    console.print(
        f"  found {len(groups)} clusters; "
        f"sizes={sorted((len(g) for g in groups.values()), reverse=True)[:10]}..."
    )

    llm = make_chat_model(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    start = time.time()
    written = 0
    for cluster_id, members in groups.items():
        if len(members) < 2:
            continue  # singletons can't be generalized
        prompt = build_clustering_prompt(_render_cluster_input(members))
        # Rough sanity check: a cluster of ~20+ E1 entries (~6KB JSON
        # input) often produces output that overruns small max_tokens.
        # Warn the user up front so a parse failure isn't a mystery.
        approx_in_tokens = len(prompt) // 4
        if approx_in_tokens > max_tokens:
            console.print(
                f"[yellow]warn[/yellow] cluster={cluster_id} size={len(members)} "
                f"prompt~{approx_in_tokens} tokens > max_tokens={max_tokens}; "
                "lower --distance-threshold or raise --max-tokens"
            )
        try:
            raw = chat_once(llm, prompt)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]LLM err[/red] cluster={cluster_id}: {exc}")
            continue
        parsed = parse_json_array(raw)
        if parsed is None:
            head = (raw or "").strip()[:140]
            reason = "empty response" if not head else f"head={head!r}"
            console.print(
                f"[yellow]parse fail[/yellow] cluster={cluster_id}: {reason}"
            )
            continue
        for obj in parsed:
            if not isinstance(obj, dict) or "title" not in obj or "description" not in obj:
                continue
            try:
                entry = ExperienceEntry(
                    level=target_level,  # type: ignore[arg-type]
                    type=str(obj.get("type", "")),
                    title=str(obj["title"]),
                    tags=list(obj.get("tags") or []),
                    description=str(obj["description"]),
                    thinking=str(obj.get("thinking", "")),
                    source_ids=[m.id for m in members],
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]schema err[/red] cluster={cluster_id}: {exc}")
                continue
            append_jsonl(out_path, [entry])
            written += 1

    console.print(
        f"[bold green]✓[/bold green] {written} e{target_level} entries → {out_path} "
        f"({round(time.time()-start, 1)}s)"
    )


if __name__ == "__main__":
    app()

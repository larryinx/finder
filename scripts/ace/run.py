"""ACE on a benchmark — entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Upstream ACE isn't pip-installable (no setup.py / pyproject), so we
# add ``third_party/ace`` to sys.path to make ``import ace`` resolve to
# ``third_party/ace/ace/__init__.py``. Our own packages (hiexp,
# hiexp_prompt, ace_react) are installed via the project's pyproject
# wheel, so no further path hacks are needed.
sys.path.insert(0, str(_PROJECT_ROOT / "third_party" / "ace"))

# Load .env so AGENT_LLM_* env vars resolve before anything imports them.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

import openai  # noqa: E402

from ace import ACE  # type: ignore[import-not-found]  # upstream  # noqa: E402

from hiexp import (  # noqa: E402
    HiExpConfig,
    compose,
    init_weave,
    list_options,
    log_playbook_checkpoint,
)

from ace_react.bcp import (  # noqa: E402
    BCP_INITIAL_PLAYBOOK,
    DataProcessor,
    ReActGenerator,
    install_progress_eval,
    load_bcp_split,
)

# Replace ACE's silent-for-50-samples evaluate_test_set with a Rich
# progress bar version. Must happen *after* ``from ace import ACE`` so
# the upstream module is loaded into sys.modules.
install_progress_eval()


def parse_args() -> argparse.Namespace:
    benches = " | ".join(list_options("ace", "benchmark")) or "(none)"
    retrs = " | ".join(list_options("ace", "retriever")) or "(none)"
    modes = " | ".join(list_options("ace", "mode")) or "(none)"
    ap = argparse.ArgumentParser(description="Run ACE on a benchmark.")
    ap.add_argument(
        "--benchmark", default="bcp",
        help=f"Benchmark config under config/ace/benchmark/. One of: {benches}.",
    )
    ap.add_argument(
        "--retriever", default="qwen3-8b",
        help=f"Retriever config under config/ace/retriever/. One of: {retrs}.",
    )
    ap.add_argument(
        "--mode", default=None,
        help=f"Mode config under config/ace/mode/. One of: {modes}. "
        "Required unless overridden via -o ace.mode=...",
    )
    ap.add_argument(
        "--save-path", required=True,
        help="Top-level output dir; runs/ace/<config>/ etc. land underneath.",
    )
    ap.add_argument(
        "--no-ground-truth", action="store_true",
        help="Deny gold to Reflector (paper Table 1 GT=✗ setting). The "
        "controller's verdict still comes from the LLM-as-judge in "
        "answer_is_correct, since that path can't be turned off.",
    )
    ap.add_argument("--initial-playbook-path", default=None,
                    help="Pre-trained playbook to start from. eval_only requires this.")
    ap.add_argument("--cap-train", type=int, default=0,
                    help=">0 to truncate the train split (smoke runs).")
    ap.add_argument("--cap-test", type=int, default=0,
                    help=">0 to truncate the test split (smoke runs).")
    ap.add_argument(
        "-o", "--override", nargs="*", default=[],
        help="OmegaConf dotlist overrides applied after YAML merge. "
        "Example: -o ace.num_epochs=2 splits.train=train100",
    )
    return ap.parse_args()


def _run_dir_name(mode: str, no_ground_truth: bool) -> str:
    gt = "nogt" if no_ground_truth else "gt"
    return f"ace_{mode}_{gt}"


def _maybe_truncate(samples, cap):
    return samples if (not cap or cap <= 0) else samples[:cap]


def _probe_mcp_or_die(base_cfg) -> None:
    """Quick MCP connectivity check; exits with a clear message if it fails.

    The agent loop creates a fresh ``ReActAgent`` per query, each of which
    re-connects to MCP. If MCP is unreachable, every single sample fails
    with an opaque ``ExceptionGroup`` and ACE plows ahead until it hits
    a 0/0 divide deep inside its windowing code. Probe once at startup
    so the failure mode is "5 seconds and a useful error message"
    instead of "5 minutes and a stack trace from the wrong place".
    """
    import asyncio

    from hiexp.retrieval.mcp import MCPToolLoader

    print(f"[probe] checking MCP at {base_cfg.mcp_url} ...")
    try:
        async def _check():
            loader = MCPToolLoader(base_cfg)
            tools = await loader.aload()
            return [t.name for t in tools]
        names = asyncio.run(asyncio.wait_for(_check(), timeout=15.0))
    except BaseExceptionGroup as eg:  # type: ignore[misc]
        inner = eg.exceptions[0] if eg.exceptions else eg
        print(
            f"\n[FATAL] MCP connection failed: {type(inner).__name__}: {inner}\n"
            f"        mcp_url = {base_cfg.mcp_url}\n"
            f"        is your MCP server (BCP retriever) running on that port?",
            file=sys.stderr,
        )
        sys.exit(2)
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        print(
            f"\n[FATAL] MCP connection failed: {type(exc).__name__}: {exc}\n"
            f"        mcp_url = {base_cfg.mcp_url}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"[probe] MCP OK — tools: {names}")


def _patch_phase_tagging(ace_obj, react_generator) -> None:
    """Bucket persisted runs by phase so they don't overwrite each other.

    Without this, in online mode the same ``run_<qid>.json`` is written
    by (a) the initial test, (b) the window pre-test that covers that
    qid, and (c) the training step that uses that qid as input — each
    overwriting the previous, leaving only the LAST generation on disk.
    By tagging the generator before each phase, we route writes into
    ``runs/initial_test/``, ``runs/window_<N>_pretest/``,
    ``runs/training/``, and ``runs/final_test/`` — three or four
    distinct snapshots of the model on the same qid as the playbook
    evolves.
    """
    original_run_test = ace_obj._run_test
    state = {"window_idx": 0}

    def run_test_wrapper(*args, **kw):
        prefix = kw.get("prefix", "test")
        # ``prefix`` is one of "initial" / "final" / "test" (eval_only).
        react_generator.set_phase(f"{prefix}_test")
        try:
            return original_run_test(*args, **kw)
        finally:
            react_generator.set_phase(None)

    ace_obj._run_test = run_test_wrapper

    # Tag online window pre-tests too. ACE invokes ``evaluate_test_set``
    # directly (not via _run_test) for those, so we can't wrap _run_test
    # here. Instead we wrap the *patched* evaluate_test_set we installed
    # via install_progress_eval — see ``ace_react.bcp.eval_progress``.
    try:
        from ace.utils import evaluate_test_set as installed  # type: ignore[import-not-found]
    except ImportError:
        installed = None
    if installed is None:
        return

    def eval_test_set_wrapper(*args, **kw):
        # If a phase was already set externally (by run_test_wrapper) honor it;
        # otherwise we're being invoked from _online_train_and_test's window
        # pre-test path → tag with a window counter.
        if react_generator._phase is None:
            state["window_idx"] += 1
            react_generator.set_phase(f"window_{state['window_idx']}_pretest")
            try:
                return installed(*args, **kw)
            finally:
                react_generator.set_phase(None)
        return installed(*args, **kw)

    # Same monkey-patch dance as install_progress_eval — replace the
    # binding in both places ACE consults.
    import importlib
    for mod_name in ("utils", "ace.ace"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "evaluate_test_set"):
            mod.evaluate_test_set = eval_test_set_wrapper  # type: ignore[attr-defined]


def _patch_skip_initial_test(ace_obj) -> None:
    """Make the *first* ``ace._run_test`` call a no-op stub.

    ACE's offline flow calls ``_run_test`` exactly twice: once at the
    top with prefix="initial" (empty-playbook baseline), once at the
    bottom with prefix="final" (best-playbook eval). With ReAct
    generators that take 30-120s per sample and ``test_workers=1``,
    the initial test eats hours scoring something the user didn't ask
    for. We replace it with a stub that returns the same shape so
    ACE's bookkeeping doesn't break, and let call #2 (final test)
    pass through unchanged.
    """
    original = ace_obj._run_test
    state = {"calls": 0}

    def wrapper(*args, **kwargs):
        state["calls"] += 1
        prefix = kwargs.get("prefix", "?")
        if state["calls"] == 1:
            print(
                f"[skip] _run_test #{state['calls']} (prefix={prefix!r}) — "
                "skip_initial_test=True in YAML"
            )
            return {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "no_answer": 0,
                "answers": [],
                "targets": [],
                "errors": [],
            }
        return original(*args, **kwargs)

    ace_obj._run_test = wrapper


def _build_no_think_openai_client(base_url: str, api_key: str) -> "openai.OpenAI":
    """OpenAI client that auto-injects ``enable_thinking=False`` on every call.

    SGLang serving Qwen3 reasoning models defaults to thinking on. With the
    Reflector/Curator's ~3000-char prompts and 4096-token budget, the model
    spends the whole budget in ``reasoning_content`` and ``content`` comes
    back as ``None``. ACE's ``timed_llm_call`` then raises
    ``Exception("API returned None content")`` and the run dies mid-training.

    Reflector / Curator both produce structured outputs (JSON schemas) —
    they don't need CoT. Forcing ``enable_thinking=False`` server-side via
    ``extra_body.chat_template_kwargs`` puts the whole budget into
    ``content`` where ACE expects it.

    Nemotron-3 needs the inverse rescue: its template honors
    ``enable_thinking=False`` (the model answers directly) but then never
    emits ``</think>``, so SGLang's ``nemotron_3`` reasoning parser files
    the entire answer under ``reasoning_content`` and ``content`` comes
    back ``None`` anyway. When that happens we lift ``reasoning_content``
    into ``content`` so ACE sees the answer either way.
    """
    client = openai.OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    original_create = client.chat.completions.create

    def patched_create(**kw):
        extra = dict(kw.pop("extra_body", {}) or {})
        ctk = dict(extra.get("chat_template_kwargs", {}) or {})
        ctk.setdefault("enable_thinking", False)
        extra["chat_template_kwargs"] = ctk
        response = original_create(extra_body=extra, **kw)
        for choice in response.choices or []:
            msg = choice.message
            if msg.content is None:
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    msg.content = reasoning
        return response

    client.chat.completions.create = patched_create  # type: ignore[assignment]
    return client


def _patch_openai_clients(ace_obj, base_url: str, api_key: str) -> None:
    """Repoint ACE's reflector + curator clients at our SGLang/vLLM endpoint.

    ACE's ``initialize_clients(api_provider='openai')`` hardcodes
    ``base_url=https://api.openai.com/v1`` — we don't want that. The
    Generator client gets replaced by ``ReActGenerator`` anyway, so we
    only need to fix Reflector and Curator here. We also force
    ``enable_thinking=False`` on these clients (see
    :func:`_build_no_think_openai_client`).
    """
    real = _build_no_think_openai_client(base_url, api_key)
    ace_obj.reflector.api_client = real
    ace_obj.curator.api_client = real
    if hasattr(ace_obj, "bulletpoint_analyzer") and ace_obj.bulletpoint_analyzer is not None:
        ace_obj.bulletpoint_analyzer.api_client = real


def main() -> None:
    args = parse_args()

    # Mirror EVAL_JUDGE_API_KEY / AGENT_LLM_API_KEY → OPENAI_API_KEY so ACE's
    # ``initialize_clients("openai")`` doesn't bail out at import time.
    os.environ.setdefault(
        "OPENAI_API_KEY",
        os.environ.get("EVAL_JUDGE_API_KEY")
        or os.environ.get("AGENT_LLM_API_KEY")
        or "EMPTY",
    )

    # ── Compose YAML config ─────────────────────────────────────────────
    cfg = compose(
        family="ace",
        benchmark=args.benchmark,
        retriever=args.retriever,
        mode=args.mode,
        overrides=args.override,
    )
    mode = cfg.ace.mode
    no_gt = bool(args.no_ground_truth or cfg.ace.no_ground_truth)

    # ── Tracing (Weave + W&B) ───────────────────────────────────────────
    # Reads WEAVE_DISABLED / WEAVE_CAPTURE_CODE / WANDB_API_KEY /
    # WANDB_ENTITY (or WANDB_TEAM) / WANDB_PROJECT from env. Returns
    # None silently when WEAVE_DISABLED=1 or W&B isn't configured, so
    # the run still works without tracing.
    from omegaconf import OmegaConf
    init_weave(
        run_name=f"ace-{mode}-{'nogt' if no_gt else 'gt'}-{args.benchmark}-{args.retriever}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[f"benchmark:{cfg.benchmark.name}", f"mode:{mode}",
              f"retriever:{cfg.retriever.name}", "ace"],
        job_type=f"ace_{mode}",
        phase=f"ace_{mode}",  # gate via WEAVE_DISABLED_ACE_OFFLINE / _ACE_ONLINE
    )

    # ── Load splits ─────────────────────────────────────────────────────
    if cfg.benchmark.name != "browsecomp_plus":
        raise NotImplementedError(
            f"ACE entrypoint currently only supports BCP; got benchmark="
            f"{cfg.benchmark.name!r}. Add a loader for the new benchmark "
            "and update this dispatch."
        )
    decrypted = cfg.data.decrypted_jsonl
    if mode == "offline":
        train = _maybe_truncate(
            load_bcp_split(cfg.data.splits.train, decrypted_jsonl=decrypted),
            args.cap_train,
        )
        val = load_bcp_split(cfg.data.splits.val, decrypted_jsonl=decrypted)
        test = _maybe_truncate(
            load_bcp_split(cfg.data.splits.test, decrypted_jsonl=decrypted),
            args.cap_test,
        )
    else:
        train, val = None, None
        test = _maybe_truncate(
            load_bcp_split(cfg.data.splits.test, decrypted_jsonl=decrypted),
            args.cap_test,
        )

    print(f"[run] mode={mode} no_ground_truth={no_gt}")
    if train is not None:
        print(f"[run] train={len(train)}  val={len(val)}  test={len(test)}")
    else:
        print(f"[run] test={len(test)}")

    # ── Base HiExpConfig (YAML + env) ────────────────────────────────────
    # ``question_template="browsecomp_plus"`` is REQUIRED here. It tells
    # the agent to ask the model for ``Explanation: / Exact Answer: /
    # Confidence:`` blocks — which is what ``ReActGenerator`` then runs
    # ``extract_exact_answer`` against.
    base_cfg = HiExpConfig.from_omegaconf(cfg, enable_hek=False)
    base_cfg.validate()

    # Fail fast on a dead MCP server. Without this, ACE will silently
    # eat 200+ samples (each erroring as
    # ``ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)``)
    # before crashing in ``_online_train_and_test`` on a 0/0 divide.
    _probe_mcp_or_die(base_cfg)
    print(f"[run] llm={base_cfg.llm_model}  base={base_cfg.llm_base_url}  "
          f"flavor={base_cfg.llm_api_flavor}  mcp={base_cfg.mcp_url}")

    # ── DataProcessor (judge LLM picked from env) ────────────────────────
    processor = DataProcessor()

    # ── Initial playbook ────────────────────────────────────────────────
    if args.initial_playbook_path:
        initial_playbook = Path(args.initial_playbook_path).read_text(encoding="utf-8")
        print(f"[run] loaded initial playbook from {args.initial_playbook_path} "
              f"({len(initial_playbook)} chars)")
    else:
        initial_playbook = BCP_INITIAL_PLAYBOOK
        sections = [
            ln.strip()[3:]
            for ln in BCP_INITIAL_PLAYBOOK.splitlines()
            if ln.startswith("## ")
        ]
        print(f"[run] using BCP-flavored empty playbook (sections: {sections})")

    # ── Build ACE ───────────────────────────────────────────────────────
    ace = ACE(
        api_provider="openai",  # value-irrelevant after _patch_openai_clients
        generator_model=base_cfg.llm_model,
        reflector_model=base_cfg.llm_model,
        curator_model=base_cfg.llm_model,
        max_tokens=base_cfg.max_response_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=cfg.ace.use_bulletpoint_analyzer,
    )
    _patch_openai_clients(
        ace,
        base_url=base_cfg.llm_base_url,
        api_key=base_cfg.llm_api_key,
    )
    if mode == "offline" and cfg.ace.skip_initial_test:
        _patch_skip_initial_test(ace)

    # ── Build the BCP-shape runs dir + ReActGenerator + register ────────
    save_root = Path(args.save_path) / _run_dir_name(mode, no_gt)
    runs_dir = save_root / "runs"
    save_root.mkdir(parents=True, exist_ok=True)

    # Build the question→task_dict registry once. Includes train/val too
    # so qid → run_<qid>.json works for any persisted call.
    registry: dict[str, dict] = {}
    for samples in (train, val, test):
        for td in samples or []:
            registry[td["question"]] = td

    react_generator = ReActGenerator(
        base_config=base_cfg,
        runs_dir=runs_dir,
        task_registry=registry,
        run_metadata_extra={
            "mode": mode,
            "no_ground_truth": no_gt,
            "retriever": cfg.retriever.name,
        },
    )
    ace.generator = react_generator
    _patch_phase_tagging(ace, react_generator)

    # ── Drive ACE ───────────────────────────────────────────────────────
    config = {
        "num_epochs": cfg.ace.num_epochs,
        "max_num_rounds": cfg.ace.max_num_rounds,
        "curator_frequency": cfg.ace.curator_frequency,
        "playbook_token_budget": cfg.ace.token_budget,
        "eval_steps": cfg.ace.eval_steps,
        "save_steps": cfg.ace.save_steps,
        "task_name": cfg.benchmark.name,
        "mode": mode,
        "json_mode": False,  # ReAct generator returns BCP-shape, not JSON mode
        "no_ground_truth": no_gt,
        "save_dir": str(save_root),
        "test_workers": cfg.ace.test_workers,
        "use_bulletpoint_analyzer": cfg.ace.use_bulletpoint_analyzer,
    }
    ace.run(
        mode=mode,
        train_samples=train,
        val_samples=val,
        test_samples=test,
        data_processor=processor,
        config=config,
    )

    # ── Upload playbook checkpoints to W&B ──────────────────────────────
    # ACE writes its artifacts into ``save_root/ace_run_<ts>_<task>_<mode>/``.
    # Find the latest such dir (ACE created it just above) and log each
    # checkpoint as a versioned artifact. No-op when WEAVE_DISABLED.
    try:
        ace_run_dirs = sorted(save_root.glob("ace_run_*"), key=lambda p: p.stat().st_mtime)
        if ace_run_dirs:
            run_dir = ace_run_dirs[-1]
            artifact_basename = f"ace-playbook-{mode}-{'nogt' if no_gt else 'gt'}"
            # Intermediate checkpoints (offline: epoch_X_step_Y; online: window_N_final)
            inter_dir = run_dir / "intermediate_playbooks"
            if inter_dir.exists():
                for p in sorted(inter_dir.glob("*.txt")):
                    log_playbook_checkpoint(
                        p,
                        name=artifact_basename,
                        metadata={
                            "checkpoint_kind": "intermediate",
                            "filename": p.name,
                            "ace_run_dir": str(run_dir),
                        },
                    )
            # Terminal checkpoints
            for fname, kind in [
                ("final_playbook.txt", "final"),
                ("best_playbook.txt", "best"),
            ]:
                p = run_dir / fname
                if p.exists():
                    log_playbook_checkpoint(
                        p,
                        name=artifact_basename,
                        metadata={"checkpoint_kind": kind, "filename": fname,
                                  "ace_run_dir": str(run_dir)},
                    )
            print(f"[run] uploaded ACE playbook checkpoints from {run_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[run] playbook checkpoint upload skipped: {exc!r}")

    print(f"[run] done; results under {save_root}")
    print(f"[run] BCP-shape run JSONs in {runs_dir} — re-judge with "
          f"scripts/hiexp_prompt/eval/run_judge.py for cross-validation.")

    from hiexp.tracing import finish as _finish_tracing
    _finish_tracing()


if __name__ == "__main__":
    main()

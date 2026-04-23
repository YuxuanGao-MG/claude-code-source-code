"""Upload a finished run directory to a public Hugging Face dataset.

Each run becomes a folder in the dataset repo::

    {HF_DATASET_REPO}/
        runs/
            {run_id}/
                calls.jsonl
                completed_tasks.jsonl
                meta.json              # written here; summary of the run
        index.json                     # list of all runs, updated per upload

Public dataset URL pattern::

    https://huggingface.co/datasets/{HF_DATASET_REPO}
    https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/runs/{run_id}/calls.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from .config import load_config


def _iter_call_rows(calls_path: Path) -> Iterable[dict]:
    if not calls_path.exists():
        return
    with calls_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _summarize(run_dir: Path) -> dict:
    calls = list(_iter_call_rows(run_dir / "calls.jsonl"))
    by_model: dict[str, dict] = {}
    total_cost = 0.0
    total_calls = len(calls)
    total_errors = 0
    for row in calls:
        model = row.get("model_name") or "?"
        m = by_model.setdefault(model, {"calls": 0, "errors": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0})
        m["calls"] += 1
        if row.get("error"):
            m["errors"] += 1
            total_errors += 1
        cost = float(row.get("cost_usd") or 0.0)
        m["cost_usd"] += cost
        total_cost += cost
        usage = row.get("usage") or {}
        m["tokens_in"] += int(usage.get("prompt_tokens") or 0)
        m["tokens_out"] += int(usage.get("completion_tokens") or 0)
    return {
        "run_id": run_dir.name,
        "generated_at": time.time(),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "total_cost_usd": round(total_cost, 6),
        "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_model.items()},
    }


def _ensure_dataset(api, repo_id: str) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        api.dataset_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)


def _update_index(api, repo_id: str, run_id: str, summary: dict) -> None:
    from huggingface_hub.utils import EntryNotFoundError
    try:
        path = api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="index.json")
        index = json.loads(Path(path).read_text())
        if not isinstance(index, list):
            index = []
    except EntryNotFoundError:
        index = []
    except Exception:
        index = []

    index = [e for e in index if isinstance(e, dict) and e.get("run_id") != run_id]
    index.append({
        "run_id": run_id,
        "generated_at": summary["generated_at"],
        "total_cost_usd": summary["total_cost_usd"],
        "total_calls": summary["total_calls"],
        "total_errors": summary["total_errors"],
        "models": sorted(summary["by_model"].keys()),
    })
    index.sort(key=lambda e: e.get("generated_at", 0), reverse=True)

    tmp = Path("/tmp") / f"openmesh_index_{int(time.time()*1000)}.json"
    tmp.write_text(json.dumps(index, indent=2))
    try:
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo="index.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"update index for {run_id}",
        )
    finally:
        tmp.unlink(missing_ok=True)


def upload_run(run_dir: Path, *, repo_id: str, hf_token: str) -> dict:
    from huggingface_hub import HfApi

    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    run_id = run_dir.name

    summary = _summarize(run_dir)
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(summary, indent=2))

    api = HfApi(token=hf_token)
    _ensure_dataset(api, repo_id)

    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=f"runs/{run_id}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"upload run {run_id}",
        ignore_patterns=["*.tmp", ".DS_Store"],
    )
    _update_index(api, repo_id, run_id, summary)

    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/runs/{run_id}"
    return {"run_id": run_id, "url": url, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(prog="openmesh-upload")
    ap.add_argument("run_dir", help="Path to a run directory (contains calls.jsonl)")
    ap.add_argument("--repo", help="HF dataset repo id (override HF_DATASET_REPO)")
    args = ap.parse_args()

    cfg = load_config()
    repo_id = args.repo or cfg.hf_dataset_repo
    if not repo_id:
        print("error: HF_DATASET_REPO is not set (or pass --repo)", file=sys.stderr)
        return 2
    if not cfg.hf_token:
        print("error: HF_TOKEN is not set", file=sys.stderr)
        return 2

    result = upload_run(Path(args.run_dir), repo_id=repo_id, hf_token=cfg.hf_token)
    print(f"uploaded {result['run_id']}")
    print(f"  calls:  {result['summary']['total_calls']}")
    print(f"  errors: {result['summary']['total_errors']}")
    print(f"  cost:   ${result['summary']['total_cost_usd']:.4f}")
    print(f"  url:    {result['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

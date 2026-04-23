from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from .accounting import Tracer, trace_session
from .client import TracingClient
from .config import load_config
from .registry import MODELS


def main() -> int:
    ap = argparse.ArgumentParser(prog="openmesh-smoke")
    ap.add_argument("--stream", action="store_true", help="Force streaming on")
    ap.add_argument("--no-stream", action="store_true", help="Force streaming off")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of friendly model names")
    ap.add_argument("--run-id", default=f"smoke-{int(time.time())}")
    ap.add_argument(
        "--prompt",
        default="Reply with exactly the word OK.",
        help="Prompt to send for the smoke test",
    )
    args = ap.parse_args()

    if args.stream and args.no_stream:
        print("error: --stream and --no-stream are mutually exclusive", file=sys.stderr)
        return 2

    cfg = load_config()
    if args.stream:
        cfg.stream = True
    if args.no_stream:
        cfg.stream = False

    run_dir = cfg.traces_dir / args.run_id
    tracer = Tracer(run_dir, cfg)
    client = TracingClient(cfg, tracer)

    remote_ids: set[str] | None
    try:
        r = httpx.get(
            f"{cfg.base_url}/models",
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json().get("data") or []
        remote_ids = {m["id"] for m in data if isinstance(m, dict) and "id" in m}
        print(f"[ /models] OpenMesh reports {len(remote_ids)} model ids")
    except Exception as e:
        print(f"[ /models] WARN: could not list models ({e}); skipping diff")
        remote_ids = None

    names = args.models or list(MODELS)
    ok: list[str] = []
    fail: list[tuple[str, str]] = []

    for name in names:
        if name not in MODELS:
            print(f"[SKIP] {name}: not in registry")
            fail.append((name, "not in registry"))
            continue
        spec = MODELS[name]
        if remote_ids is not None and spec.openmesh_id not in remote_ids:
            print(f"[SKIP] {name} ({spec.openmesh_id}): not listed by OpenMesh /models")
            fail.append((name, "not listed by OpenMesh"))
            continue
        try:
            with trace_session(f"smoke/{name}"):
                resp = client.chat.completions.create(
                    model=name,
                    messages=[{"role": "user", "content": args.prompt}],
                    max_tokens=16,
                    temperature=0,
                )
            choice = (resp.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content") or "").strip()
            usage = resp.get("usage") or {}
            tot = usage.get("total_tokens", "?")
            print(f"[ OK ] {name:<20} tokens={tot!s:<6} reply={content!r}")
            ok.append(name)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            fail.append((name, str(e)))

    print()
    print(f"{len(ok)}/{len(names)} OK   run_dir={run_dir}")
    print(f"spend: run=${tracer.totals.run:.4f}")
    for m, v in sorted(tracer.totals.by_model.items()):
        print(f"  {m:<20} ${v:.4f}")
    print()
    print("Next: open the OpenMesh billing dashboard and confirm these call_ids landed:")
    if tracer.calls_path.exists():
        for line in tracer.calls_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            print(f"  {row.get('model_name'):<20} {row.get('response_id')}  call_id={row.get('call_id')}")

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

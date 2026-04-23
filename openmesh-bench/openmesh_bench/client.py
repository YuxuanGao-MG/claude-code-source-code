from __future__ import annotations

import time
import uuid
from typing import Any

from openai import OpenAI

from .accounting import Tracer, current_session
from .config import Config
from .registry import MODELS, ModelSpec


class _Completions:
    def __init__(self, parent: "TracingClient"):
        self._p = parent

    def create(self, **kwargs):
        return self._p._create(**kwargs)


class _Chat:
    def __init__(self, parent: "TracingClient"):
        self.completions = _Completions(parent)


class TracingClient:
    """OpenAI-compatible client that traces every call and enforces spend caps.

    Call shape mirrors ``openai.OpenAI`` so harnesses that expect
    ``client.chat.completions.create(model=..., messages=...)`` work unchanged —
    except ``model`` is a friendly name from ``registry.MODELS``.
    """

    def __init__(self, config: Config, tracer: Tracer):
        self.config = config
        self.tracer = tracer
        default_headers: dict[str, str] = {}
        if config.app_url:
            default_headers["HTTP-Referer"] = config.app_url
        if config.app_name:
            default_headers["X-Title"] = config.app_name
        self._oai = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout_s,
            max_retries=0,
            default_headers=default_headers or None,
        )
        self.chat = _Chat(self)

    def _resolve(self, model: str) -> ModelSpec:
        spec = MODELS.get(model)
        if spec is None:
            raise KeyError(f"unknown model '{model}'; not in openmesh_bench.registry.MODELS")
        return spec

    def _create(self, *, model: str, messages: list[dict], stream: bool | None = None, **kwargs):
        spec = self._resolve(model)
        use_stream = self.config.stream if stream is None else bool(stream)
        if use_stream and not spec.supports_streaming:
            use_stream = False

        payload: dict[str, Any] = {"model": spec.openmesh_id, "messages": messages, **kwargs}
        if use_stream:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})

        task_id = current_session()
        call_id = str(uuid.uuid4())
        attempts: list[dict] = []

        t0 = time.perf_counter()
        ts_start = time.time()
        result: dict
        chunks: list[dict] | None = None
        ttft_ms: float | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if use_stream:
                    result, chunks, ttft_ms = self._run_stream(payload)
                else:
                    raw = self._oai.chat.completions.create(**payload)
                    result = raw.model_dump()
                break
            except Exception as e:
                attempts.append({"attempt": attempt, "error": repr(e), "ts": time.time()})
                if attempt >= self.config.max_retries:
                    self._log_failure(
                        call_id=call_id, task_id=task_id, model_name=model, spec=spec,
                        payload=payload, use_stream=use_stream, attempts=attempts,
                        ts_start=ts_start, t0=t0,
                    )
                    raise
                time.sleep(min(2 ** attempt, 30))

        t1 = time.perf_counter()

        usage = dict(result.get("usage") or {})
        usage_source = "api"
        if not usage:
            usage = _estimate_usage(payload, result)
            usage_source = "estimated"
        cost_usd = _cost(spec, usage)

        row = {
            "call_id": call_id,
            "task_id": task_id,
            "ts_start": ts_start,
            "ts_end": time.time(),
            "latency_ms": (t1 - t0) * 1000.0,
            "ttft_ms": ttft_ms,
            "model_name": model,
            "openmesh_id": spec.openmesh_id,
            "tier": spec.tier,
            "open_weight": spec.open_weight,
            "stream": use_stream,
            "request": payload,
            "response": result,
            "chunks": chunks,
            "usage": usage,
            "usage_source": usage_source,
            "cost_usd": cost_usd,
            "attempts": attempts,
            "response_id": result.get("id"),
            "upstream_provider": result.get("provider"),
            "error": None,
        }
        self.tracer.log_call(row)
        self.tracer.record_spend(model, cost_usd, task_id)
        return result

    def _run_stream(self, payload: dict) -> tuple[dict, list[dict], float | None]:
        chunks: list[dict] = []
        content_accum: dict[int, str] = {}
        tool_accum: dict[tuple[int, int], dict] = {}
        finish_reason: dict[int, str] = {}
        role_by_idx: dict[int, str] = {}
        assembled: dict[str, Any] = {
            "id": None,
            "model": payload["model"],
            "object": "chat.completion",
            "choices": [],
            "usage": None,
        }

        t_start = time.perf_counter()
        t_first: float | None = None

        stream = self._oai.chat.completions.create(**payload)
        try:
            for chunk in stream:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                d = chunk.model_dump()
                chunks.append({"t_rel_ms": (now - t_start) * 1000.0, "chunk": d})

                if d.get("id") and not assembled["id"]:
                    assembled["id"] = d["id"]

                for ch in d.get("choices") or []:
                    idx = ch.get("index", 0)
                    delta = ch.get("delta") or {}
                    if delta.get("role"):
                        role_by_idx[idx] = delta["role"]
                    if delta.get("content"):
                        content_accum[idx] = content_accum.get(idx, "") + delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        tidx = tc.get("index", 0)
                        slot = tool_accum.setdefault(
                            (idx, tidx),
                            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if tc.get("type"):
                            slot["type"] = tc["type"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        finish_reason[idx] = ch["finish_reason"]

                if d.get("usage"):
                    assembled["usage"] = d["usage"]
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        indices = sorted(
            set(content_accum)
            | set(finish_reason)
            | set(role_by_idx)
            | {i for i, _ in tool_accum}
        )
        for i in indices:
            tool_calls = [
                tool_accum[(ii, tj)]
                for (ii, tj) in sorted(tool_accum)
                if ii == i
            ]
            msg: dict[str, Any] = {
                "role": role_by_idx.get(i, "assistant"),
                "content": content_accum.get(i),
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            assembled["choices"].append(
                {"index": i, "message": msg, "finish_reason": finish_reason.get(i)}
            )

        ttft_ms = (t_first - t_start) * 1000.0 if t_first else None
        return assembled, chunks, ttft_ms

    def _log_failure(self, *, call_id, task_id, model_name, spec, payload, use_stream, attempts, ts_start, t0):
        self.tracer.log_call({
            "call_id": call_id,
            "task_id": task_id,
            "ts_start": ts_start,
            "ts_end": time.time(),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "ttft_ms": None,
            "model_name": model_name,
            "openmesh_id": spec.openmesh_id,
            "tier": spec.tier,
            "open_weight": spec.open_weight,
            "stream": use_stream,
            "request": payload,
            "response": None,
            "chunks": None,
            "usage": None,
            "usage_source": None,
            "cost_usd": 0.0,
            "attempts": attempts,
            "response_id": None,
            "error": attempts[-1]["error"] if attempts else "unknown",
        })


def _cost(spec: ModelSpec, usage: dict) -> float:
    in_tok = usage.get("prompt_tokens") or 0
    out_tok = usage.get("completion_tokens") or 0
    return (in_tok * spec.input_usd_per_mtok + out_tok * spec.output_usd_per_mtok) / 1_000_000.0


def _estimate_usage(request: dict, response: dict) -> dict:
    """Fallback when the gateway strips `usage`. 4 chars/token rule of thumb."""
    def toks(s: str | None) -> int:
        if not s:
            return 0
        return max(1, len(s) // 4)

    prompt = 0
    for m in request.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            prompt += toks(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    prompt += toks(part["text"])

    completion = 0
    for ch in response.get("choices") or []:
        msg = ch.get("message") or {}
        if isinstance(msg.get("content"), str):
            completion += toks(msg["content"])
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            completion += toks(fn.get("arguments"))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }

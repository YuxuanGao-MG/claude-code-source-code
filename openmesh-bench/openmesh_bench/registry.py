"""Benchmark-model registry.

`openmesh_id` values are OpenRouter model slugs. Verify them with
``openmesh-smoke`` (which diffs against OpenRouter's ``/models`` endpoint)
before any billed run. Prices are OpenRouter list prices in USD per million
tokens; the spend guardrails read these directly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    openmesh_id: str
    tier: str  # "frontier" | "mid"
    open_weight: bool
    context_window: int
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    supports_streaming: bool = True


def _m(**kw) -> ModelSpec:
    return ModelSpec(**kw)


# The 8-model benchmark set (4 frontier + 4 mid-tier), routed via OpenRouter.
MODELS: dict[str, ModelSpec] = {
    # Frontier
    "gpt-5.4":           _m(name="gpt-5.4",           openmesh_id="openai/gpt-5.4",             tier="frontier", open_weight=False, context_window=400_000,   input_usd_per_mtok=15.0, output_usd_per_mtok=75.0),
    "claude-opus-4.7":   _m(name="claude-opus-4.7",   openmesh_id="anthropic/claude-opus-4.7",  tier="frontier", open_weight=False, context_window=1_000_000, input_usd_per_mtok=15.0, output_usd_per_mtok=75.0),
    "gemini-3.1-pro":    _m(name="gemini-3.1-pro",    openmesh_id="google/gemini-3.1-pro",      tier="frontier", open_weight=False, context_window=2_000_000, input_usd_per_mtok=10.0, output_usd_per_mtok=40.0),
    "kimi-k2.6":         _m(name="kimi-k2.6",         openmesh_id="moonshotai/kimi-k2.6",       tier="frontier", open_weight=True,  context_window=200_000,   input_usd_per_mtok=2.0,  output_usd_per_mtok=8.0),

    # Mid-tier
    "claude-sonnet-4.6": _m(name="claude-sonnet-4.6", openmesh_id="anthropic/claude-sonnet-4.6", tier="mid", open_weight=False, context_window=1_000_000, input_usd_per_mtok=3.0, output_usd_per_mtok=15.0),
    "gemini-3-flash":    _m(name="gemini-3-flash",    openmesh_id="google/gemini-3-flash",       tier="mid", open_weight=False, context_window=1_000_000, input_usd_per_mtok=0.3, output_usd_per_mtok=2.5),
    "qwen3.6-plus":      _m(name="qwen3.6-plus",      openmesh_id="qwen/qwen3.6-plus",           tier="mid", open_weight=True,  context_window=128_000,   input_usd_per_mtok=1.0, output_usd_per_mtok=4.0),
    "minimax-m2.5":      _m(name="minimax-m2.5",      openmesh_id="minimax/minimax-m2.5",        tier="mid", open_weight=True,  context_window=200_000,   input_usd_per_mtok=1.0, output_usd_per_mtok=4.0),
}


def by_tier(tier: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.tier == tier]

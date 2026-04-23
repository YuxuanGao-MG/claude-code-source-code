"""Benchmark-model registry.

`openmesh_id` and the per-million-token prices below are PLACEHOLDERS. Replace
them with the real values from your OpenMesh dashboard before running for real
money — the spend guardrails depend on `input_usd_per_mtok` / `output_usd_per_mtok`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    openmesh_id: str
    tier: str  # "frontier" | "mid" | "small"
    open_weight: bool
    context_window: int
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    supports_streaming: bool = True


def _m(**kw) -> ModelSpec:
    return ModelSpec(**kw)


MODELS: dict[str, ModelSpec] = {
    # Frontier (agent) — closed-source
    "gpt-5.4":           _m(name="gpt-5.4",           openmesh_id="gpt-5.4",            tier="frontier", open_weight=False, context_window=400_000, input_usd_per_mtok=15.0, output_usd_per_mtok=75.0),
    "claude-opus-4.7":   _m(name="claude-opus-4.7",   openmesh_id="claude-opus-4-7",    tier="frontier", open_weight=False, context_window=1_000_000, input_usd_per_mtok=15.0, output_usd_per_mtok=75.0),
    "gemini-3.1-pro":    _m(name="gemini-3.1-pro",    openmesh_id="gemini-3.1-pro",     tier="frontier", open_weight=False, context_window=2_000_000, input_usd_per_mtok=10.0, output_usd_per_mtok=40.0),

    # Frontier (agent) — open-weight
    "glm-5":             _m(name="glm-5",             openmesh_id="zai/glm-5",          tier="frontier", open_weight=True,  context_window=200_000, input_usd_per_mtok=2.0,  output_usd_per_mtok=8.0),
    "kimi-k2.5":         _m(name="kimi-k2.5",         openmesh_id="moonshot/kimi-k2.5", tier="frontier", open_weight=True,  context_window=200_000, input_usd_per_mtok=2.0,  output_usd_per_mtok=8.0),

    # Mid-tier (agent) — closed-source
    "claude-sonnet-4.6": _m(name="claude-sonnet-4.6", openmesh_id="claude-sonnet-4-6",  tier="mid",      open_weight=False, context_window=1_000_000, input_usd_per_mtok=3.0,  output_usd_per_mtok=15.0),
    "gemini-3-flash":    _m(name="gemini-3-flash",    openmesh_id="gemini-3-flash",     tier="mid",      open_weight=False, context_window=1_000_000, input_usd_per_mtok=0.3,  output_usd_per_mtok=2.5),

    # Mid-tier (agent) — open-weight
    "qwen3.6-plus":      _m(name="qwen3.6-plus",      openmesh_id="qwen/qwen3.6-plus",  tier="mid",      open_weight=True,  context_window=128_000, input_usd_per_mtok=1.0,  output_usd_per_mtok=4.0),
    "minimax-m2.5":      _m(name="minimax-m2.5",      openmesh_id="minimax/m2.5",       tier="mid",      open_weight=True,  context_window=200_000, input_usd_per_mtok=1.0,  output_usd_per_mtok=4.0),

    # Small (worker-only) — open-weight
    "qwen3-coder-30b":   _m(name="qwen3-coder-30b",   openmesh_id="qwen/qwen3-coder-30b",    tier="small", open_weight=True, context_window=128_000, input_usd_per_mtok=0.3, output_usd_per_mtok=1.0),
    "glm-4.7-flash":     _m(name="glm-4.7-flash",     openmesh_id="zai/glm-4.7-flash",       tier="small", open_weight=True, context_window=128_000, input_usd_per_mtok=0.3, output_usd_per_mtok=1.0),
    "nemotron-nano-30b": _m(name="nemotron-nano-30b", openmesh_id="nvidia/nemotron-nano-30b", tier="small", open_weight=True, context_window=128_000, input_usd_per_mtok=0.2, output_usd_per_mtok=0.8),
    "gemma-4-26b":       _m(name="gemma-4-26b",       openmesh_id="google/gemma-4-26b",      tier="small", open_weight=True, context_window=128_000, input_usd_per_mtok=0.2, output_usd_per_mtok=0.8),
}


def by_tier(tier: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.tier == tier]

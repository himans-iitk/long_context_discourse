#!/usr/bin/env python3
"""Print API call counts and rough wall-time / USD brackets for the full paid sweep.

Mirrors configs/exp{1,2,4,5}.yaml as of the EMNLP reference implementation.
Rates are **illustrative** — verify live pricing at https://openrouter.ai/models
before budgeting (per-token pricing moves frequently).

Usage:
  python scripts/estimate_full_paid_sweep.py
"""

from __future__ import annotations

# --- Constants aligned with YAML + code paths ---

N_PDTB_TEST = 400
N_PRESUP = 500
N_LANGS = 6
PER_LANG_PER_TYPE = 50  # exp4.yaml run.per_lang_per_type
N_EXP4_EXAMPLES = N_LANGS * PER_LANG_PER_TYPE * 2  # Implicit + Explicit per language

EXP1_LENGTHS = [512, 2048, 8192, 32768, 65536]
LONG_CTX_MODELS = {
    "gpt4o",
    "deepseek_r1",
    "llama3_70b",
}
ALL_CHAT_MODELS = [
    "gpt4o",
    "deepseek_r1",
    "llama3_70b",
    "llama3_8b",
    "mistral_7b",
    "mixtral",
    "phi4",
    "deepseek_v3",
]

EXP4_MODELS = [
    "gpt4o",
    "llama3_70b",
    "mistral_7b",
    "deepseek_v3",
]
EXP4_LENGTHS = [512, 8192, 32768]

# Exp 2B: 100 examples at distance 20 x 4 marker conditions x N models
EXP2B_DIST20_N = 100
MARKER_CONDITIONS = 4


def _exp1_calls() -> int:
    total = 0
    for m in ALL_CHAT_MODELS:
        lens = [
            L
            for L in EXP1_LENGTHS
            if L <= 8192 or m in LONG_CTX_MODELS
        ]
        total += N_PDTB_TEST * len(lens)
    return total


def _exp2_calls() -> int:
    # Each example: one target model + one gpt-4o-mini judge
    return N_PRESUP * len(ALL_CHAT_MODELS) * 2


def _exp2b_calls() -> int:
    return EXP2B_DIST20_N * MARKER_CONDITIONS * len(ALL_CHAT_MODELS) * 2


def _exp4_calls() -> int:
    total = 0
    for m in EXP4_MODELS:
        lens = [L for L in EXP4_LENGTHS if L <= 8192 or m in LONG_CTX_MODELS]
        total += N_EXP4_EXAMPLES * len(lens)
    return total


def _exp5_calls() -> int:
    """CoT completions: same eight models × lengths policy as Exp 1 (skip long ctx if unsupported)."""
    n = 0
    for m in ALL_CHAT_MODELS:
        lens = [L for L in EXP1_LENGTHS if L <= 8192 or m in LONG_CTX_MODELS]
        n += N_PDTB_TEST * len(lens)
    return n


def main() -> None:
    e1 = _exp1_calls()
    e2 = _exp2_calls()
    e2b = _exp2b_calls()
    e4 = _exp4_calls()
    e5 = _exp5_calls()
    chat_total = e1 + e2 + e2b + e4 + e5

    sleep = 0.2  # paid_full_base rate_limit_sleep_seconds (approximate lower bound)
    avg_turn_s = 4.0  # blended guess: network + queue + generation for mixed lengths

    wall_serial_sec = chat_total * (sleep + avg_turn_s)

    # Illustrative $ — dominated by long-context frontier calls; wide bracket.
    low_per_call = 0.0008
    high_per_call = 0.08

    n_models = len(ALL_CHAT_MODELS)
    print("Full paid sweep (Exp 1, 2, 2B, 4, 5) — OpenRouter chat calls")
    print(f"  (Anthropic Claude excluded; {n_models} target models in registry)")
    print(f"  Exp 1 (PDTB):     {e1:,}")
    print(f"  Exp 2 (presup):   {e2:,}  (includes judge)")
    print(f"  Exp 2B (markers): {e2b:,}  (includes judge)")
    print(f"  Exp 4 (TED-MDB):  {e4:,}")
    print(f"  Exp 5 (CoT):      {e5:,}")
    print("  -----------------------------------------")
    print(f"  Total API calls:  {chat_total:,}")
    print()
    print(
        "Rough wall time (single serial worker, ~"
        f"{sleep}s sleep + ~{avg_turn_s}s avg latency placeholder):"
    )
    print(f"  ~{wall_serial_sec / 3600:.1f} hours ({wall_serial_sec / 86400:.2f} days)")
    print("  Add retries, long 65k rows, and provider variance; often 1.5-3x longer.")
    print()
    ideal_parallel = wall_serial_sec / max(1, n_models)
    print(
        "Illustrative parallel floor (perfect "
        f"{n_models}-way overlap, neglecting merge/analyze): ~{ideal_parallel / 3600:.1f} h"
    )
    print()
    print("Rough USD (very wide bracket - verify OpenRouter model cards):")
    print(f"  {chat_total * low_per_call:,.0f} USD  …  {chat_total * high_per_call:,.0f} USD")
    print("  Upper bound can exceed this if many ultra-long prompts hit premium tiers.")
    print()
    print("Models (short_name → OpenRouter ID) — see configs/default.yaml:")
    pairs = [
        ("gpt4o", "openai/gpt-4o", "closed"),
        ("deepseek_r1", "deepseek/deepseek-r1", "open weights/API"),
        ("deepseek_v3", "deepseek/deepseek-chat-v3.1", "open weights/API"),
        ("llama3_70b", "meta-llama/llama-3.3-70b-instruct", "open weights"),
        ("llama3_8b", "meta-llama/llama-3.1-8b-instruct", "open weights"),
        ("mistral_7b", "mistralai/mistral-7b-instruct-v0.1", "open weights"),
        ("mixtral", "mistralai/mixtral-8x22b-instruct", "open weights"),
        ("phi4", "microsoft/phi-4", "open weights"),
    ]
    for short, oid, tag in pairs:
        print(f"  {short:14} {oid:42} [{tag}]")
    print("  Judge (Exp 2): openai/gpt-4o-mini")


if __name__ == "__main__":
    main()

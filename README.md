# long-context-discourse

Reference implementation of the five experiments described in
*"Coherence at Scale: A Survey of Discourse and Pragmatic Phenomena in Long-Context
Large Language Models"* (EMNLP 2026 submission).

| # | Experiment | What it measures |
|---|---|---|
| 1 | Implicit vs. explicit discourse relation degradation (PDTB 3.0) | Macro‑F1 vs. context length |
| 2 | Presupposition tracking across conversational distance | False‑Presupposition Acceptance Rate (FPAR) |
| 2b | Discourse‑marker rescue ablation | FPAR under no/weak/strong/repeated markers |
| 3 | Mechanistic probing of discourse roles (GUM) | Linear probe accuracy by layer and document position |
| 4 | Cross‑lingual discourse degradation (TED‑MDB) | Macro‑F1 by language and length |
| 5 | Reasoning vs. standard models (+ CoT prompt) | Per‑condition macro‑F1 |

## Layout

```
src/long_context_discourse/   library code (importable)
configs/                      YAML configs per experiment
scripts/                      thin CLI wrappers around library entry points
tests/                        pytest unit tests for pure logic
results/                      run outputs (mostly git-ignored; demo bundle optional)
figures/                      generated figures (demo PNGs tracked as `*_demo.png`)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Optional: GPU probing for Experiment 3.
pip install -e ".[probe]"
```

Copy `.env.example` to `.env` and fill in keys. Loaders use `python-dotenv`.

## Data

Experiments expect the local repository
`Research/long_context_discourse_dataset/` (PDTB 3.0, GUM, TED-MDB) and the
synthetic presupposition file
`long_context_discourse_dataset/SyntheticDataset/presupposition_dataset.json`.
Override paths in `configs/*.yaml` or via CLI flags.

### Preprocessing raw corpora → JSON

The experiment loaders expect already-converted JSON; the converters live
under `scripts/`. Each accepts a `--smoke N` flag for a quick subset run:

```bash
# PDTB 3.0 → train pool + balanced section-23 test set
python scripts/prepare_pdtb.py \
    --pdtb-root /Volumes/LDC2019T05/PDTB-3.0 \
    --out-dir   ../../long_context_discourse_dataset/processed/pdtb

# GUM RST → segment-level JSON for the probe
python scripts/prepare_gum.py \
    --rs4-dir  ../../long_context_discourse_dataset/GUM/rst/rstweb \
    --out-path ../../long_context_discourse_dataset/processed/exp3/gum_rst_processed.json

# TED-MDB → cross-lingual records
python scripts/prepare_ted_mdb.py \
    --ted-root ../../long_context_discourse_dataset/Ted-MDB \
    --out-path ../../long_context_discourse_dataset/processed/exp4/ted_mdb_processed.json
```

Conversion is fast (PDTB ≈ 2 s, GUM ≈ 0.5 s, TED-MDB ≈ 0.2 s on local disk).

## Running

Configs **`exp1.yaml`**, **`exp2.yaml`**, **`exp4.yaml`**, and **`exp5.yaml`**
extend **`configs/paid_full_base.yaml`**, which sets longer HTTP timeouts and a
small inter-call delay for stable paid OpenRouter runs (non-`:free` model IDs
from `configs/default.yaml`). **`exp3.yaml`** still extends `default.yaml` only
(local GPU / HF weights).

```bash
# Each experiment is a "run" stage (calls models / probes) followed by an
# "analyze" stage (computes metrics, writes figures and LaTeX tables).
python -m long_context_discourse.scripts_entry run_exp1 --config configs/exp1.yaml
python -m long_context_discourse.scripts_entry analyze_exp1 --config configs/exp1.yaml

python -m long_context_discourse.scripts_entry run_exp2 --config configs/exp2.yaml
python -m long_context_discourse.scripts_entry run_exp2b --config configs/exp2.yaml
python -m long_context_discourse.scripts_entry analyze_exp2 --config configs/exp2.yaml

python -m long_context_discourse.scripts_entry run_exp3 --config configs/exp3.yaml
python -m long_context_discourse.scripts_entry analyze_exp3 --config configs/exp3.yaml

python -m long_context_discourse.scripts_entry run_exp4 --config configs/exp4.yaml
python -m long_context_discourse.scripts_entry analyze_exp4 --config configs/exp4.yaml

python -m long_context_discourse.scripts_entry run_exp5 --config configs/exp5.yaml
python -m long_context_discourse.scripts_entry analyze_exp5 --config configs/exp5.yaml

python -m long_context_discourse.scripts_entry compile_master_results \
  --config configs/exp1.yaml --output results/MASTER_FULL_PAID.json
```

All runs are deterministic (`temperature=0`, fixed `random.seed(42)` /
`numpy.random.default_rng(42)`). Long sweeps checkpoint to
`results/<exp>/checkpoints/`.

### Full paid OpenRouter sweep (Experiments 1, 2, 2B, 4, 5 — **not** Exp 3)

End-to-end API workflow for the EMNLP-scale evaluation: **11 chat models**
(closed + open-weight routes via OpenRouter; **Anthropic Claude is not included**),
**PDTB** balanced test (**400**
examples), full **presupposition** set (**500**), **TED-MDB** cross-lingual
subset (**600** pairs), marker **ablation at distance 20** (**100** examples × **11**
models × **4** conditions), and **CoT** vs standard (**Exp 5** on the **same eight**
registry chat models as **Exp 1**; context lengths above **8192** are skipped when the
model is not listed under `long_context_models`, matching Exp 1).

```bash
cd Projects/long_context_discourse
chmod +x scripts/run_full_paid_api_experiments.sh   # once
bash scripts/run_full_paid_api_experiments.sh
```

This writes `results/MASTER_FULL_PAID.json` (with `experiment_3` left `null`
until you run the local probe).

**Call-count / time / money (approximate):**

```bash
python scripts/estimate_full_paid_sweep.py
```

Typical output is on the order of **~48k** OpenRouter chat completions (including
**gpt-4o-mini** judge calls in Exp 2 / 2B), **~2–5+ days** wall time on a **single
serial** worker depending on latency and retries, and **roughly tens to a few
thousand USD** depending on how many rows hit **65k**/**32k** context with frontier
pricing — **always** reconcile with live rates at
[OpenRouter models](https://openrouter.ai/models).

### Parallel paid sweep (per-model shards)

From ``Projects/long_context_discourse``:

```bash
cd Projects/long_context_discourse
chmod +x scripts/run_full_paid_parallel.sh    # once
MAX_PARALLEL=11 bash scripts/run_full_paid_parallel.sh
# Optional: ENV_FILE=/path/to/.env  NO_RESUME=1  (force re-run all shards)
```

Resume: re-run the **same** command after an interruption; finished shards are skipped automatically (see ``shard_resume``). **Within one shard**, runs normally write the final JSON only at the end — if that file is missing, the shard starts over on the next attempt (checkpoints under ``results/.../checkpoints/`` are not auto-resumed yet).

Shards follow **Exp 4’s** ``models_subset`` and **Exp 5’s** ``cot_models``. Each stage regenerates YAML under ``configs/parallel/shards/``, runs ``run_parallel_stage.py`` (resume on by default), then merges and analyzes.


```bash
python scripts/merge_parallel_results.py --project-root . --all
```

Shard directories remain under ``results/exp*__shard_*/`` for debugging.

**Exp 4 outputs:** `exp4_summary_f1.json` includes a **`model`** column;
`exp4_degradation_per_language.json` is nested as
`language → model → relation_type → percent_change`.

Per-model shard configs for the full sweep live under **`configs/parallel/shards/`**
(regenerate with `python scripts/generate_parallel_shard_configs.py`).

### Synthetic demo outputs (offline layout / plots)

To populate **`results/exp{1,2,4,5}/`** and **`results/MASTER_DEMO.json`** with
JSON that uses the **same filenames** as analyze stages (plus **`figures/*_demo.png`**),
without calling APIs:

```bash
cd Projects/long_context_discourse
python scripts/bootstrap_demo_pipeline_outputs.py --project-root .
```

Numbers are illustrative only; replace with real **`run_*` / `analyze_*`** outputs for publication.

## Conventions

- **Library code** lives under `src/long_context_discourse`. Scripts are thin.
- **Type hints** everywhere; `from __future__ import annotations` enabled per file.
- **Logging** via stdlib `logging`; no print in library code.
- **Config** is a frozen dataclass loaded from YAML; CLI flags override fields.
- **Tests** cover pure logic (parsing, padding, metrics, conversation building).
  Network-bound modules are tested via tiny fakes.

## License

MIT, see `pyproject.toml`.

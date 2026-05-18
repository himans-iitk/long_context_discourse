#!/usr/bin/env bash
# Full paid OpenRouter sweep — Experiments 1, 2, 2B, 4, 5 (configs/exp*.yaml).
# Experiment 3 (local Llama/GPU probing) is intentionally skipped.
#
# Order: Exp 1 must finish before Exp 5 (CoT reuses upstream exp1 rows).
#
# Usage (from project root):
#   bash scripts/run_full_paid_api_experiments.sh
#   ENV_FILE=/path/to/.env bash scripts/run_full_paid_api_experiments.sh
#
# Estimates: python scripts/estimate_full_paid_sweep.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
CFG="${ROOT}/configs"
ENV="${ENV_FILE:-${ROOT}/.env}"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv Python: $PY — create .venv and pip install -e '.[dev]' first." >&2
  exit 1
fi

run() {
  echo "=== $1 ==="
  "$PY" -m long_context_discourse.scripts_entry "$@"
}

cd "$ROOT"

run run_exp1 --config "${CFG}/exp1.yaml" --env "$ENV"
run analyze_exp1 --config "${CFG}/exp1.yaml" --env "$ENV"

run run_exp2 --config "${CFG}/exp2.yaml" --env "$ENV"
run run_exp2b --config "${CFG}/exp2.yaml" --env "$ENV"
run analyze_exp2 --config "${CFG}/exp2.yaml" --env "$ENV"

run run_exp4 --config "${CFG}/exp4.yaml" --env "$ENV"
run analyze_exp4 --config "${CFG}/exp4.yaml" --env "$ENV"

run run_exp5 --config "${CFG}/exp5.yaml" --env "$ENV"
run analyze_exp5 --config "${CFG}/exp5.yaml" --env "$ENV"

run compile_master_results --config "${CFG}/exp1.yaml" \
  --output "${ROOT}/results/MASTER_FULL_PAID.json"

echo "Done. Results under ${ROOT}/results/{exp1,exp2,exp4,exp5}/"
echo "Master JSON: ${ROOT}/results/MASTER_FULL_PAID.json"

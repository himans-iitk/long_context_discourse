#!/usr/bin/env bash
# Full paid OpenRouter sweep with per-model parallel workers + merge.
# Skips Experiment 3 (local probe).
#
# Env:
#   ENV_FILE      Path to .env (default: project_root/.env)
#   MAX_PARALLEL  Concurrent shards per stage (default: 11)
#   NO_RESUME     Set to 1 to force re-running every shard (--no-resume)
#
# Resume: parallel stages skip shards whose output JSON already has the expected
# row count. Re-run this same script after a crash to continue. (Within one
# shard, a partial run usually leaves no final JSON — that shard restarts from
# scratch.) Use NO_RESUME=1 to redo all API calls.
#
# Usage:
#   bash scripts/run_full_paid_parallel.sh
#   MAX_PARALLEL=6 ENV_FILE=/path/.env bash scripts/run_full_paid_parallel.sh
#   NO_RESUME=1 bash scripts/run_full_paid_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
CFG="${ROOT}/configs"
SHARDS="${CFG}/parallel/shards"
ENV="${ENV_FILE:-${ROOT}/.env}"
MAX_PARALLEL="${MAX_PARALLEL:-11}"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv Python: $PY — create .venv and pip install -e '.[dev]' first." >&2
  exit 1
fi

run_entry() {
  echo "=== $* ==="
  "$PY" -m long_context_discourse.scripts_entry "$@"
}

# Build argv without empty-array expansion (breaks `set -u` on some Bash builds).
run_parallel_stage() {
  local stage="$1"
  local -a cmd=(
    "$PY" "${ROOT}/scripts/run_parallel_stage.py"
    --project-root "$ROOT"
    --shards-dir "$SHARDS"
    --stage "$stage"
    --max-workers "$MAX_PARALLEL"
  )
  if [[ "${NO_RESUME:-}" == "1" ]]; then
    cmd+=(--no-resume)
  fi
  if [[ -f "$ENV" ]]; then
    cmd+=(--env-file "$ENV")
  fi
  "${cmd[@]}"
}

cd "$ROOT"

echo "=== generate shard YAMLs ==="
"$PY" "${ROOT}/scripts/generate_parallel_shard_configs.py" \
  --project-root "$ROOT" --output-dir "$SHARDS"

echo "=== parallel exp1 ==="
run_parallel_stage exp1

echo "=== merge exp1 ==="
"$PY" "${ROOT}/scripts/merge_parallel_results.py" --project-root "$ROOT" --experiment exp1
run_entry analyze_exp1 --config "${CFG}/exp1.yaml" --env "$ENV"

echo "=== parallel exp2 + exp2b (per shard) ==="
run_parallel_stage exp2_pair

echo "=== merge exp2 / exp2b ==="
"$PY" "${ROOT}/scripts/merge_parallel_results.py" --project-root "$ROOT" --experiment exp2
"$PY" "${ROOT}/scripts/merge_parallel_results.py" --project-root "$ROOT" --experiment exp2b
run_entry analyze_exp2 --config "${CFG}/exp2.yaml" --env "$ENV"

echo "=== parallel exp4 ==="
run_parallel_stage exp4

echo "=== merge exp4 ==="
"$PY" "${ROOT}/scripts/merge_parallel_results.py" --project-root "$ROOT" --experiment exp4
run_entry analyze_exp4 --config "${CFG}/exp4.yaml" --env "$ENV"

echo "=== parallel exp5 ==="
run_parallel_stage exp5

echo "=== merge exp5 ==="
"$PY" "${ROOT}/scripts/merge_parallel_results.py" --project-root "$ROOT" --experiment exp5
run_entry analyze_exp5 --config "${CFG}/exp5.yaml" --env "$ENV"

run_entry compile_master_results --config "${CFG}/exp1.yaml" \
  --output "${ROOT}/results/MASTER_FULL_PAID.json"

echo "Done. Canonical outputs: ${ROOT}/results/{exp1,exp2,exp4,exp5}/"
echo "Shard dirs: ${ROOT}/results/exp*__shard_*/"
echo "Master JSON: ${ROOT}/results/MASTER_FULL_PAID.json"

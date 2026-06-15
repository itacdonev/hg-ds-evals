#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  hg_ds_evals/evals/run.sh \
    --run-id <RUN_ID> [<RUN_ID> ...] \
    --report-name <REPORT_NAME> \
    --run-judge \
    --delete-previous-checkpoint

Example:
  hg_ds_evals/evals/run.sh \
    --run-id 0d71e994b97a49c684df563317288cab 12484008b18f42548728d35987a6c6a7 \
    --report-name pr_620 \
    --run-judge \
    --delete-previous-checkpoint
USAGE
  exit 2
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m hg_ds_evals.evals.api_eval_runner "$@"

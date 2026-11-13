#!/usr/bin/env bash
# Launch one shard of the board tournament on a GCP VM.
# Usage on the VM, from the repo root:
#   bash tools/tournament_board_gcp.sh 0 4
set -euo pipefail

SHARD_INDEX="${1:?shard index}"
SHARD_COUNT="${2:?shard count}"
WORKERS="${WORKERS:-$(python3 -c 'import os; print(os.cpu_count() or 1)')}"
OUT="${OUT:-results/tournament_board}"

cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then
  PY=(uv run)
else
  PY=(python3)
fi

"${PY[@]}" tools/tournament_board.py \
  --shard "${SHARD_INDEX}/${SHARD_COUNT}" \
  --workers "${WORKERS}" \
  --out "${OUT}" \
  --write-inventory

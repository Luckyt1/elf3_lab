#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir:$project_dir/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mode="${1:-train}"
if (( $# > 0 )); then shift; fi
case "$mode" in
  train)    script=legged_lab/scripts/train.py; set -- --task stand_gl165 "$@" ;;
  evaluate) script=legged_lab/scripts/evaluate_gl165.py ;;
  mujoco)   script=legged_lab/scripts/play_mujoco.py ;;
  *) echo "Usage: scripts/gl165.sh {train|evaluate|mujoco} [arguments...]" >&2; exit 2 ;;
esac
exec "${GL165_PYTHON:-python3}" -B "$script" "$@"

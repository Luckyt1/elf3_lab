#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${GL165_PYTHON:-$project_dir/.venv-gl165/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf '%s\n' 'Set GL165_PYTHON to a Python interpreter with Isaac Lab 2.1 / Isaac Sim 4.5 installed.' >&2
    exit 1
fi
# Prefer this checkout over editable installs belonging to other projects.
export PYTHONPATH="$project_dir:$project_dir/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
mode="${1:-train}"
if (( $# > 0 )); then shift; fi
case "$mode" in
    train) exec "$python_bin" legged_lab/scripts/train.py --task stand_gl165 "$@" ;;
    play) exec "$python_bin" legged_lab/scripts/play_ppo.py --task stand_gl165 "$@" ;;
    evaluate) exec "$python_bin" legged_lab/scripts/evaluate_gl165.py "$@" ;;
    preview) exec "$python_bin" legged_lab/scripts/preview_gl165_ros.py "$@" ;;
    *) printf '%s\n' 'Usage: scripts/gl165.sh {train|play|evaluate|preview} [arguments...]' >&2; exit 2 ;;
esac

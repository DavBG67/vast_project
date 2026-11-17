#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  exec python process.py --help
fi

exec python process.py "$@"

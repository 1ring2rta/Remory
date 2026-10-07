#!/usr/bin/env bash
set -euo pipefail
remory_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$remory_root/deploy/start.py" "$@"

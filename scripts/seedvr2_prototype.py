from __future__ import annotations

import sys

from vast_upscaler.seedvr2_prototype import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))


"""Compatibility entry point for the original install-and-start shortcut."""
import argparse
import os
from pathlib import Path

from install import ROOT, prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".remory")
    parser.add_argument("--cuda", choices=["auto", "cu130"], default="auto")
    parser.add_argument("--install-only", action="store_true")
    args, extra = parser.parse_known_args()
    state, python = prepare_runtime(args.runtime_dir, args.cuda)
    if args.install_only:
        print(f"Runtime ready: {python}")
        return
    os.execve(python, [str(python), "-m", "remory.launch_server",
                       "--runtime-dir", str(state), *extra], os.environ.copy())


if __name__ == "__main__":
    main()

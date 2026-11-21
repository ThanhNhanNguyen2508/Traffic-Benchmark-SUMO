#!/usr/bin/env python3
"""Launch SUMO GUI for a traffic-density scenario."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMO_DIR = ROOT / "sumo"
SCENARIOS = SUMO_DIR / "scenarios"
MODES = ("light", "medium", "heavy", "test-heavy")
SPLITS = ("train", "val", "test")


def find_sumo_gui() -> str:
    """Resolve sumo-gui from PATH or SUMO_HOME."""
    found = shutil.which("sumo-gui")
    if found:
        return found

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        for name in ("sumo-gui.exe", "sumo-gui"):
            candidate = Path(sumo_home) / "bin" / name
            if candidate.is_file():
                return str(candidate)

    raise FileNotFoundError(
        "sumo-gui not found. Install SUMO and add it to PATH, "
        "or set the SUMO_HOME environment variable."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SUMO GUI with a light/medium/heavy traffic scenario."
    )
    parser.add_argument(
        "-mode",
        "--mode",
        dest="mode",
        required=True,
        choices=MODES,
        help="Traffic density mode",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=SPLITS,
        help="Scenario split (fixed seeds: train 1001, val 2002, test 3003). Default: test",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use sumo (no GUI) instead of sumo-gui",
    )
    parser.add_argument(
        "sumo_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to SUMO after '--'",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_dir = (SCENARIOS / "stress" / "test-heavy" if args.mode == "test-heavy"
                    else SCENARIOS / args.split / args.mode)
    config_file = scenario_dir / "osm.sumocfg"

    if not config_file.is_file():
        print(f"Config not found: {config_file}", file=sys.stderr)
        return 1

    try:
        if args.headless:
            binary = shutil.which("sumo")
            if not binary:
                sumo_home = os.environ.get("SUMO_HOME")
                if sumo_home:
                    for name in ("sumo.exe", "sumo"):
                        candidate = Path(sumo_home) / "bin" / name
                        if candidate.is_file():
                            binary = str(candidate)
                            break
            if not binary:
                raise FileNotFoundError("sumo not found (headless mode).")
        else:
            binary = find_sumo_gui()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    extra = args.sumo_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [binary, "-c", str(config_file), *extra]
    print(f"Split: {args.split}  Mode: {args.mode}")
    print(f"Config: {config_file}")
    print(f"Command: {' '.join(cmd)}")

    # Run from scenario dir so edgeData/tripinfos/stats write next to the trips.
    completed = subprocess.run(cmd, cwd=scenario_dir)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

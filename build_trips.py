#!/usr/bin/env python3
"""Regenerate random trips for a traffic-density mode on the study network.

Densities are unchanged from the original setup. Pedestrians are no longer
generated. Motorcycles get a sublane vType (they ride side by side and filter);
other vehicle types keep SUMO defaults.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMO_DIR = ROOT / "sumo"
MAP_DIR = SUMO_DIR / "map"
SCENARIOS = SUMO_DIR / "scenarios"

# Base densities (light). medium = 2x, heavy = 3x
DENSITY_SCALE = {"light": 1, "medium": 2, "heavy": 3, "test-heavy": 12}  # test-heavy: stress test, oversaturated
BASE_DENSITY = {
    "bicycle": 1,
    "bus": 1,
    "motorcycle": 24,
    "passenger": 4,
    "truck": 1,
}

# Motorcycle behaviour (requires --lateral-resolution in the .sumocfg).
# Starting values; to be calibrated against camera data.
MOTORCYCLE_VTYPE = {
    "length": "1.9", "width": "0.7", "minGap": "0.5", "minGapLat": "0.3",
    "maxSpeed": "13.9", "accel": "2.5", "decel": "5.0", "sigma": "0.5", "tau": "1.0",   # tau >= step length (1 s) to avoid rear-end collisions
    "latAlignment": "arbitrary", "maxSpeedLat": "1.5",
    "lcSublane": "5", "lcPushy": "0.6", "lcAssertive": "1.5",
}


def random_trips() -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise FileNotFoundError("SUMO_HOME is not set.")
    script = Path(sumo_home) / "tools" / "randomTrips.py"
    if not script.is_file():
        raise FileNotFoundError(f"randomTrips.py not found: {script}")
    return script


def set_motorcycle_vtype(trips: Path) -> None:
    """Add the sublane attributes to the motorcycle vType written by randomTrips."""
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(trips)
    for vt in tree.getroot().iter("vType"):
        if vt.get("vClass") == "motorcycle":
            for k, v in MOTORCYCLE_VTYPE.items():
                vt.set(k, v)
    tree.write(trips, encoding="UTF-8", xml_declaration=True)


def run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build(mode: str) -> None:
    scale = DENSITY_SCALE[mode]
    # benchmark scenarios come from build_demand.py; this script now only builds the random stress test
    # (sumo/scenarios/stress/test-heavy) and, for comparison, random-density versions under sumo/scenarios/random/
    scenario = SCENARIOS / ("stress" if mode == "test-heavy" else "random") / mode
    scenario.mkdir(parents=True, exist_ok=True)
    net = MAP_DIR / "hcmc.net.xml.gz"
    tools = str(random_trips())
    py = sys.executable

    via = (
        "highway.motorway,highway.motorway_link,highway.trunk_link,"
        "highway.primary_link,highway.secondary_link,highway.tertiary_link"
    )
    common_vehicle = [
        "--trip-attributes", 'departLane="best"',
        "--fringe-start-attributes", 'departSpeed="max"',
        "--validate",
        "--remove-loops",
        "--via-edge-types", via,
    ]

    jobs = [
        {
            "class": "bicycle",
            "prefix": "bike",
            "fringe": 2,
            "extra": ["--max-distance", "8000", "--seed", "42"],
        },
        {
            "class": "bus",
            "prefix": "bus",
            "fringe": 5,
            "extra": [
                "--min-distance", "600",
                "--min-distance.fringe", "10",
                "--seed", "43",
            ],
        },
        {
            "class": "motorcycle",
            "prefix": "motorcycle",
            "fringe": 5,
            "extra": ["--max-distance", "1200", "--seed", "44"],
        },
        {
            "class": "passenger",
            "prefix": "veh",
            "fringe": 5,
            "extra": [
                "--min-distance", "300",
                "--min-distance.fringe", "10",
                "--allow-fringe.min-length", "1000",
                "--lanes",
                "--seed", "45",
            ],
        },
        {
            "class": "truck",
            "prefix": "truck",
            "fringe": 5,
            "extra": [
                "--min-distance", "600",
                "--min-distance.fringe", "10",
                "--seed", "47",
            ],
        },
    ]

    for job in jobs:
        density = BASE_DENSITY[job["class"]] * scale
        trips = scenario / f"osm.{job['class']}.trips.xml"
        routes = scenario / f"osm.{job['class']}.rou.xml"
        cmd = [
            py, tools,
            "-n", str(net),
            "--fringe-factor", str(job["fringe"]),
            "--insertion-density", str(density),
            "-o", str(trips),
            "-r", str(routes),
            "-b", "0", "-e", "3600",
            "--vehicle-class", job["class"],
            "--vclass", job["class"],
            "--prefix", job["prefix"],
            *common_vehicle,
            *job["extra"],
        ]
        run(cmd, cwd=scenario)

    set_motorcycle_vtype(scenario / "osm.motorcycle.trips.xml")
    print(f"Trips generated for mode={mode} in {scenario}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SUMO trips by density mode.")
    parser.add_argument(
        "-mode",
        "--mode",
        dest="mode",
        required=True,
        choices=tuple(DENSITY_SCALE),
        help="Traffic density mode",
    )
    args = parser.parse_args()
    try:
        build(args.mode)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

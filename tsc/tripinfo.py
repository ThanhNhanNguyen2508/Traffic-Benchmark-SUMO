"""SUMO tripinfo post-processing (stdlib only for parsing).

Semantics verified on SUMO 1.28.0 with ``--tripinfo-output F --tripinfo-output.write-unfinished true
--tripinfo-output.write-undeparted true`` and ``--end T``:
  * arrived     : ``arrival >= 0``; ``duration`` / ``timeLoss`` / ``waitingTime`` are full-trip values.
  * unfinished  : ``depart >= 0`` and ``arrival == -1.00`` (still driving at the end; ``duration``, ``timeLoss``,
                  ``waitingTime`` are the PARTIAL values up to the end; some also carry ``vaporized="end"``).
  * undeparted  : ``depart == "-1"`` (``departLane=""``): the vehicle's departure time has passed but SUMO could
                  not insert it yet; ``departDelay`` is the wait so far, ``timeLoss``/``duration`` are 0.
                  Vehicles whose scheduled departure lies after the end are NOT written.
``timeLoss`` = time lost vs driving at the allowed speed; ``departDelay`` = insertion wait after the scheduled
departure; ``delay_with_depart`` = timeLoss + departDelay.
"""
import functools
import math
import subprocess
import xml.etree.ElementTree as ET

NAN = float("nan")


@functools.lru_cache(maxsize=1)
def _undeparted_supported():
    try:
        import sumolib
        out = subprocess.run([sumolib.checkBinary("sumo"), "--help"], capture_output=True, text=True,
                             timeout=60).stdout
        return "--tripinfo-output.write-undeparted" in out
    except Exception:  # noqa: BLE001
        return False


def tripinfo_sumo_args(path, undeparted=None):
    """SUMO CLI args writing tripinfo to ``path`` incl. unfinished (and undeparted if supported / True)."""
    args = ["--tripinfo-output", str(path), "--tripinfo-output.write-unfinished", "true"]
    if _undeparted_supported() if undeparted is None else undeparted:
        args += ["--tripinfo-output.write-undeparted", "true"]
    return args


def _mean(v):
    return sum(v) / len(v) if v else NAN


def parse_tripinfo(path):
    """Flat metrics dict from a tripinfo XML (see module docstring for the arrived/unfinished/undeparted split).

    Keys: n_trips_arrived, n_trips_unfinished, n_trips_undeparted; arrived-only means delay, delay_with_depart,
    trip_duration, waiting_time, waiting_count; all-vehicle means delay_all, delay_with_depart_all,
    waiting_time_all (arrived + unfinished + undeparted, partial values for the latter two). NaN if empty.
    """
    arr, unf, und = [], [], []
    try:
        for _, el in ET.iterparse(str(path), events=("end",)):
            if el.tag != "tripinfo":
                continue
            a = el.attrib
            rec = (float(a.get("timeLoss", 0)), float(a.get("departDelay", 0)), float(a.get("duration", 0)),
                   float(a.get("waitingTime", 0)), float(a.get("waitingCount", 0)))
            if float(a.get("depart", "-1")) < 0:
                und.append(rec)
            elif float(a.get("arrival", "-1")) < 0:
                unf.append(rec)
            else:
                arr.append(rec)
            el.clear()
    except (ET.ParseError, FileNotFoundError):   # empty / unreadable file => whatever parsed so far (maybe none)
        pass
    allv = arr + unf + und
    col = lambda rows, i: [r[i] for r in rows]
    wd = lambda rows: [r[0] + r[1] for r in rows]
    return {
        "n_trips_arrived": len(arr), "n_trips_unfinished": len(unf), "n_trips_undeparted": len(und),
        "delay": _mean(col(arr, 0)), "delay_with_depart": _mean(wd(arr)),
        "trip_duration": _mean(col(arr, 2)), "waiting_time": _mean(col(arr, 3)),
        "waiting_count": _mean(col(arr, 4)),
        "delay_all": _mean(col(allv, 0)), "delay_with_depart_all": _mean(wd(allv)),
        "waiting_time_all": _mean(col(allv, 3)),
    }

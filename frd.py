"""Read nodal displacement and stress from CalculiX .frd result files.

Pure Python so both the main process and the FreeCAD worker can import it.
CalculiX works in mm, N and MPa; callers convert to SI.
"""

from __future__ import annotations

import math
import re

_NUMBER = re.compile(r"[-+]?\d+\.\d+E[-+]\d+")


def von_mises(sxx, syy, szz, sxy, syz, szx):
    return math.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    )


def read_frd_nodal(frd_file):
    """Return {"displacement": {node: |u| mm}, "stress": {node: von Mises MPa}}."""
    displacement: dict[int, float] = {}
    stress: dict[int, float] = {}
    section = None
    try:
        with open(frd_file) as stream:
            for line in stream:
                if line.startswith(" -4  DISP"):
                    section = "displacement"
                    continue
                if line.startswith(" -4  STRESS"):
                    section = "stress"
                    continue
                if line.startswith(" -4"):
                    section = None
                    continue
                if line.startswith(" -3"):
                    section = None
                    continue
                if section is None or not line.startswith(" -1"):
                    continue
                node = int(line[3:13])
                values = [float(value) for value in _NUMBER.findall(line[13:])]
                if section == "displacement" and len(values) >= 3:
                    displacement[node] = math.sqrt(sum(v * v for v in values[:3]))
                elif section == "stress" and len(values) >= 6:
                    stress[node] = von_mises(*values[:6])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not parse CalculiX FRD results: {exc}") from exc
    if not displacement or not stress:
        raise RuntimeError("CalculiX FRD file contains no displacement or stress results.")
    return {"displacement": displacement, "stress": stress}


def extract_frd_results(frd_file, excluded_stress_nodes=()):
    """Peak displacement magnitude (mm) and von Mises stress (MPa).

    Nodes in ``excluded_stress_nodes`` are ignored for the stress peak only,
    which is how support singularities are kept out of the stress constraint.
    """
    nodal = read_frd_nodal(frd_file)
    excluded = set(excluded_stress_nodes)
    counted = {node: value for node, value in nodal["stress"].items() if node not in excluded}
    if not counted:
        raise RuntimeError("Every stress node was excluded; reduce stress_exclusion_radius.")
    peak_node = max(counted, key=counted.get)
    return {
        "stress": counted[peak_node],
        "stress_node": peak_node,
        "displacement": max(nodal["displacement"].values()),
    }

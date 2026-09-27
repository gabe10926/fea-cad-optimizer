"""Command-line parametric CAD/FEA optimization without opening any GUI.

    python headless.py examples/ibeam.py --profile examples/ibeam_profile.json --workers 4
    python headless.py examples/ibeam.py --list-faces
    python headless.py part.FCStd --bound Width:10:30 --step Width:0.2
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

from fem_profile import AnalysisProfile
from pipeline import OptimizerSettings, design_space, open_model, run_optimization


def _named_values(values: Sequence[str], count: int, option: str) -> dict[str, list[float]]:
    parsed: dict[str, list[float]] = {}
    for value in values:
        fields = value.split(":")
        if len(fields) != count:
            raise ValueError(f"{option} expects NAME:" + ":".join(["VALUE"] * (count - 1)))
        name = fields[0]
        if not name or name in parsed:
            raise ValueError(f"{option} contains an empty or duplicate parameter name: {name!r}")
        parsed[name] = [float(field) for field in fields[1:]]
        if any(not math.isfinite(item) for item in parsed[name]):
            raise ValueError(f"{option} values must be finite for {name}.")
    return parsed


def print_faces(metadata) -> None:
    print(f"{'Face':<8} {'center (mm)':<32} {'area (mm2)':>12}  on bounding-box plane")
    for face in metadata.faces:
        center = ", ".join(f"{value:.2f}" for value in face["center"])
        print(f"{face['face']:<8} {center:<32} {face['area']:>12.2f}  {' '.join(face['on'])}")
    if metadata.selections:
        print("\nProfile selections resolve to:")
        for name, faces in metadata.selections.items():
            print(f"  {name}: {', '.join(faces)}")


def optimize_model(args: argparse.Namespace) -> dict:
    profile = AnalysisProfile.load(args.profile) if args.profile else AnalysisProfile()
    profile_errors = profile.validate()
    if profile_errors:
        raise ValueError("; ".join(profile_errors))

    session = open_model(
        args.model, profile, jobs_dir=args.jobs_dir, workers=args.workers,
        freecad_python=args.freecad_python, ccx=args.ccx,
    )
    try:
        metadata = session.metadata
        if args.list_faces:
            print_faces(metadata)
            return {"feasible": True}
        errors = metadata.validate() + metadata.warnings
        if errors:
            raise ValueError("; ".join(errors))
        initial, bounds, steps = design_space(
            metadata,
            {name: tuple(values) for name, values in _named_values(args.bound, 3, "--bound").items()},
            {name: values[0] for name, values in _named_values(args.step, 2, "--step").items()},
        )
        print(f"design variables: {[parameter.alias for parameter in metadata.parameters]}", flush=True)
        result = run_optimization(
            session, profile, initial, steps, bounds,
            OptimizerSettings(args.learning_rate, args.tolerance, args.max_iterations, args.method),
            export_dir=args.export_dir,
            callback=lambda iteration, params, value, norm, info: print(
                f"iteration={iteration} objective={value:.8g} solves={info['solves']} "
                f"gradient_norm={norm:.6g} parameters={params.round(4).tolist()}",
                flush=True,
            ),
        )
    finally:
        session.close()
    printable = {key: value for key, value in result.items() if key != "history"}
    print(json.dumps(printable, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a parametric part (build123d .py or FreeCAD .FCStd) with Gmsh and CalculiX."
    )
    parser.add_argument("model", help="build123d part script (.py) or parametric FreeCAD document (.FCStd)")
    parser.add_argument("--profile", help="FEM analysis profile JSON (defaults to built-in aluminum profile)")
    parser.add_argument("--list-faces", action="store_true",
                        help="Print the model's faces and how the profile's selectors resolve, then exit")
    parser.add_argument(
        "--bound", action="append", default=[], metavar="NAME:MIN:MAX",
        help="Override bounds for a design variable; may be repeated",
    )
    parser.add_argument(
        "--step", action="append", default=[], metavar="NAME:VALUE",
        help="Override finite-difference step for a design variable; may be repeated",
    )
    parser.add_argument("--method", choices=["slsqp", "gradient"], default="slsqp",
                        help="slsqp: constrained SQP (default); gradient: penalty + gradient descent")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="gradient method only")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1, help="Designs solved in parallel")
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--export-dir", default="exports")
    parser.add_argument("--freecad-python", help="Path to FreeCADCmd/freecadcmd (FreeCAD models only)")
    parser.add_argument("--ccx", help="Path/name of CalculiX executable (defaults to CCX_BINARY or ccx)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = optimize_model(args)
    except Exception as exc:
        parser.exit(2, f"headless optimization failed: {exc}\n")
    return 0 if result["feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

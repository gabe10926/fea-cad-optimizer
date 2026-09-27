"""
CAD Worker Process - Runs inside FreeCAD's Python 3.11 environment.
Executed as a subprocess by CADController.

Communication:
  Input:  params.json   {"Height": value, ...}
  Output: results.json  {"stress": value, "mass": value, "error": null}
"""

import sys
import os
import json
import shutil
import subprocess
import traceback
import argparse
import tempfile
import math
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_selectors  # noqa: E402
from frd import extract_frd_results  # noqa: E402

# Extend path so FreeCAD libs can be found from the AppImage layout.
_BASE = os.path.expanduser("~/.local/bin/squashfs-root")
sys.path.append(f"{_BASE}/usr/lib")
sys.path.append(f"{_BASE}/usr/Mod")

def load_freecad():
    import FreeCAD as App
    from femtools.ccxtools import CcxTools
    return App, CcxTools


def terminal_solids(doc):
    """Objects with positive-volume shapes that no other shaped object builds on."""
    shaped = [
        obj for obj in doc.Objects
        if hasattr(obj, "Shape")
        and not getattr(obj.Shape, "isNull", lambda: False)()
        and float(getattr(obj.Shape, "Volume", 0.0)) > 0
    ]
    return [
        obj for obj in shaped
        if not any(dependent in shaped for dependent in getattr(obj, "InList", []))
    ]


def extract_mass_from_model(doc, density=2700.0):
    """Return physical mass in kg from terminal solid geometry and kg/m^3 density."""
    density = float(density)
    terminal = terminal_solids(doc)
    if not terminal:
        raise ValueError("Model contains no terminal solid geometry with positive volume.")
    volume_mm3 = sum(float(obj.Shape.Volume) for obj in terminal)
    mass_kg = volume_mm3 * 1e-9 * density
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Could not calculate a positive physical mass from model volume and density.")
    return mass_kg


def face_infos(shape_object):
    faces = []
    for index, face in enumerate(shape_object.Shape.Faces, 1):
        box = face.BoundBox
        center = face.CenterOfMass
        faces.append(face_selectors.FaceInfo(
            index=index,
            center=(center.x, center.y, center.z),
            bbox=(box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax),
            area=face.Area,
        ))
    return faces


def set_parameters(doc, params):
    """Write each alias to the spreadsheet that defines it."""
    sheets = [obj for obj in doc.Objects if obj.TypeId == "Spreadsheet::Sheet"]
    if not sheets:
        raise ValueError("No spreadsheet object found in model")
    for key, value in params.items():
        owner = next((sheet for sheet in sheets if sheet.getCellFromAlias(key)), None)
        if owner is None:
            raise ValueError(f"No spreadsheet defines the alias {key!r}")
        owner.set(owner.getCellFromAlias(key), str(value))
    doc.recompute()


def find_fem_analysis(doc):
    """Return the first Fem::FemAnalysis object in the document."""
    for obj in doc.Objects:
        if obj.TypeId == "Fem::FemAnalysis":
            return obj
    raise ValueError("No FEM Analysis found in model")


def apply_analysis_profile(doc, profile_file):
    """Store the GUI-defined FEM setup in the document for deterministic export.

    Geometry selections are represented by semantic names in the profile. A
    later adapter can resolve those names to faces without changing the GUI or
    optimization protocol.
    """
    if not profile_file:
        return None
    from fem_profile import AnalysisProfile
    with open(profile_file) as stream:
        raw_profile = json.load(stream)
    profile_object = AnalysisProfile.from_dict(raw_profile)
    errors = profile_object.validate()
    if errors:
        raise ValueError("; ".join(errors))
    profile = profile_object.to_dict()
    material = profile.get("material", {})
    mesh = profile.get("mesh", {})

    import FreeCAD as App
    import ObjectsFem

    shape_objects = terminal_solids(doc)
    if not shape_objects:
        raise ValueError("FEM profile requires at least one solid geometry object")
    shape_object = max(shape_objects, key=lambda obj: obj.Shape.Volume)
    analysis = find_fem_analysis(doc)
    solver = doc.getObject("ProfileSolver")
    if solver is None:
        solver = ObjectsFem.makeSolverCalculix(doc, "ProfileSolver")
        analysis.addObject(solver)
    solver.AnalysisType = "static"

    material_object = doc.getObject("ProfileMaterial")
    if material_object is None:
        material_object = ObjectsFem.makeMaterialSolid(doc, "ProfileMaterial")
        analysis.addObject(material_object)
    material_object.Label = str(material.get("name", "Profile material"))
    material_object.Material = {
        "Name": material_object.Label,
        "YoungsModulus": f"{material['youngs_modulus']} Pa",
        "PoissonRatio": str(material["poisson_ratio"]),
        "Density": f"{material['density']} kg/m^3",
        "YieldStrength": f"{material.get('yield_strength', 0)} Pa",
    }

    mesh_object = doc.getObject("ProfileMesh")
    if mesh_object is None:
        mesh_object = ObjectsFem.makeMeshGmsh(doc, "ProfileMesh")
        analysis.addObject(mesh_object)
    mesh_object.Shape = shape_object
    mesh_object.CharacteristicLengthMin = float(mesh["minimum_size"])
    mesh_object.CharacteristicLengthMax = min(
        float(mesh["size"]), float(mesh["maximum_size"])
    )
    mesh_object.ElementOrder = "2nd" if int(mesh.get("element_order", 1)) == 2 else "1st"
    mesh_object.ElementDimension = "3D"
    doc.recompute()
    from femmesh.gmshtools import GmshTools
    tools = GmshTools(mesh_object)
    tools.update_mesh_data()
    tools.get_tmp_file_paths(os.path.join(tempfile.gettempdir(), "fea_cad_mesh"), True)
    tools.get_gmsh_command()
    tools.write_gmsh_input_files()
    error = tools.run_gmsh_with_geo()
    if error:
        raise RuntimeError(f"Gmsh failed: {error}")
    tools.read_and_set_new_mesh()
    if mesh_object.FemMesh is None or mesh_object.FemMesh.VolumeCount == 0:
        raise RuntimeError("Gmsh produced no volume elements")

    faces = face_infos(shape_object)

    def references(selection):
        names = [f"Face{index}" for index in face_selectors.resolve(selection.get("value", ""), faces)]
        return [(shape_object, names)]

    for index, condition in enumerate(profile.get("boundary_conditions", []), 1):
        object_name = f"ProfileSupport{index}"
        constraint = doc.getObject(object_name)
        if constraint is None:
            constraint = ObjectsFem.makeConstraintFixed(doc, object_name)
            analysis.addObject(constraint)
        constraint.Label = str(condition.get("target", {}).get("name", "Fixed support"))
        constraint.References = references(condition.get("target", {}))

    for index, load in enumerate(profile.get("loads", []), 1):
        object_name = f"ProfileLoad{index}"
        constraint = doc.getObject(object_name)
        load_type = str(load.get("type", "force")).lower()
        if constraint is None:
            factory = ObjectsFem.makeConstraintPressure if load_type == "pressure" else ObjectsFem.makeConstraintForce
            constraint = factory(doc, object_name)
            analysis.addObject(constraint)
        constraint.Label = str(load.get("target", {}).get("name", "Load"))
        constraint.References = references(load.get("target", {}))
        magnitude = float(load.get("magnitude", 1000.0))
        if load_type == "pressure":
            if not hasattr(constraint, "Pressure"):
                raise RuntimeError("FreeCAD pressure constraint does not expose a Pressure property.")
            constraint.Pressure = magnitude / 1e3
        else:
            if not hasattr(constraint, "Force") or not hasattr(constraint, "DirectionVector"):
                raise RuntimeError("FreeCAD force constraint is missing Force or DirectionVector.")
            vector = [float(value) for value in load.get("vector", [0.0, -1.0, 0.0])]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 0:
                raise ValueError("Force direction vector must be non-zero.")
            # FreeCAD's CalculiX writer scales the Force property by 1e-3.
            constraint.Force = magnitude * 1000.0
            constraint.DirectionVector = App.Vector(*(value / norm for value in vector))

    feature = doc.getObject("AnalysisProfile")
    if feature is None:
        feature = doc.addObject("App::FeaturePython", "AnalysisProfile")
        feature.Label = "FEM Setup Profile"
    if not hasattr(feature, "ProfileJSON"):
        feature.addProperty("App::PropertyString", "ProfileJSON", "FEM")
    if not hasattr(feature, "MaterialName"):
        feature.addProperty("App::PropertyString", "MaterialName", "FEM")
    feature.ProfileJSON = json.dumps(profile, sort_keys=True)
    feature.MaterialName = str(material.get("name", "Custom"))
    if hasattr(analysis, "addObject") and feature not in analysis.Group:
        analysis.addObject(feature)
    return profile


def inspect_model(model_file):
    App, _ = load_freecad()
    doc = App.open(model_file)
    profile = apply_analysis_profile(doc, os.environ.get("FEA_CAD_PROFILE"))
    parameters = []
    for obj in doc.Objects:
        if obj.TypeId != "Spreadsheet::Sheet":
            continue
        aliases = obj.getUsedCells() if hasattr(obj, "getUsedCells") else []
        for cell in aliases:
            alias = obj.getAlias(cell) if hasattr(obj, "getAlias") else ""
            if not alias:
                continue
            try:
                value = float(obj.get(cell))
            except (TypeError, ValueError):
                continue
            parameters.append({
                "name": alias,
                "alias": alias,
                "value": value,
                "minimum": value * 0.5 if value > 0 else value - 1.0,
                "maximum": value * 1.5 if value > 0 else value + 1.0,
                "unit": "",
            })
    geometry_count = sum(
        1 for obj in doc.Objects
        if hasattr(obj, "Shape") and not getattr(obj.Shape, "isNull", lambda: False)()
    )
    mass = extract_mass_from_model(
        doc, density=float(profile["material"]["density"]) if profile else 2700.0
    )
    has_analysis = any(obj.TypeId == "Fem::FemAnalysis" for obj in doc.Objects)
    solids = terminal_solids(doc)
    faces = face_infos(max(solids, key=lambda obj: obj.Shape.Volume)) if solids else []
    return {
        "model_path": str(model_file),
        "backend": "freecad",
        "faces": face_selectors.describe(faces) if faces else [],
        "parameters": parameters,
        "has_fem_analysis": has_analysis,
        "geometry_count": geometry_count,
        "mass": mass,
        "mass_kg": mass,
        "warnings": [],
        "profile_applied": bool(os.environ.get("FEA_CAD_PROFILE")),
    }


def run_worker(params_file, results_file, model_file):
    """Main worker: read params -> update model -> solve -> write results."""
    results = {"stress": None, "mass": None, "error": None}

    try:
        with open(params_file) as f:
            params = json.load(f)

        App, CcxTools = load_freecad()
        doc = App.open(model_file)
        set_parameters(doc, params)
        profile = apply_analysis_profile(doc, os.environ.get("FEA_CAD_PROFILE"))
        if profile is None:
            raise ValueError("An FEM analysis profile is required for a simulation.")

        # Export INP via FreeCAD FEM tools
        analysis = find_fem_analysis(doc)
        solver = doc.getObject("ProfileSolver")
        if solver is None:
            raise RuntimeError("FEM profile did not create a CalculiX solver")
        fea = CcxTools(solver=solver)
        fea.setup_working_dir()
        fea.update_objects()
        fea.write_inp_file()

        generated_inp = fea.inp_file_name
        if not os.path.exists(generated_inp):
            raise RuntimeError("INP file was not generated")
        work_dir = fea.working_dir
        solver_stem = os.path.splitext(os.path.basename(generated_inp))[0]

        # Solve with CalculiX
        result = subprocess.run(
            [os.environ.get("FEA_CAD_CCX", "ccx"), solver_stem],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).decode(errors="replace")
            raise RuntimeError(f"CalculiX failed (exit {result.returncode}): {output[-4000:]}")

        frd_results = extract_frd_results(os.path.join(work_dir, f"{solver_stem}.frd"))
        results["stress"] = frd_results["stress"] * 1e6
        results["stress_pa"] = results["stress"]
        results["displacement"] = frd_results["displacement"] * 1e-3
        results["displacement_m"] = results["displacement"]
        results["mass"] = float(extract_mass_from_model(
            doc, density=float(profile["material"]["density"])
        ))
        results["mass_kg"] = results["mass"]
        results["volume_mm3"] = results["mass"] / float(profile["material"]["density"]) * 1e9
        if results["stress"] <= 0 or results["displacement"] <= 0:
            raise RuntimeError("Structural solve produced no stress or displacement result")

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


def save_design(params_file, results_file, model_file, output_file):
    """Regenerate and save the selected parameterized FEM model headlessly."""
    results = {"error": None}
    try:
        with open(params_file) as stream:
            params = json.load(stream)
        App, _ = load_freecad()
        doc = App.open(model_file)
        set_parameters(doc, params)
        profile = apply_analysis_profile(doc, os.environ.get("FEA_CAD_PROFILE"))
        if profile is None:
            raise ValueError("An FEM analysis profile is required to save the optimized model.")
        doc.recompute()
        output = Path(output_file).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        doc.saveAs(str(output))
    except Exception as exc:
        results["error"] = str(exc)
        results["traceback"] = traceback.format_exc()
    with open(results_file, "w") as stream:
        json.dump(results, stream, indent=2)


def probe_runtime():
    """Fail early unless this executable is the FreeCAD command runtime."""
    load_freecad()
    return {"freecad": True, "version": str(__import__("FreeCAD").Version())}


if __name__ in {"__main__", "cad_worker"}:
    parser = argparse.ArgumentParser(description="FreeCAD FEA worker process")
    parser.add_argument("--params", default=os.environ.get("FEA_CAD_PARAMS"), help="Input parameters JSON")
    parser.add_argument("--results", default=os.environ.get("FEA_CAD_RESULTS"), help="Output results JSON")
    parser.add_argument("--model", default=os.environ.get("FEA_CAD_MODEL"), help="FreeCAD model file (.FCStd)")
    parser.add_argument("--profile", default=os.environ.get("FEA_CAD_PROFILE"), help="FEM setup profile JSON")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe", action="store_true")
    modes.add_argument("--inspect", action="store_true", help="Inspect model metadata")
    modes.add_argument("--simulate", action="store_true", help="Run a structural simulation")
    modes.add_argument("--save-design", action="store_true", help="Save a parameterized FEM model")
    args = parser.parse_args(
        [] if os.environ.get("FEA_CAD_RESULTS") or os.environ.get("FEA_CAD_PROBE") else None
    )

    if args.probe or os.environ.get("FEA_CAD_PROBE") == "1":
        probe_runtime()
    elif args.inspect or os.environ.get("FEA_CAD_INSPECT") == "1":
        if not args.results or not args.model:
            parser.error("--results and --model are required for inspection")
        try:
            output = inspect_model(args.model)
        except Exception as exc:
            output = {"error": str(exc), "traceback": traceback.format_exc()}
        with open(args.results, "w") as stream:
            json.dump(output, stream, indent=2)
    elif args.save_design or os.environ.get("FEA_CAD_EXPORT"):
        if not args.results or not args.model or not args.params or not os.environ.get("FEA_CAD_EXPORT"):
            parser.error("--params, --results, --model, and FEA_CAD_EXPORT are required")
        save_design(args.params, args.results, args.model, os.environ["FEA_CAD_EXPORT"])
    else:
        if not args.params or not args.results or not args.model:
            parser.error("--params, --results, and --model are required")
        run_worker(args.params, args.results, args.model)

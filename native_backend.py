"""Python-native CAD/FEA backend: build123d geometry, Gmsh meshing, CalculiX solve.

A model is a Python file that defines ``part(**dimensions)`` returning one
build123d solid. Keyword defaults are the starting design; optional module
constants refine the design space:

    DESIGN_VARIABLES = ["Width", "Height"]      # default: every numeric keyword
    BOUNDS = {"Width": (10, 30)}                # default: 0.5x to 1.5x the default
    STEPS = {"Width": 0.5}                      # default: 3% of the range

Every evaluation runs in a worker process from a pool, so a crash in the
geometry kernel or mesher cannot take down the optimizer or GUI, and several
designs can be solved at once.
"""

from __future__ import annotations

import importlib.util
import inspect as pyinspect
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

import ccx_deck
import face_selectors
from face_selectors import FaceInfo
from fem_profile import AnalysisProfile
from frd import extract_frd_results

PART_FUNCTION = "part"
_MODULE_CACHE: dict[tuple[str, float], object] = {}


# ---------------------------------------------------------------- part files

def load_part_module(model_path, import_paths=()):
    """Import a part file, cached by path and modification time."""
    path = Path(model_path).resolve()
    key = (str(path), path.stat().st_mtime)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    for directory in (path.parent, *map(Path, import_paths)):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(f"fea_part_{abs(hash(key))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, PART_FUNCTION, None)):
        raise ValueError(f"{path.name} must define a function `def {PART_FUNCTION}(...)` returning a solid.")
    _MODULE_CACHE[key] = module
    return module


def design_parameters(module) -> list[dict]:
    """Design variables from the keyword defaults of ``part()``."""
    signature = pyinspect.signature(getattr(module, PART_FUNCTION))
    allowed = getattr(module, "DESIGN_VARIABLES", None)
    bounds = dict(getattr(module, "BOUNDS", {}))
    steps = dict(getattr(module, "STEPS", {}))
    names = [
        name for name, item in signature.parameters.items()
        if item.kind not in (item.VAR_POSITIONAL, item.VAR_KEYWORD)
    ]
    for label, mapping in (("DESIGN_VARIABLES", allowed or []), ("BOUNDS", bounds), ("STEPS", steps)):
        unknown = set(mapping) - set(names)
        if unknown:
            raise ValueError(f"{label} names unknown part() arguments: {', '.join(sorted(unknown))}")
    parameters = []
    for name in names:
        default = signature.parameters[name].default
        if default is pyinspect.Parameter.empty:
            raise ValueError(f"part() argument {name!r} needs a default value.")
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        if allowed is not None and name not in allowed:
            continue
        value = float(default)
        minimum, maximum = bounds.get(name, (value * 0.5, value * 1.5) if value > 0 else (value - 1.0, value + 1.0))
        parameters.append({
            "name": name,
            "alias": name,
            "value": value,
            "minimum": float(minimum),
            "maximum": float(maximum),
            "step": float(steps[name]) if name in steps else None,
            "unit": "mm",
        })
    return parameters


def build_shape(module, params: dict):
    """Call ``part(**params)`` and return a single build123d solid."""
    from build123d import Shape

    result = getattr(module, PART_FUNCTION)(**{name: float(value) for name, value in params.items()})
    if hasattr(result, "part") and not isinstance(result, Shape):  # BuildPart context
        result = result.part
    if not isinstance(result, Shape):
        value = getattr(result, "val", None)  # CadQuery Workplane
        if callable(value):
            result = value()
        wrapped = getattr(result, "wrapped", None)
        if wrapped is None:
            raise TypeError(f"part() returned {type(result).__name__}, not a build123d or CadQuery shape.")
        result = Shape.cast(wrapped)
    solids = result.solids()
    if len(solids) != 1:
        raise ValueError(f"part() must return exactly one solid, got {len(solids)}. Fuse the bodies first.")
    solid = solids[0]
    if not solid.is_valid or solid.volume <= 0:
        raise ValueError("part() returned an invalid solid or one with no volume.")
    return solid


# ---------------------------------------------------------------- meshing

def _gmsh_session(brep_path):
    import gmsh

    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    gmsh.model.add("part")
    gmsh.model.occ.importShapes(str(brep_path))
    gmsh.model.occ.synchronize()
    return gmsh


def _faces(gmsh) -> list[FaceInfo]:
    faces = []
    for _, tag in gmsh.model.getEntities(2):
        faces.append(FaceInfo(
            index=tag,
            center=tuple(gmsh.model.occ.getCenterOfMass(2, tag)),
            bbox=tuple(gmsh.model.getBoundingBox(2, tag)),
            area=gmsh.model.occ.getMass(2, tag),
        ))
    return faces


def _surface_triangles(gmsh, face_tags) -> np.ndarray:
    blocks = []
    for tag in face_tags:
        types, _, connectivity = gmsh.model.mesh.getElements(2, tag)
        for element_type, nodes in zip(types, connectivity):
            count = gmsh.model.mesh.getElementProperties(element_type)[3]
            blocks.append(np.asarray(nodes, dtype=np.int64).reshape(-1, count))
    if not blocks:
        raise RuntimeError(f"Faces {face_tags} have no surface mesh.")
    return np.vstack(blocks)


def _surface_nodes(gmsh, face_tags) -> set[int]:
    nodes: set[int] = set()
    for tag in face_tags:
        tags, _, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        nodes.update(int(n) for n in tags)
    return nodes


def mesh_and_solve(solid, profile: AnalysisProfile, workdir: Path, ccx: str, threads: int = 1,
                   timeout: float = 600) -> dict:
    """Mesh one solid, apply the profile, run CalculiX, and reduce the results."""
    from build123d import export_brep
    from scipy.spatial import cKDTree

    started = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(stage):
        timings[stage] = round(time.perf_counter() - started - sum(timings.values()), 3)

    brep = workdir / "part.brep"
    export_brep(solid, str(brep))
    gmsh = _gmsh_session(brep)
    try:
        faces = _faces(gmsh)
        mesh = profile.mesh
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh.minimum_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", min(mesh.size, mesh.maximum_size))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", mesh.curvature_elements)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh.element_order)
        # Straight mid-edge nodes: curved 10-node tets on small fillets can invert
        # (CalculiX "nonpositive jacobian"); mass still comes from the exact CAD volume.
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        lap("geometry")
        gmsh.model.mesh.generate(3)
        lap("mesh")

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = node_tags.astype(np.int64)
        node_coords = node_coords.reshape(-1, 3)
        coords = np.zeros((int(node_tags.max()) + 1, 3))
        coords[node_tags] = node_coords

        types, element_tags, connectivity = gmsh.model.mesh.getElements(3)
        if len(types) != 1:
            raise RuntimeError("Expected a single tetrahedral element type from Gmsh.")
        per_element = gmsh.model.mesh.getElementProperties(types[0])[3]
        elements = np.asarray(connectivity[0], dtype=np.int64).reshape(-1, per_element)
        if len(elements) == 0:
            raise RuntimeError("Gmsh produced no volume elements.")

        fixed: set[int] = set()
        selections: dict[str, list[int]] = {}
        for condition in profile.boundary_conditions:
            tags = face_selectors.resolve(condition.target.value, faces)
            selections[condition.target.name] = tags
            fixed |= _surface_nodes(gmsh, tags)

        forces: dict[int, np.ndarray] = {}
        for load in profile.loads:
            tags = face_selectors.resolve(load.target.value, faces)
            selections[load.target.name] = tags
            triangles = _surface_triangles(gmsh, tags)
            areas, _ = ccx_deck.triangle_geometry(coords[triangles[:, :3]])
            if load.type.lower() == "pressure":
                normals = ccx_deck.outward_normals(triangles, elements[:, :4], coords)
                tractions = -(load.magnitude / 1e6) * normals
            else:
                direction = np.asarray(load.vector, dtype=float)
                direction /= np.linalg.norm(direction)
                tractions = np.tile(load.magnitude * direction / areas.sum(), (len(triangles), 1))
            for node, force in ccx_deck.consistent_loads(triangles, tractions, coords).items():
                forces[node] = forces.get(node, 0.0) + force
    finally:
        gmsh.finalize()
    lap("loads")

    excluded: list[int] = []
    radius = profile.analysis.stress_exclusion_radius
    if radius > 0:
        distance, _ = cKDTree(coords[sorted(fixed)]).query(node_coords, distance_upper_bound=radius)
        excluded = node_tags[distance < radius].tolist()

    material = profile.material
    ccx_deck.write_inp(
        workdir / "job.inp", node_tags, node_coords, elements, element_tags[0], fixed, forces,
        material.youngs_modulus / 1e6, material.poisson_ratio,
    )
    environment = {**os.environ, "OMP_NUM_THREADS": str(max(1, threads))}
    completed = subprocess.run(
        [ccx, "-i", "job"], cwd=workdir, capture_output=True, text=True, timeout=timeout, env=environment,
    )
    lap("solve")
    frd_path = workdir / "job.frd"
    if completed.returncode != 0 or "*ERROR" in completed.stdout or not frd_path.is_file():
        output = (completed.stdout + completed.stderr)[-4000:]
        raise RuntimeError(f"CalculiX failed (exit {completed.returncode}): {output}")
    peaks = extract_frd_results(frd_path, excluded)
    lap("results")

    volume = float(solid.volume)
    mass = volume * 1e-9 * material.density
    stress = peaks["stress"] * 1e6
    displacement = peaks["displacement"] * 1e-3
    if stress <= 0 or displacement <= 0:
        raise RuntimeError("Structural solve produced no stress or displacement result.")
    return {
        "stress": stress,
        "stress_pa": stress,
        "displacement": displacement,
        "displacement_m": displacement,
        "mass": mass,
        "mass_kg": mass,
        "volume_mm3": volume,
        "peak_stress_location_mm": coords[peaks["stress_node"]].round(4).tolist(),
        "nodes": int(len(node_tags)),
        "elements": int(len(elements)),
        "stress_nodes_excluded": len(excluded),
        "selections": {name: [f"Face{tag}" for tag in tags] for name, tags in selections.items()},
        "solve_seconds": round(time.perf_counter() - started, 3),
        "timings": timings,
        "error": None,
    }


# ------------------------------------------------ pool entry points (top level so they pickle)

def _worker_init(parent_pid):
    """Exit when the parent dies, so a killed run never leaves orphaned workers."""
    import threading

    def watch():
        while True:
            time.sleep(2)
            if os.getppid() != parent_pid:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _inspect_task(model_path, profile_dict, import_paths):
    module = load_part_module(model_path, import_paths)
    parameters = design_parameters(module)
    solid = build_shape(module, {item["name"]: item["value"] for item in parameters})
    profile = AnalysisProfile.from_dict(profile_dict) if profile_dict else AnalysisProfile()
    from build123d import export_brep

    with tempfile.TemporaryDirectory() as tmp:
        brep = Path(tmp) / "part.brep"
        export_brep(solid, str(brep))
        gmsh = _gmsh_session(brep)
        try:
            faces = _faces(gmsh)
        finally:
            gmsh.finalize()
    selections, warnings = {}, []
    for target in [item.target for item in profile.boundary_conditions] + [item.target for item in profile.loads]:
        try:
            selections[target.name] = [f"Face{tag}" for tag in face_selectors.resolve(target.value, faces)]
        except ValueError as exc:
            warnings.append(str(exc))
    mass = float(solid.volume) * 1e-9 * profile.material.density
    return {
        "model_path": str(model_path),
        "backend": "build123d",
        "parameters": parameters,
        "has_fem_analysis": True,
        "geometry_count": 1,
        "mass": mass,
        "mass_kg": mass,
        "faces": face_selectors.describe(faces),
        "selections": selections,
        "warnings": warnings,
        "profile_applied": profile_dict is not None,
    }


def _simulate_task(model_path, params, profile_dict, ccx, import_paths, threads, timeout, keep_dir=None):
    module = load_part_module(model_path, import_paths)
    solid = build_shape(module, params)
    profile = AnalysisProfile.from_dict(profile_dict)
    if keep_dir is not None:
        workdir = Path(keep_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        return mesh_and_solve(solid, profile, workdir, ccx, threads, timeout)
    with tempfile.TemporaryDirectory(prefix="fea_eval_") as tmp:
        return mesh_and_solve(solid, profile, Path(tmp), ccx, threads, timeout)


def _export_task(model_path, params, output_stem, import_paths):
    from build123d import export_step, export_stl

    solid = build_shape(load_part_module(model_path, import_paths), params)
    step, stl = Path(f"{output_stem}.step"), Path(f"{output_stem}.stl")
    export_step(solid, str(step))
    export_stl(solid, str(stl))
    return [str(step), str(stl)]


# ---------------------------------------------------------------- controller

class NativeController:
    """Same interface as the FreeCAD CADController, backed by a process pool."""

    def __init__(self, model_path, ccx_binary=None, workers=1, timeout=600, import_paths=()):
        self.model_path = str(Path(model_path).resolve())
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.ccx_binary = shutil.which(ccx_binary or os.environ.get("CCX_BINARY", "ccx"))
        if self.ccx_binary is None:
            raise RuntimeError("CalculiX executable 'ccx' was not found; install CalculiX or set CCX_BINARY.")
        self.workers = max(1, int(workers))
        self.timeout = timeout
        self.import_paths = tuple(str(path) for path in import_paths)
        # CalculiX's default SPOOLES solver scales poorly past a few threads; parallel
        # designs are the better use of cores, so each worker gets one thread.
        self.threads = 1 if self.workers > 1 else min(4, os.cpu_count() or 1)
        self._pool: ProcessPoolExecutor | None = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init, initargs=(os.getpid(),),
            )
        return self._pool

    def _reset_pool(self):
        if self._pool is not None:
            for process in list(getattr(self._pool, "_processes", {}).values()):
                process.terminate()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _call(self, function, *args):
        future = self._executor().submit(function, *args)
        try:
            return future.result(timeout=self.timeout)
        except (FutureTimeout, BrokenProcessPool) as exc:
            self._reset_pool()
            raise RuntimeError(f"Evaluation worker failed or exceeded {self.timeout} s: {exc!r}") from exc

    @staticmethod
    def _checked(profile: AnalysisProfile | None) -> dict:
        if profile is None:
            raise ValueError("An FEM analysis profile is required for a simulation.")
        errors = profile.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return profile.to_dict()

    def inspect_model(self, analysis_profile: AnalysisProfile | None = None) -> dict:
        profile_dict = self._checked(analysis_profile) if analysis_profile is not None else None
        return self._call(_inspect_task, self.model_path, profile_dict, self.import_paths)

    def run_simulation(self, params: dict, analysis_profile: AnalysisProfile | None = None) -> dict:
        result = self.run_many([params], analysis_profile)[0]
        if isinstance(result, Exception):
            raise result
        return result

    def run_many(self, params_list: list[dict], analysis_profile: AnalysisProfile | None = None) -> list:
        """Solve several designs in parallel; each item is a result dict or the exception raised."""
        profile_dict = self._checked(analysis_profile)
        futures = [
            self._executor().submit(
                _simulate_task, self.model_path, params, profile_dict, self.ccx_binary,
                self.import_paths, self.threads, self.timeout,
            )
            for params in params_list
        ]
        results, broken = [], False
        for future in futures:
            try:
                results.append(future.result(timeout=self.timeout))
            except (FutureTimeout, BrokenProcessPool) as exc:
                broken = True
                results.append(RuntimeError(f"Evaluation worker failed or timed out: {exc!r}"))
            except Exception as exc:  # geometry, mesh or solver error for this design
                results.append(exc)
        if broken:
            self._reset_pool()
        return results

    def save_design(self, params: dict, output_path, analysis_profile: AnalysisProfile) -> Path:
        """Export STEP and STL of the design and keep its CalculiX deck and results."""
        profile_dict = self._checked(analysis_profile)
        output = Path(output_path).expanduser().resolve().with_suffix(".step")
        output.parent.mkdir(parents=True, exist_ok=True)
        self._call(_export_task, self.model_path, params, str(output.with_suffix("")), self.import_paths)
        self._call(
            _simulate_task, self.model_path, params, profile_dict, self.ccx_binary,
            self.import_paths, self.threads, self.timeout, str(output.parent / f"{output.stem}_fea"),
        )
        return output

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

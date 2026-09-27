"""Tests for the build123d + Gmsh + CalculiX backend and the shared pipeline."""

import csv
import json
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import face_selectors
from face_selectors import FaceInfo
from fem_profile import AnalysisProfile, Load, Selection
from model_intake import ModelMetadata, ParameterSpec
from optimizer import GDOptimizer, SLSQPOptimizer
from pipeline import OptimizerSettings, design_space, open_model, run_optimization

ROOT = Path(__file__).resolve().parent.parent
CANTILEVER = ROOT / "examples" / "cantilever.py"
NOMINAL = {"Length": 100.0, "Width": 20.0, "Height": 10.0}
HAS_CCX = shutil.which("ccx") is not None


def box_faces():
    """The six faces of a 100 x 20 x 10 box at the origin."""
    L, W, H = 100.0, 20.0, 10.0
    return [
        FaceInfo(1, (0, W / 2, H / 2), (0, 0, 0, 0, W, H), W * H),
        FaceInfo(2, (L, W / 2, H / 2), (L, 0, 0, L, W, H), W * H),
        FaceInfo(3, (L / 2, 0, H / 2), (0, 0, 0, L, 0, H), L * H),
        FaceInfo(4, (L / 2, W, H / 2), (0, W, 0, L, W, H), L * H),
        FaceInfo(5, (L / 2, W / 2, 0), (0, 0, 0, L, W, 0), L * W),
        FaceInfo(6, (L / 2, W / 2, H), (0, 0, H, L, W, H), L * W),
    ]


class FaceSelectorTests(unittest.TestCase):
    def test_extreme_planes_and_aliases(self):
        faces = box_faces()
        self.assertEqual(face_selectors.resolve("xmin", faces), [1])
        self.assertEqual(face_selectors.resolve("right", faces), [2])
        self.assertEqual(face_selectors.resolve("ZMAX", faces), [6])

    def test_nearest_and_explicit_faces(self):
        faces = box_faces()
        self.assertEqual(face_selectors.resolve("nearest:50,20,5", faces), [4])
        self.assertEqual(face_selectors.resolve("Face3,Face5", faces), [3, 5])

    def test_bad_selectors_raise(self):
        for selector in ("top", "Face9", "nearest:1,2"):
            with self.assertRaises(ValueError):
                face_selectors.resolve(selector, box_faces())

    def test_describe_marks_planes(self):
        table = face_selectors.describe(box_faces())
        self.assertEqual(table[0]["on"], ["xmin"])


class DesignSpaceTests(unittest.TestCase):
    def test_steps_follow_overridden_bounds_and_initial_is_clipped(self):
        metadata = ModelMetadata("part.py", [ParameterSpec("W", "W", 20.0, 10.0, 30.0)])
        initial, bounds, steps = design_space(metadata, {"W": (25.0, 35.0)})
        self.assertEqual(initial.tolist(), [25.0])
        self.assertEqual(bounds.tolist(), [[25.0, 35.0]])
        self.assertAlmostEqual(steps[0], 0.3)

    def test_part_file_step_is_used(self):
        metadata = ModelMetadata("part.py", [ParameterSpec("W", "W", 20.0, 10.0, 30.0, step=0.5)])
        self.assertEqual(design_space(metadata)[2].tolist(), [0.5])


class BatchOptimizerTests(unittest.TestCase):
    def test_parallel_batches_match_sequential_search(self):
        def make(batched):
            def objective(params):
                objective.last_evaluation = {"feasible": True}
                return float((params[0] - 2.0) ** 2 + 3 * (params[1] + 1.0) ** 2)
            objective.last_evaluation = None
            if batched:
                objective.evaluate_many = lambda points: [(objective(p), {"feasible": True}) for p in points]
                objective.parallelism = 4
            return objective

        results = []
        for batched in (False, True):
            optimizer = GDOptimizer([0.0, 0.0], [0.05, 0.05], learning_rate=0.2,
                                    bounds=[[-3, 3], [-3, 3]], max_iterations=40)
            results.append((optimizer.optimize(make(batched)), [h["objective"] for h in optimizer.history]))
        np.testing.assert_allclose(results[0][0], results[1][0])
        np.testing.assert_allclose(results[0][1], results[1][1])
        np.testing.assert_allclose(results[1][0], [2.0, -1.0], atol=0.05)


class SLSQPTests(unittest.TestCase):
    def test_finds_constrained_optimum_and_stays_feasible(self):
        # Minimize x + y subject to x * y >= 1: optimum x = y = 1, mass 2.
        def evaluation(params):
            x, y = params
            ratio = 1.0 / (x * y) - 1.0
            return {"mass": x + y, "objective": x + y, "feasible": ratio <= 0,
                    "constraint_ratios": {"stress": ratio}}

        def objective(params):
            objective.last_evaluation = evaluation(params)
            return objective.last_evaluation["objective"]

        objective.last_evaluation = None
        objective.evaluate_many = lambda points: [(evaluation(p)["objective"], evaluation(p)) for p in points]
        objective.parallelism = 4
        optimizer = SLSQPOptimizer([3.0, 2.0], [1e-3, 1e-3], bounds=[[0.1, 5.0], [0.1, 5.0]],
                                   max_iterations=50, tolerance=1e-8)
        result = optimizer.optimize(objective)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=0.01)
        self.assertEqual(optimizer.status, "converged")
        self.assertTrue(evaluation(result)["feasible"])


class PartLoaderTests(unittest.TestCase):
    def _write(self, directory, body):
        path = Path(directory) / "part.py"
        path.write_text(textwrap.dedent(body))
        return path

    def test_design_variables_and_errors(self):
        import native_backend
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, """
                from build123d import Box, Pos
                DESIGN_VARIABLES = ["A"]
                BOUNDS = {"A": (1, 5)}
                def part(A=2.0, B=3.0, label="x", split=False):
                    box = Box(A, B, 1)
                    return box + Pos(10, 0, 0) * Box(1, 1, 1) if split else box
            """)
            module = native_backend.load_part_module(path)
            parameters = native_backend.design_parameters(module)
            self.assertEqual([(p["name"], p["minimum"], p["maximum"]) for p in parameters], [("A", 1.0, 5.0)])
            self.assertAlmostEqual(native_backend.build_shape(module, {"A": 4.0}).volume, 12.0)
            with self.assertRaisesRegex(ValueError, "exactly one solid"):
                native_backend.build_shape(module, {"split": 1.0})


@unittest.skipUnless(HAS_CCX, "CalculiX (ccx) is not installed")
class NativeSolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_backend import NativeController
        cls.controller = NativeController(CANTILEVER, workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.controller.close()

    def test_tet10_node_order_matches_calculix(self):
        import gmsh
        from build123d import Box, export_brep
        import ccx_deck
        import native_backend
        with tempfile.TemporaryDirectory() as directory:
            brep = Path(directory) / "box.brep"
            export_brep(Box(10, 10, 10), str(brep))
            session = native_backend._gmsh_session(brep)
            try:
                gmsh.option.setNumber("Mesh.MeshSizeMax", 5)
                gmsh.option.setNumber("Mesh.ElementOrder", 2)
                gmsh.model.mesh.generate(3)
                tags, coords, _ = gmsh.model.mesh.getNodes()
                _, _, connectivity = gmsh.model.mesh.getElements(3)
            finally:
                session.finalize()
        lookup = np.zeros((int(tags.max()) + 1, 3))
        lookup[tags.astype(int)] = coords.reshape(-1, 3)
        elements = np.asarray(connectivity[0], dtype=int).reshape(-1, 10)[:, ccx_deck.GMSH_TO_CCX_TET10]
        for mid, (a, b) in zip(range(4, 10), ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
            np.testing.assert_allclose(
                lookup[elements[:, mid]], 0.5 * (lookup[elements[:, a]] + lookup[elements[:, b]]), atol=1e-9
            )

    def test_cantilever_matches_beam_theory(self):
        profile = AnalysisProfile()
        profile.analysis.stress_exclusion_radius = 5.0
        result = self.controller.run_simulation(NOMINAL, profile)
        self.assertAlmostEqual(result["mass"], 0.054, places=6)
        # Euler-Bernoulli 0.726 mm + shear 0.023 mm = 0.749 mm.
        self.assertAlmostEqual(result["displacement"], 0.000749, delta=0.000015)
        # Beam theory at x = 5 mm from the wall: 150 MPa x 0.95 = 142.5 MPa.
        self.assertAlmostEqual(result["stress"] / 1e6, 142.5, delta=6.0)
        self.assertEqual(result["selections"], {"Support": ["Face1"], "Load": ["Face2"]})

    def test_pressure_gives_uniform_axial_stress(self):
        profile = AnalysisProfile()
        profile.loads = [Load(type="pressure", target=Selection("End", value="xmax"), magnitude=1_000_000.0)]
        profile.analysis.stress_exclusion_radius = 5.0
        result = self.controller.run_simulation(NOMINAL, profile)
        self.assertAlmostEqual(result["stress"], 1_000_000.0, delta=100_000.0)

    def test_run_many_returns_errors_in_place(self):
        profile = AnalysisProfile()
        results = self.controller.run_many([NOMINAL, {**NOMINAL, "Width": -5.0}], profile)
        self.assertIsInstance(results[0], dict)
        self.assertIsInstance(results[1], Exception)

    @unittest.skipUnless((ROOT / "parametric_fixture.FCStd").exists(), "FreeCAD fixture not present")
    def test_matches_freecad_backend(self):
        try:
            from cad_controller import CADController
            freecad = CADController(ROOT / "parametric_fixture.FCStd")
        except RuntimeError as exc:
            self.skipTest(str(exc))
        profile = AnalysisProfile()
        native = self.controller.run_simulation(NOMINAL, profile)
        reference = freecad.run_simulation(NOMINAL, profile)
        self.assertAlmostEqual(native["mass"], reference["mass"], places=9)
        self.assertAlmostEqual(native["displacement"], reference["displacement"], delta=reference["displacement"] * 0.02)


@unittest.skipUnless(HAS_CCX, "CalculiX (ccx) is not installed")
class PipelineTests(unittest.TestCase):
    def test_end_to_end_optimization_exports_results(self):
        profile = AnalysisProfile()
        profile.analysis.stress_exclusion_radius = 5.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = open_model(CANTILEVER, profile, jobs_dir=root / "jobs", workers=2)
            try:
                initial, bounds, steps = design_space(session.metadata)
                result = run_optimization(
                    session, profile, initial, steps, bounds,
                    OptimizerSettings(max_iterations=3), export_dir=root / "exports",
                )
            finally:
                session.close()
            export = Path(result["export_dir"])
            self.assertTrue(result["feasible"])
            self.assertLess(result["evaluation"]["mass"], 0.054)
            for name in ("cantilever_optimized.step", "cantilever_optimized.stl", "report.json",
                         "history.csv", "analysis_profile.json"):
                self.assertTrue((export / name).is_file(), name)
            self.assertTrue((export / "cantilever_optimized_fea" / "job.frd").is_file())
            report = json.loads((export / "report.json").read_text())
            self.assertEqual(report["backend"], "build123d")
            self.assertEqual(set(report["best_parameters"]), {"Length", "Width", "Height"})
            with (export / "history.csv").open() as stream:
                self.assertIn("Width", next(csv.reader(stream)))


if __name__ == "__main__":
    unittest.main()

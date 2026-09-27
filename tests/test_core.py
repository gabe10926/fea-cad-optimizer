import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_intake import ModelMetadata, OptimizationJob, ParameterSpec
from optimizer import GDOptimizer
from reporting import export_results
from fem_profile import AnalysisProfile, Load, Selection
import simulation


class CoreTests(unittest.TestCase):
    def test_bounded_optimizer_converges_and_caches(self):
        calls = 0

        def objective(params):
            nonlocal calls
            calls += 1
            return float((params[0] - 2.0) ** 2)

        optimizer = GDOptimizer([0.0], [0.1], learning_rate=0.2,
                                bounds=[[0.0, 3.0]], max_iterations=30)
        result = optimizer.optimize(objective)
        self.assertAlmostEqual(result[0], 2.0, places=3)
        self.assertLess(calls, 100)
        self.assertTrue(optimizer.history)

    def test_optimizer_prefers_a_feasible_candidate(self):
        def objective(params):
            objective.last_evaluation = {"feasible": params[0] >= 0.5}
            return float((params[0] - 0.2) ** 2)

        objective.last_evaluation = None
        optimizer = GDOptimizer([0.8], [0.05], learning_rate=0.2,
                                bounds=[[0.0, 1.0]], max_iterations=20)
        result = optimizer.optimize(objective)
        self.assertGreaterEqual(result[0], 0.5)
        self.assertTrue(any(item.get("best") for item in optimizer.history))

    def test_final_candidate_verification_retries_feasible_alternatives(self):
        def objective(params):
            objective.last_evaluation = {
                "feasible": bool(params[0] >= 1.0),
                "objective": float(params[0]),
            }
            return float(params[0])

        objective.last_evaluation = None
        selected = simulation.verify_feasible_candidate(
            objective, [np.array([0.5]), np.array([1.2])]
        )
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected[0], np.array([1.2]))
        self.assertTrue(selected[1]["feasible"])

    def test_objective_uses_profile_constraints_without_closure_errors(self):
        class FakeCADController:
            results = {"stress": 200.0, "mass": 0.05, "displacement": 0.003}

            def run_simulation(self, params, analysis_profile=None):
                return self.results

            def run_many(self, params_list, analysis_profile=None):
                return [self.results for _ in params_list]

        profile = AnalysisProfile()
        profile.analysis.stress_limit = 100.0
        profile.analysis.displacement_limit = 0.002
        profile.analysis.mass_limit = 0.04
        profile.analysis.penalty_weight = 10.0
        profile.analysis.constraint_safety_factor = 1.0
        objective = simulation.make_objective(
            "part.FCStd", parameter_names=["Length"], analysis_profile=profile,
            controller=FakeCADController(),
        )
        self.assertAlmostEqual(objective(np.array([1.0])), 0.70625)
        self.assertFalse(objective.last_evaluation["feasible"])
        FakeCADController.results = {
            "stress": 100.0, "mass": 0.03, "displacement": 0.001
        }
        self.assertAlmostEqual(objective(np.array([1.0])), 0.03)
        self.assertTrue(objective.last_evaluation["feasible"])
        (value, evaluation), = objective.evaluate_many([np.array([2.0])])
        self.assertAlmostEqual(value, 0.03)
        self.assertEqual(evaluation["parameters"], {"Length": 2.0})

    def test_metadata_validation(self):
        metadata = ModelMetadata(
            "part.FCStd",
            [ParameterSpec("Width", "Width", 10, 1, 20)],
            has_fem_analysis=True,
            geometry_count=1,
        )
        self.assertEqual(metadata.validate(), [])
        metadata.parameters[0] = ParameterSpec("Width", "Width", 10, 20, 1)
        self.assertTrue(metadata.validate())

    def test_job_copies_supported_model_and_exports_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "part.FCStd"
            source.write_text("fixture")
            job = OptimizationJob.create(source, root / "jobs")
            self.assertTrue(job.model_path.is_file())
            history = [{"iteration": 0, "params": np.array([1.0]),
                        "objective": 2.0, "grad_norm": 0.1, "status": "completed"}]
            output = export_results(job.root, job.model_path, history, root / "exports")
            self.assertTrue((output / "history.json").is_file())
            self.assertEqual(json.loads((output / "history.json").read_text())[0]["params"], [1.0])

    def test_analysis_profile_round_trip_and_validation(self):
        profile = AnalysisProfile()
        encoded = profile.to_json()
        restored = AnalysisProfile.from_json(encoded)
        self.assertEqual(restored.material.name, profile.material.name)
        self.assertEqual(restored.mesh.size, profile.mesh.size)
        self.assertEqual(restored.validate(), [])

    @unittest.skipUnless(Path("parametric_fixture.FCStd").exists(), "fixture not present")
    def test_real_freecad_dummy_inspection_accepts_profile(self):
        from cad_controller import CADController
        profile = AnalysisProfile()
        metadata = CADController("parametric_fixture.FCStd").inspect_model(profile)
        self.assertTrue(metadata["profile_applied"])
        self.assertEqual(metadata["parameters"][0]["alias"], "Length")

    @unittest.skipUnless(Path("parametric_fixture.FCStd").exists(), "fixture not present")
    def test_real_dummy_structural_solve_has_response(self):
        from cad_controller import CADController
        result = CADController("parametric_fixture.FCStd").run_simulation(
            {"Length": 100, "Width": 20, "Height": 10}, AnalysisProfile()
        )
        self.assertIsNone(result["error"])
        self.assertGreater(result["mass"], 0)
        self.assertGreater(result["stress"], 0)
        self.assertGreater(result["displacement"], 0)
        self.assertAlmostEqual(result["mass"], 0.054, places=6)
        # Default 2nd-order tets: beam theory with shear gives 0.749 mm; the
        # un-excluded peak stress sits above the 150 MPa root value (singularity).
        self.assertAlmostEqual(result["displacement"], 0.000749, delta=0.00002)
        self.assertGreater(result["stress"], 140_000_000)
        self.assertLess(result["stress"], 220_000_000)

    @unittest.skipUnless(Path("parametric_fixture.FCStd").exists(), "fixture not present")
    def test_headless_save_persists_optimized_parameters_and_fem_setup(self):
        from cad_controller import CADController
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimized.FCStd"
            controller = CADController("parametric_fixture.FCStd")
            controller.save_design(
                {"Length": 90, "Width": 18, "Height": 9}, path, AnalysisProfile()
            )
            metadata = CADController(path).inspect_model()
            values = {item["alias"]: item["value"] for item in metadata["parameters"]}
            self.assertEqual(values, {"Length": 90.0, "Width": 18.0, "Height": 9.0})
            self.assertTrue(path.is_file())

    @unittest.skipUnless(Path("parametric_fixture.FCStd").exists(), "fixture not present")
    def test_pressure_profile_is_applied_in_pascal_units(self):
        from cad_controller import CADController
        profile = AnalysisProfile()
        profile.loads = [
            Load(
                type="pressure",
                target=Selection("End pressure", value="right"),
                magnitude=1_000_000.0,
            )
        ]
        # This checks the Pa conversion. 2nd-order tets resolve the Poisson
        # singularity at the fixed corners (~1.2 MPa peak), and the FreeCAD
        # backend has no stress exclusion radius, so use linear tets here.
        profile.mesh.element_order = 1
        result = CADController("parametric_fixture.FCStd").run_simulation(
            {"Length": 100, "Width": 20, "Height": 10}, profile
        )
        self.assertAlmostEqual(result["stress"], 1_000_000.0, delta=100_000.0)


if __name__ == "__main__":
    unittest.main()

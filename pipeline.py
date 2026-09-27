"""End-to-end optimization run shared by the command line and the GUI."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from controllers import create_controller
from fem_profile import AnalysisProfile
from model_intake import ModelMetadata, OptimizationJob
from optimizer import OPTIMIZERS
from reporting import export_results
from simulation import make_objective, verify_feasible_candidate

logger = logging.getLogger(__name__)

DEFAULT_STEP_FRACTION = 0.03
UNITS = {"length": "mm", "force": "N", "stress": "Pa", "displacement": "m", "mass": "kg"}


@dataclass
class OptimizerSettings:
    learning_rate: float = 0.1
    tolerance: float = 1e-4
    max_iterations: int = 30
    method: str = "slsqp"  # "slsqp" (constrained SQP) or "gradient" (penalty + gradient descent)


@dataclass
class Session:
    """An opened model: its job folder, backend controller and inspected metadata."""
    job: OptimizationJob
    controller: object
    metadata: ModelMetadata

    def close(self):
        self.controller.close()


def open_model(model, profile: AnalysisProfile | None = None, jobs_dir="jobs", workers=1,
               freecad_python=None, ccx=None) -> Session:
    """Copy the model into a job folder, start its backend and inspect it."""
    job = OptimizationJob.create(model, jobs_dir)
    controller = create_controller(
        job.model_path, freecad_python=freecad_python, ccx_binary=ccx, workers=workers,
        import_paths=[job.source_path.parent],
    )
    try:
        metadata = ModelMetadata.from_dict(controller.inspect_model(profile))
    except Exception:
        controller.close()
        raise
    job.save_metadata(metadata)
    return Session(job, controller, metadata)


def design_space(metadata: ModelMetadata, bound_overrides=None, step_overrides=None):
    """Initial vector, bounds (n, 2) and finite-difference steps after overrides."""
    bound_overrides = dict(bound_overrides or {})
    step_overrides = dict(step_overrides or {})
    names = [parameter.alias for parameter in metadata.parameters]
    unknown = (set(bound_overrides) | set(step_overrides)) - set(names)
    if unknown:
        raise ValueError(f"Unknown parameter name(s): {', '.join(sorted(unknown))}")
    initial, bounds, steps = [], [], []
    for parameter in metadata.parameters:
        low, high = bound_overrides.get(parameter.alias, (parameter.minimum, parameter.maximum))
        if not low < high:
            raise ValueError(f"Lower bound must be below upper bound for {parameter.alias}.")
        value = min(max(parameter.value, low), high)
        if value != parameter.value:
            logger.warning("%s starts at %g, clipped into [%g, %g].", parameter.alias, parameter.value, low, high)
        step = step_overrides.get(parameter.alias, parameter.step)
        initial.append(value)
        bounds.append((low, high))
        steps.append(step if step else DEFAULT_STEP_FRACTION * (high - low))
    return np.array(initial, dtype=float), np.array(bounds, dtype=float), np.array(steps, dtype=float)


def run_optimization(session: Session, profile: AnalysisProfile, initial, steps, bounds,
                     settings: OptimizerSettings | None = None, export_dir="exports",
                     callback=None) -> dict:
    """Optimize, re-verify the winner, save the optimized model and export everything.

    ``callback(iteration, params, objective, grad_norm, info)`` receives, in
    ``info``, the iteration's evaluation dict and the number of solves so far.
    """
    errors = profile.validate()
    if errors:
        raise ValueError("; ".join(errors))
    settings = settings or OptimizerSettings()
    job, controller = session.job, session.controller
    names = [parameter.alias for parameter in session.metadata.parameters]
    job.save_profile(profile)
    started = time.perf_counter()

    if settings.method not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {settings.method!r}; choose from {', '.join(OPTIMIZERS)}.")
    objective = make_objective(job.model_path, names, profile, controller=controller)
    optimizer = OPTIMIZERS[settings.method](
        initial_params=initial,
        step_sizes=steps,
        learning_rate=settings.learning_rate,
        tolerance=settings.tolerance,
        max_iterations=settings.max_iterations,
        bounds=bounds,
        callback=None if callback is None else lambda iteration, params, value, norm: callback(
            iteration, params, value, norm,
            {"evaluation": optimizer.evaluations.get(optimizer._key(params)),
             "solves": optimizer.evaluation_count},
        ),
    )
    best_params = optimizer.optimize(objective)
    verified = verify_feasible_candidate(objective, [best_params, *optimizer.feasible_candidates])
    if verified is None:
        raise RuntimeError("Final verification solves failed; no optimized model was exported.")
    best_params, evaluation = verified
    if not evaluation["feasible"]:
        optimizer.status = "no_feasible_design"
    if not np.array_equal(optimizer.history[-1]["params"], best_params):
        optimizer.history.append({
            "iteration": len(optimizer.history),
            "params": best_params.copy(),
            "objective": evaluation["objective"],
            "grad_norm": None,
            "status": optimizer.status,
            "best": True,
        })

    parameter_values = {name: float(best_params[index]) for index, name in enumerate(names)}
    suffix = ".step" if session.metadata.backend == "build123d" else ".FCStd"
    optimized_model = controller.save_design(
        parameter_values, job.root / f"{job.model_path.stem}_optimized{suffix}", profile
    )
    extra_files = [
        path for path in sorted(optimized_model.parent.glob(f"{optimized_model.stem}*"))
        if path != optimized_model
    ]
    job.history_path.write_text(json.dumps(
        [
            {**item, "params": np.asarray(item["params"]).tolist()}
            for item in optimizer.history
        ],
        indent=2,
        allow_nan=False,
    ))
    initial_evaluation = optimizer.evaluations.get(optimizer._key(np.asarray(initial, dtype=float)))
    details = {
        "backend": session.metadata.backend,
        "method": settings.method,
        "optimizer_status": optimizer.status,
        "best_parameters_by_name": parameter_values,
        "initial_parameters_by_name": {name: float(initial[index]) for index, name in enumerate(names)},
        "initial_evaluation": initial_evaluation,
        "best_evaluation": evaluation,
        "feasible": bool(evaluation["feasible"]),
        "solves": optimizer.evaluation_count,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "units": UNITS,
    }
    output = export_results(
        job.root,
        job.model_path,
        optimizer.history,
        output_root=export_dir,
        optimized_model=optimized_model,
        details=details,
        profile_path=job.profile_path,
        extra_files=extra_files,
        parameter_names=names,
    )
    return {
        "optimized_parameters": parameter_values,
        "evaluation": evaluation,
        "optimizer_status": optimizer.status,
        "feasible": bool(evaluation["feasible"]),
        "solves": optimizer.evaluation_count,
        "wall_seconds": details["wall_seconds"],
        "optimized_model": str(output / optimized_model.name),
        "export_dir": str(output),
        "report": str(output / "report.json"),
        "history": optimizer.history,
    }

"""
Simulation module - Wraps CAD+FEA pipeline into a callable objective function.
"""

import logging
from collections.abc import Iterable

import numpy as np
from controllers import create_controller
from fem_profile import AnalysisProfile

logger = logging.getLogger(__name__)


def verify_feasible_candidate(objective_function, candidates: Iterable[np.ndarray]):
    """Re-solve candidates in objective order and prefer one that remains feasible."""
    first_valid = None
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        params = np.asarray(candidate, dtype=float)
        key = tuple(np.round(params, 12))
        if key in seen:
            continue
        seen.add(key)
        value = objective_function(params)
        evaluation = getattr(objective_function, "last_evaluation", None)
        if not np.isfinite(value) or not isinstance(evaluation, dict):
            continue
        if first_valid is None:
            first_valid = (params.copy(), evaluation)
        if evaluation.get("feasible") is True:
            return params.copy(), evaluation
    return first_valid


def make_objective(model_path, parameter_names, analysis_profile: AnalysisProfile,
                   stress_limit=None, mass_limit=None, penalty_weight=None,
                   controller=None, **controller_options):
    """
    Return J(x): mass with normalized quadratic penalties for broken limits.

    The controller (CAD + mesh + solver backend) is created once and reused.
    The returned function also exposes ``evaluate_many`` so the optimizer can
    solve independent designs in parallel, and ``parallelism`` (worker count).
    """
    names = list(parameter_names)
    if not names:
        raise ValueError("At least one parameter name is required")
    errors = analysis_profile.validate()
    if errors:
        raise ValueError("; ".join(errors))
    cad = controller if controller is not None else create_controller(model_path, **controller_options)
    settings = analysis_profile.analysis
    limits = {
        "stress": stress_limit if stress_limit is not None else settings.stress_limit,
        "displacement": settings.displacement_limit,
        "mass": mass_limit if mass_limit is not None else settings.mass_limit,
    }
    weight = penalty_weight if penalty_weight is not None else settings.penalty_weight
    safety_factor = settings.constraint_safety_factor
    if weight <= 0 or not np.isfinite(weight):
        raise ValueError("penalty_weight must be finite and positive")
    for name, limit in limits.items():
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError(f"{name}_limit must be finite and positive when specified")

    def to_dict(params) -> dict:
        if len(params) != len(names):
            raise ValueError("Parameter vector length does not match model parameters")
        return {name: float(params[i]) for i, name in enumerate(names)}

    def score(params_dict: dict, results) -> tuple[float, dict]:
        if isinstance(results, Exception):
            logger.warning("Simulation failed for %s: %s", params_dict, results)
            return float("inf"), {"error": str(results), "feasible": False, "parameters": params_dict}
        stress, mass, displacement = (results.get(key) for key in ("stress", "mass", "displacement"))
        if any(value is None or not np.isfinite(value) or value < 0
               for value in (stress, mass, displacement)) or mass <= 0:
            return float("inf"), {
                "error": "Simulation returned missing, non-finite, or non-positive mass results.",
                "feasible": False,
                "parameters": params_dict,
                "results": results,
            }
        responses = {"stress": stress, "displacement": displacement, "mass": mass}
        violations, ratios = {}, {}
        for name, limit in limits.items():
            if limit is not None:
                factor = safety_factor if name in {"stress", "displacement"} else 1.0
                ratios[name] = (responses[name] * factor - limit) / limit
                violations[name] = max(0.0, ratios[name])
        violation_norm = sum(value * value for value in violations.values())
        objective = float(mass * (1.0 + weight * violation_norm))
        evaluation = {
            "feasible": violation_norm == 0.0,
            "objective": objective,
            "mass": float(mass),
            "stress": float(stress),
            "displacement": float(displacement),
            "violations": violations,
            "constraint_ratios": ratios,
            "parameters": params_dict,
            "constraint_safety_factor": safety_factor,
        }
        for key in ("peak_stress_location_mm", "nodes", "elements", "selections", "solve_seconds"):
            if key in results:
                evaluation[key] = results[key]
        return objective, evaluation

    def objective_function(params: np.ndarray) -> float:
        params_dict = to_dict(params)
        try:
            results = cad.run_simulation(params_dict, analysis_profile=analysis_profile)
        except Exception as exc:
            results = exc
        value, objective_function.last_evaluation = score(params_dict, results)
        return value

    def evaluate_many(params_list) -> list[tuple[float, dict]]:
        dicts = [to_dict(params) for params in params_list]
        return [
            score(params_dict, results)
            for params_dict, results in zip(dicts, cad.run_many(dicts, analysis_profile))
        ]

    objective_function.last_evaluation = None
    objective_function.evaluate_many = evaluate_many
    objective_function.parallelism = getattr(cad, "workers", 1)
    objective_function.controller = cad
    return objective_function

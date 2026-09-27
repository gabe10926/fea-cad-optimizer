## Bounded finite-difference optimizer for expensive simulation objectives.

from collections.abc import Callable

import numpy as np


class GDOptimizer:
    def __init__(self, initial_params, step_sizes, learning_rate=0.1,
                 tolerance=1e-4, max_iterations=50, callback=None,
                 bounds=None, cache=True, max_line_search_steps=10):
        self.params = np.array(initial_params, dtype=float)
        self.step_sizes = np.array(step_sizes, dtype=float)
        self.alpha = float(learning_rate)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.callback = callback
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.cache_enabled = cache
        self.max_line_search_steps = int(max_line_search_steps)
        self.history = []
        self.errors: list[str] = []
        self.cache: dict[tuple[float, ...], float] = {}
        self.best_params: np.ndarray | None = None
        self.best_objective = float("inf")
        self.best_feasible_params: np.ndarray | None = None
        self.best_feasible_objective = float("inf")
        self.feasible_candidate_records: dict[tuple[float, ...], tuple[float, np.ndarray]] = {}
        self._evaluation_feasibility: dict[tuple[float, ...], bool] = {}
        self.evaluations: dict[tuple[float, ...], dict] = {}
        self.evaluation_count = 0
        self._last_probes: list[tuple[np.ndarray, float]] = []
        self.status = "not_started"
        self._validate()

    def _validate(self):
        if self.params.ndim != 1 or self.params.size == 0 or self.step_sizes.shape != self.params.shape:
            raise ValueError("initial_params and step_sizes must be non-empty one-dimensional arrays of equal length")
        if np.any(~np.isfinite(self.params)) or np.any(~np.isfinite(self.step_sizes)):
            raise ValueError("Parameters and step sizes must be finite")
        if not np.isfinite(self.alpha) or self.alpha <= 0 or not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Learning rate and tolerance must be finite and positive")
        if np.any(self.step_sizes <= 0) or self.max_iterations < 1 or self.max_line_search_steps < 1:
            raise ValueError("Step sizes and iteration limits must be positive")
        if self.bounds is not None:
            if self.bounds.shape != (len(self.params), 2) or np.any(~np.isfinite(self.bounds)):
                raise ValueError("bounds must be a finite array with shape (parameter_count, 2)")
            if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
                raise ValueError("Each lower bound must be less than its upper bound")
            if np.any(self.params < self.bounds[:, 0]) or np.any(self.params > self.bounds[:, 1]):
                raise ValueError("Initial parameters must be within bounds")

    @staticmethod
    def _key(params) -> tuple[float, ...]:
        return tuple(np.round(params, 12))

    def _evaluate(self, objective_function: Callable[[np.ndarray], float],
                  params: np.ndarray | None = None) -> float:
        params = self.params if params is None else np.asarray(params, dtype=float)
        key = self._key(params)
        if self.cache_enabled and key in self.cache:
            self._record_candidate(params, self.cache[key], self._evaluation_feasibility.get(key))
            return self.cache[key]
        value = float(objective_function(params.copy()))
        return self._store(params, value, getattr(objective_function, "last_evaluation", None))

    def _evaluate_many(self, objective_function, points: list[np.ndarray]) -> list[float]:
        """Evaluate several designs, in parallel when the objective supports it."""
        batch = getattr(objective_function, "evaluate_many", None)
        pending: dict[tuple[float, ...], np.ndarray] = {}
        for point in points:
            key = self._key(point)
            if not (self.cache_enabled and key in self.cache):
                pending.setdefault(key, np.asarray(point, dtype=float))
        solved: dict[tuple[float, ...], float] = {}
        if batch is not None and len(pending) > 1:
            outcomes = batch([point.copy() for point in pending.values()])
            for (key, point), (value, evaluation) in zip(pending.items(), outcomes):
                solved[key] = self._store(point, float(value), evaluation)
        return [
            solved[key] if key in solved else self._evaluate(objective_function, point)
            for point, key in ((point, self._key(point)) for point in points)
        ]

    def _store(self, params, value, evaluation) -> float:
        key = self._key(params)
        self.evaluation_count += 1
        if not np.isfinite(value):
            value = float("inf")
        if self.cache_enabled:
            self.cache[key] = value
        feasible = evaluation.get("feasible") if isinstance(evaluation, dict) else None
        feasible = None if feasible is None else bool(feasible)
        if isinstance(evaluation, dict):
            self.evaluations[key] = evaluation
            if not np.isfinite(value) and evaluation.get("error"):
                self.errors.append(str(evaluation["error"]))
        self._evaluation_feasibility[key] = feasible
        self._record_candidate(params, value, feasible)
        return value

    def _record_candidate(self, params, value, feasible):
        if value < self.best_objective:
            self.best_objective = value
            self.best_params = params.copy()
        if feasible is True and value < self.best_feasible_objective:
            self.best_feasible_objective = value
            self.best_feasible_params = params.copy()
        if feasible is True:
            key = tuple(np.round(params, 12))
            current = self.feasible_candidate_records.get(key)
            if current is None or value < current[0]:
                self.feasible_candidate_records[key] = (value, params.copy())

    @property
    def feasible_candidates(self) -> list[np.ndarray]:
        return [
            params.copy()
            for _, params in sorted(
                self.feasible_candidate_records.values(), key=lambda item: item[0]
            )
        ]

    def _gradient(self, objective_function, base_objective: float) -> np.ndarray:
        """Central differences; all 2n probes are submitted as one batch."""
        gradient = np.zeros_like(self.params)
        probes = []
        for index, step in enumerate(self.step_sizes):
            plus = self.params.copy()
            minus = self.params.copy()
            plus[index] += step
            minus[index] -= step
            if self.bounds is not None:
                plus[index] = min(plus[index], self.bounds[index, 1])
                minus[index] = max(minus[index], self.bounds[index, 0])
            probes.append((index, plus, minus))
        points = [point for _, plus, minus in probes for point in (plus, minus)]
        values = self._evaluate_many(objective_function, points)
        self._last_probes = list(zip(points, values))
        for (index, plus, minus), f_plus, f_minus in zip(probes, values[0::2], values[1::2]):
            denominator = plus[index] - minus[index]
            if denominator == 0:
                continue
            if np.isfinite(f_plus) and np.isfinite(f_minus):
                gradient[index] = (f_plus - f_minus) / denominator
            elif np.isfinite(base_objective) and np.isfinite(f_plus) and plus[index] != self.params[index]:
                gradient[index] = (f_plus - base_objective) / (plus[index] - self.params[index])
            elif np.isfinite(base_objective) and np.isfinite(f_minus) and minus[index] != self.params[index]:
                gradient[index] = (base_objective - f_minus) / (self.params[index] - minus[index])
        return gradient

    def _projected_gradient(self, gradient: np.ndarray, scales: np.ndarray) -> np.ndarray:
        scaled = gradient * scales
        if self.bounds is None:
            return scaled
        at_lower = np.isclose(self.params, self.bounds[:, 0], rtol=0, atol=1e-12)
        at_upper = np.isclose(self.params, self.bounds[:, 1], rtol=0, atol=1e-12)
        scaled[at_lower & (scaled > 0)] = 0.0
        scaled[at_upper & (scaled < 0)] = 0.0
        return scaled

    def optimize(self, objective_function):
        """Run projected, scaled gradient descent with backtracking."""
        scales = (self.bounds[:, 1] - self.bounds[:, 0]
                  if self.bounds is not None else np.maximum(np.abs(self.params), 1.0))
        scales = np.maximum(scales, 1e-12)
        self.status = "max_iterations"

        for iteration in range(self.max_iterations):
            objective = self._evaluate(objective_function)
            if not np.isfinite(objective):
                self.status = "no_feasible_evaluation"
                break

            gradient = self._gradient(objective_function, objective)
            projected_gradient = self._projected_gradient(gradient, scales)
            grad_norm = float(np.linalg.norm(projected_gradient))
            if not np.isfinite(grad_norm):
                self.status = "numerical_failure"
                break

            if grad_norm < self.tolerance:
                self.status = "converged"

            entry = {
                "iteration": iteration,
                "params": self.params.copy(),
                "objective": objective,
                "grad_norm": grad_norm,
                "status": self.status if self.status != "max_iterations" else "running",
            }
            self.history.append(entry)
            if self.callback:
                self.callback(iteration, self.params.copy(), objective, grad_norm)
            if self.status == "converged":
                break

            direction = -projected_gradient / max(abs(objective), 1e-12)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0:
                self.status = "stalled"
                entry["status"] = self.status
                break
            direction /= max(1.0, float(np.max(np.abs(direction))))
            accepted = False
            fractions = [self.alpha * 0.5 ** trial for trial in range(self.max_line_search_steps)]
            batch_size = max(1, int(getattr(objective_function, "parallelism", 1)))
            for start in range(0, len(fractions), batch_size):
                candidates = []
                for step_fraction in fractions[start:start + batch_size]:
                    candidate = self.params + scales * direction * step_fraction
                    if self.bounds is not None:
                        candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])
                    if np.allclose(candidate, self.params, rtol=0, atol=1e-14):
                        break
                    candidates.append(candidate)
                if not candidates:
                    break
                # Trials are solved together but accepted in halving order, so the
                # result is identical to a sequential backtracking search.
                for candidate, candidate_objective in zip(
                    candidates, self._evaluate_many(objective_function, candidates)
                ):
                    if np.isfinite(candidate_objective) and candidate_objective < objective:
                        self.params = candidate
                        accepted = True
                        break
                if accepted or len(candidates) < len(fractions[start:start + batch_size]):
                    break

            if not accepted:
                # Near an active limit the central difference straddles the penalty
                # wall and can point the wrong way. Fall back to the best
                # finite-difference probe if it improved J (a compass-search step).
                better = [(value, point) for point, value in self._last_probes
                          if np.isfinite(value) and value < objective]
                if better:
                    self.params = min(better, key=lambda item: item[0])[1].copy()
                    entry["step"] = "probe"
                    continue
                self.status = "stalled"
                entry["status"] = self.status
                break

        return self._finish()

    def _finish(self) -> np.ndarray:
        """Pick the best feasible design seen, or the best overall if none was feasible."""
        if self.best_params is None:
            detail = f" Last simulation error: {self.errors[-1]}" if self.errors else ""
            raise RuntimeError("Optimization produced no finite objective evaluations." + detail)
        selected_params = (
            self.best_feasible_params
            if self.best_feasible_params is not None else self.best_params
        )
        selected_objective = (
            self.best_feasible_objective
            if self.best_feasible_params is not None else self.best_objective
        )
        if (self.best_feasible_params is None
                and any(value is not None for value in self._evaluation_feasibility.values())):
            self.status = "no_feasible_design"
        if self.history and self.status in {
            "converged", "stalled", "no_feasible_evaluation", "no_feasible_design", "numerical_failure"
        }:
            self.history[-1]["status"] = self.status
        elif self.history and len(self.history) == self.max_iterations:
            self.history[-1]["status"] = "max_iterations"
        if not self.history or not np.array_equal(self.history[-1]["params"], selected_params):
            self.history.append({
                "iteration": len(self.history),
                "params": selected_params.copy(),
                "objective": selected_objective,
                "grad_norm": None,
                "status": self.status,
                "best": True,
            })
        elif self.history:
            self.history[-1]["best"] = True
            self.history[-1]["objective"] = selected_objective
        self.best_params = selected_params.copy()
        self.best_objective = selected_objective
        return selected_params.copy()


class SLSQPOptimizer(GDOptimizer):
    """Sequential quadratic programming on mass with explicit limit constraints.

    Works in normalized coordinates u = (x - l) / (u - l), so every variable spans
    [0, 1]. Minimizes m(x) / m(x0) subject to 1 - s * g_k(x) / L_k >= 0 for each
    limit (tightened by CONSTRAINT_MARGIN). Gradients of mass and of every constraint come from the same batch of
    central-difference probes, solved in parallel and cached, so one SLSQP
    gradient costs 2n solves in total rather than 2n per function.
    """

    FAILED_OBJECTIVE = 10.0
    # SLSQP converges onto active constraints, where the verdict "feasible" would
    # hinge on the 4th significant digit of a solve. Aim 0.2% inside every limit.
    CONSTRAINT_MARGIN = 0.002

    def optimize(self, objective_function):
        from scipy.optimize import minimize

        if self.bounds is None:
            raise ValueError("SLSQP needs bounds for every parameter.")
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        span = upper - lower
        steps_u = np.minimum(self.step_sizes / span, 0.25)
        to_x = lambda u: lower + np.clip(u, 0.0, 1.0) * span

        self._evaluate(objective_function)
        start = self.evaluations.get(self._key(self.params))
        if not isinstance(start, dict) or "mass" not in start:
            self.status = "no_feasible_evaluation"
            return self._finish()
        mass0 = start["mass"]
        names = sorted(start.get("constraint_ratios", {}))

        def record(u):
            x = to_x(u)
            self._evaluate(objective_function, x)
            return self.evaluations.get(self._key(x))

        def values(evaluation):
            if not isinstance(evaluation, dict) or "mass" not in evaluation:
                return self.FAILED_OBJECTIVE, -np.ones(len(names))
            ratios = evaluation["constraint_ratios"]
            return evaluation["mass"] / mass0, -np.array([ratios[name] + self.CONSTRAINT_MARGIN for name in names])

        def gradients(u):
            u = np.clip(u, 0.0, 1.0)
            pairs = []
            for index, step in enumerate(steps_u):
                plus, minus = u.copy(), u.copy()
                plus[index] = min(1.0, u[index] + step)
                minus[index] = max(0.0, u[index] - step)
                pairs.append((plus, minus))
            self._evaluate_many(objective_function, [to_x(point) for pair in pairs for point in pair])
            f_grad = np.zeros(len(u))
            c_grad = np.zeros((len(names), len(u)))
            for index, (plus, minus) in enumerate(pairs):
                f_plus, c_plus = values(self.evaluations.get(self._key(to_x(plus))))
                f_minus, c_minus = values(self.evaluations.get(self._key(to_x(minus))))
                width = plus[index] - minus[index]
                if width > 0:
                    f_grad[index] = (f_plus - f_minus) / width
                    c_grad[:, index] = (c_plus - c_minus) / width
            return f_grad, c_grad

        iteration = 0

        def callback(u):
            nonlocal iteration
            x = to_x(u)
            evaluation = record(u)
            objective = evaluation.get("objective", float("inf")) if isinstance(evaluation, dict) else float("inf")
            grad_norm = float(np.linalg.norm(gradients(u)[0]))
            self.params = x
            self.history.append({
                "iteration": iteration, "params": x.copy(), "objective": objective,
                "grad_norm": grad_norm, "status": "running",
            })
            if self.callback:
                self.callback(iteration, x.copy(), objective, grad_norm)
            iteration += 1

        u0 = (self.params - lower) / span
        callback(u0)
        constraints = [{
            "type": "ineq",
            "fun": lambda u: values(record(u))[1],
            "jac": lambda u: gradients(u)[1],
        }] if names else []
        result = minimize(
            lambda u: values(record(u))[0], u0,
            jac=lambda u: gradients(u)[0],
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(u0),
            constraints=constraints,
            callback=callback,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        self.params = to_x(result.x)
        record(result.x)
        self.status = "converged" if result.success else (
            "max_iterations" if result.status == 9 else "stalled"
        )
        self.message = str(result.message)
        return self._finish()


OPTIMIZERS = {"slsqp": SLSQPOptimizer, "gradient": GDOptimizer}

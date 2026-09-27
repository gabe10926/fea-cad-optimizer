"""Portable FEM setup profiles used by the GUI and FreeCAD worker."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Material:
    name: str = "Aluminum 6061-T6"
    youngs_modulus: float = 68_900_000_000.0
    poisson_ratio: float = 0.33
    density: float = 2700.0
    yield_strength: float = 276_000_000.0


@dataclass
class MeshSettings:
    method: str = "gmsh"
    element_order: int = 2
    size: float = 4.0
    minimum_size: float = 1.0
    maximum_size: float = 8.0
    # Elements per full circle on curved faces (Gmsh MeshSizeFromCurvature). 0 disables it.
    # Refines holes and fillets, but the small size also spreads along long fillets.
    curvature_elements: int = 0


@dataclass
class Selection:
    name: str
    selector: str = "named"
    value: str = ""


@dataclass
class BoundaryCondition:
    type: str = "fixed"
    target: Selection = field(default_factory=lambda: Selection("Support", value="left"))


@dataclass
class Load:
    type: str = "force"
    target: Selection = field(default_factory=lambda: Selection("Load", value="right"))
    vector: list[float] = field(default_factory=lambda: [0.0, -1000.0, 0.0])
    magnitude: float = 1000.0


@dataclass
class AnalysisSettings:
    type: str = "static"
    solver: str = "calculix"
    stress_limit: float | None = 150_000_000.0
    displacement_limit: float | None = 0.002
    mass_limit: float | None = None
    penalty_weight: float = 1000.0
    constraint_safety_factor: float = 1.05
    # Nodes within this distance (mm) of a fixed face are ignored for peak stress,
    # which keeps support singularities out of the stress constraint. 0 disables it.
    stress_exclusion_radius: float = 0.0


@dataclass
class AnalysisProfile:
    material: Material = field(default_factory=Material)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    boundary_conditions: list[BoundaryCondition] = field(
        default_factory=lambda: [BoundaryCondition()]
    )
    loads: list[Load] = field(default_factory=lambda: [Load()])
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisProfile":
        material = Material(**data.get("material", {}))
        mesh = MeshSettings(**data.get("mesh", {}))
        boundary_conditions = [
            BoundaryCondition(
                type=item.get("type", "fixed"),
                target=Selection(**item.get("target", {})),
            )
            for item in data.get("boundary_conditions", [])
        ]
        loads = [
            Load(
                type=item.get("type", "force"),
                target=Selection(**item.get("target", {})),
                vector=[float(v) for v in item.get("vector", [0.0, -1000.0, 0.0])],
                magnitude=float(item.get("magnitude", 1000.0)),
            )
            for item in data.get("loads", [])
        ]
        return cls(
            material=material,
            mesh=mesh,
            boundary_conditions=boundary_conditions or [BoundaryCondition()],
            loads=loads or [Load()],
            analysis=AnalysisSettings(**data.get("analysis", {})),
        )

    @classmethod
    def from_json(cls, value: str) -> "AnalysisProfile":
        return cls.from_dict(json.loads(value))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if (
            not math.isfinite(self.material.youngs_modulus)
            or not math.isfinite(self.material.density)
            or not math.isfinite(self.material.yield_strength)
            or self.material.youngs_modulus <= 0
            or self.material.density <= 0
            or self.material.yield_strength <= 0
        ):
            errors.append("Material modulus, density, and yield strength must be positive.")
        if not math.isfinite(self.material.poisson_ratio) or not 0 <= self.material.poisson_ratio < 0.5:
            errors.append("Poisson ratio must be at least 0 and less than 0.5.")
        if any(not math.isfinite(value) for value in (
            self.mesh.size, self.mesh.minimum_size, self.mesh.maximum_size
        )) or self.mesh.size <= 0 or self.mesh.minimum_size <= 0 or self.mesh.maximum_size <= 0:
            errors.append("Mesh sizes must be positive.")
        if not self.mesh.minimum_size <= self.mesh.size <= self.mesh.maximum_size:
            errors.append("Mesh size must be between the minimum and maximum sizes.")
        if self.mesh.curvature_elements < 0:
            errors.append("Mesh curvature elements must not be negative.")
        if self.mesh.element_order not in {1, 2}:
            errors.append("Mesh element order must be 1 or 2.")
        if not self.boundary_conditions:
            errors.append("At least one boundary condition is required.")
        if not self.loads:
            errors.append("At least one load is required.")
        if self.analysis.type.lower() != "static" or self.analysis.solver.lower() != "calculix":
            errors.append("Only static CalculiX analysis is supported.")
        for label, value in (
            ("Stress limit", self.analysis.stress_limit),
            ("Displacement limit", self.analysis.displacement_limit),
            ("Mass limit", self.analysis.mass_limit),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                errors.append(f"{label} must be positive when specified.")
        if not math.isfinite(self.analysis.penalty_weight) or self.analysis.penalty_weight <= 0:
            errors.append("Objective penalty weight must be finite and positive.")
        if (
            not math.isfinite(self.analysis.constraint_safety_factor)
            or self.analysis.constraint_safety_factor < 1.0
        ):
            errors.append("Constraint safety factor must be finite and at least 1.")
        if (
            not math.isfinite(self.analysis.stress_exclusion_radius)
            or self.analysis.stress_exclusion_radius < 0
        ):
            errors.append("Stress exclusion radius must be finite and not negative.")
        for load in self.loads:
            if load.type.lower() not in {"force", "pressure"}:
                errors.append(f"Unsupported load type: {load.type}.")
            if len(load.vector) != 3 or any(not math.isfinite(value) for value in load.vector):
                errors.append(f"Load {load.type} vector must have three components.")
            elif load.type.lower() == "force" and math.sqrt(sum(value * value for value in load.vector)) == 0:
                errors.append(f"Force load {load.type} direction must be non-zero.")
            if not math.isfinite(load.magnitude) or load.magnitude <= 0:
                errors.append(f"Load {load.type} magnitude must be finite and positive.")
        if any(condition.type.lower() != "fixed" for condition in self.boundary_conditions):
            errors.append("Only fixed boundary conditions are supported.")
        return errors

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisProfile":
        return cls.from_json(Path(path).read_text())

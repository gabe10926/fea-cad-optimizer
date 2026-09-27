"""Model metadata, validation, and isolated optimization jobs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from controllers import backend_for
from fem_profile import AnalysisProfile


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    alias: str
    value: float
    minimum: float
    maximum: float
    unit: str = ""
    editable: bool = True
    step: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterSpec":
        return cls(
            name=str(data["name"]),
            alias=str(data.get("alias", data["name"])),
            value=float(data["value"]),
            minimum=float(data.get("minimum", data["value"] * 0.1)),
            maximum=float(data.get("maximum", data["value"] * 10.0)),
            unit=str(data.get("unit", "")),
            editable=bool(data.get("editable", True)),
            step=float(data["step"]) if data.get("step") is not None else None,
        )


@dataclass
class ModelMetadata:
    model_path: str
    parameters: list[ParameterSpec] = field(default_factory=list)
    has_fem_analysis: bool = False
    geometry_count: int = 0
    mass: float | None = None
    warnings: list[str] = field(default_factory=list)
    backend: str = ""
    faces: list[dict] = field(default_factory=list)
    selections: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        return cls(
            model_path=str(data["model_path"]),
            parameters=[ParameterSpec.from_dict(item) for item in data.get("parameters", [])],
            has_fem_analysis=bool(data.get("has_fem_analysis", False)),
            geometry_count=int(data.get("geometry_count", 0)),
            mass=float(data["mass"]) if data.get("mass") is not None else None,
            warnings=[str(item) for item in data.get("warnings", [])],
            backend=str(data.get("backend", "")),
            faces=list(data.get("faces", [])),
            selections=dict(data.get("selections", {})),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.parameters:
            errors.append("No editable spreadsheet parameters were found.")
        if not self.has_fem_analysis:
            errors.append("No FEM analysis object was found.")
        if self.geometry_count == 0:
            errors.append("The model contains no geometry.")
        seen: set[str] = set()
        for parameter in self.parameters:
            if parameter.alias in seen:
                errors.append(f"Duplicate parameter alias: {parameter.alias}")
            seen.add(parameter.alias)
            if parameter.minimum >= parameter.maximum:
                errors.append(f"Invalid bounds for {parameter.name}.")
        return errors


class OptimizationJob:
    """Owns a copied model and all artifacts produced for one optimization."""

    def __init__(self, root: Path, model_path: Path, source_path: Path | None = None):
        self.root = root
        self.model_path = model_path
        self.source_path = source_path or model_path
        self.metadata_path = root / "model.json"
        self.history_path = root / "history.json"
        self.profile_path = root / "analysis_profile.json"

    @classmethod
    def create(cls, source: str | Path, jobs_root: str | Path = "jobs") -> "OptimizationJob":
        source_path = Path(source).expanduser().resolve()
        backend_for(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Model file not found: {source_path}")
        root = Path(jobs_root).expanduser().resolve() / source_path.stem
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / source_path.name
        shutil.copy2(source_path, model_path)
        return cls(root, model_path, source_path)

    def save_metadata(self, metadata: ModelMetadata) -> None:
        self.metadata_path.write_text(json.dumps(asdict(metadata), indent=2))

    def load_metadata(self) -> ModelMetadata:
        return ModelMetadata.from_dict(json.loads(self.metadata_path.read_text()))

    def save_profile(self, profile: AnalysisProfile) -> None:
        profile.save(self.profile_path)

    def load_profile(self) -> AnalysisProfile:
        return AnalysisProfile.load(self.profile_path)

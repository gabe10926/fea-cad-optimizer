"""Export optimization history and a concise result report."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


def export_results(
    job_root: str | Path,
    model_path: str | Path,
    history: list[dict[str, Any]],
    output_root: str | Path = "exports",
    optimized_model: str | Path | None = None,
    details: dict[str, Any] | None = None,
    profile_path: str | Path | None = None,
    extra_files: list[str | Path] | None = None,
    parameter_names: list[str] | None = None,
) -> Path:
    job_root = Path(job_root)
    output = Path(output_root).expanduser().resolve() / job_root.name
    output.mkdir(parents=True, exist_ok=True)
    export_model = Path(optimized_model) if optimized_model is not None else Path(model_path)
    if not export_model.is_file():
        raise FileNotFoundError(f"Optimized model not found: {export_model}")
    shutil.copy2(export_model, output / export_model.name)
    for extra in map(Path, extra_files or []):
        if extra.is_dir():
            shutil.copytree(extra, output / extra.name, dirs_exist_ok=True)
        elif extra.is_file():
            shutil.copy2(extra, output / extra.name)
    if profile_path is not None:
        profile_source = Path(profile_path)
        if profile_source.is_file():
            shutil.copy2(profile_source, output / "analysis_profile.json")
    serializable = [
        {key: value.tolist() if hasattr(value, "tolist") else value for key, value in item.items()}
        for item in history
    ]
    (output / "history.json").write_text(json.dumps(serializable, indent=2, default=float))
    if serializable:
        with (output / "history.csv").open("w", newline="") as stream:
            names = list(parameter_names or [])
            fields = ["iteration", "objective", "grad_norm", "status", *names, "params"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in serializable:
                row = {field: item.get(field, "") for field in fields if field not in names}
                if isinstance(row["params"], list):
                    row.update(zip(names, row["params"]))
                    row["params"] = json.dumps(row["params"])
                writer.writerow(row)
    finite_history = [
        item for item in serializable
        if isinstance(item.get("objective"), (int, float))
        and item["objective"] == item["objective"]
    ]
    best = min(finite_history, key=lambda item: item["objective"]) if finite_history else {}
    report = {
        "source_model": Path(model_path).name,
        "optimized_model": export_model.name,
        "iterations": len(history),
        "best_objective": best.get("objective"),
        "best_parameters": best.get("params", []),
        "status": best.get("status", "completed"),
    }
    if details:
        report.update(details)
        final_evaluation = details.get("best_evaluation")
        if isinstance(final_evaluation, dict):
            report["best_objective"] = final_evaluation.get(
                "objective", report["best_objective"]
            )
            report["best_parameters"] = details.get(
                "best_parameters_by_name",
                final_evaluation.get("parameters", report["best_parameters"]),
            )
        if details.get("optimizer_status"):
            report["status"] = details["optimizer_status"]
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return output

"""Pick the CAD/FEA backend for a model file."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_MODEL_EXTENSIONS = {".py": "build123d", ".fcstd": "freecad"}


def backend_for(model_path) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix not in SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(
            f"Unsupported model type {suffix!r}. Use a build123d part script (.py) "
            "or a parametric FreeCAD document (.FCStd)."
        )
    return SUPPORTED_MODEL_EXTENSIONS[suffix]


def create_controller(model_path, *, freecad_python=None, ccx_binary=None, workers=1,
                      import_paths=(), timeout=600):
    """Return a controller with inspect_model / run_simulation / run_many / save_design / close."""
    if backend_for(model_path) == "build123d":
        from native_backend import NativeController
        return NativeController(
            model_path, ccx_binary=ccx_binary, workers=workers, timeout=timeout,
            import_paths=import_paths,
        )
    from cad_controller import CADController
    return CADController(
        model_path, freecad_python=freecad_python, ccx_binary=ccx_binary,
        workers=workers, timeout=timeout,
    )

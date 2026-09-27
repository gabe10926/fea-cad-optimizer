"""
CAD Controller - Manages FreeCAD worker subprocess.
Communicates with cad_worker.py via JSON files.
Handles Python 3.12+ <-> Python 3.11 compatibility.
"""

import subprocess
import json
import os
import shutil
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fem_profile import AnalysisProfile

logger = logging.getLogger(__name__)

FREECAD_PYTHON_CANDIDATES = [
    "~/.local/bin/squashfs-root/usr/bin/freecadcmd",
    "freecadcmd",
    "/usr/bin/freecadcmd",
    "/usr/bin/FreeCADCmd",
]


def find_freecad_python():
    """Auto-detect a compatible FreeCAD/Python 3.11 interpreter."""
    for cmd in FREECAD_PYTHON_CANDIDATES:
        expanded = os.path.expanduser(cmd)
        executable = shutil.which(expanded) or (
            expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
        )
        if executable is None:
            continue
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout + result.stderr
                if "3.11" in version or "FreeCAD" in version:
                    logger.info(f"Found FreeCAD Python: {cmd} ({version.strip()})")
                    return executable
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    return None


class CADController:
    """Subprocess-based CAD controller that runs cad_worker.py."""

    def __init__(self, model_path, worker_script="cad_worker.py", freecad_python=None,
                 ccx_binary=None, timeout=600, workers=1):
        self.model_path = str(Path(model_path).resolve())
        worker_path = Path(worker_script)
        if not worker_path.is_absolute():
            worker_path = Path(__file__).parent / worker_path
        self.worker_script = str(worker_path.resolve())

        if freecad_python is None:
            freecad_python = find_freecad_python()
        if freecad_python is None:
            raise RuntimeError(
                "Could not find a compatible FreeCAD Python interpreter. "
                "Install FreeCAD or pass freecad_python= explicitly."
            )
        self.freecad_python = freecad_python
        self.ccx_binary = shutil.which(ccx_binary or os.environ.get("CCX_BINARY", "ccx"))
        self.timeout = timeout
        self.workers = max(1, int(workers))

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.worker_script):
            raise FileNotFoundError(f"Worker script not found: {self.worker_script}")
        if self.ccx_binary is None:
            raise RuntimeError("CalculiX executable 'ccx' was not found; install CalculiX or set CCX_BINARY.")
        self._run_worker([], {"FEA_CAD_PROBE": "1"})

    def _run_worker(self, args: list[str], environment: dict[str, str] | None = None) -> dict:
        command = [self.freecad_python, self.worker_script]
        try:
            result = subprocess.run(
                [*command, *args],
                capture_output=True, text=True, timeout=self.timeout,
                env={**os.environ, **(environment or {})},
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"FreeCAD worker exceeded the {self.timeout}-second timeout.") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not start FreeCAD worker '{self.freecad_python}': {exc}") from exc
        if result.returncode != 0:
            diagnostics = (result.stderr + "\n" + result.stdout).strip()[-4000:]
            raise RuntimeError(f"Worker failed (exit {result.returncode}):\n{diagnostics}")
        return {"stdout": result.stdout, "stderr": result.stderr}

    def inspect_model(self, analysis_profile: AnalysisProfile | None = None) -> dict:
        """Inspect a FreeCAD model and return serializable parameter metadata."""
        if analysis_profile is not None:
            errors = analysis_profile.validate()
            if errors:
                raise ValueError("; ".join(errors))
        with tempfile.TemporaryDirectory() as tmp:
            results_file = os.path.join(tmp, "model.json")
            environment = {
                "FEA_CAD_INSPECT": "1",
                "FEA_CAD_RESULTS": results_file,
                "FEA_CAD_MODEL": self.model_path,
                "TMPDIR": tmp,
            }
            if analysis_profile is not None:
                profile_file = os.path.join(tmp, "analysis_profile.json")
                analysis_profile.save(profile_file)
                environment["FEA_CAD_PROFILE"] = profile_file
            self._run_worker([], environment)
            if not os.path.exists(results_file):
                raise RuntimeError("Worker did not produce model metadata")
            with open(results_file) as stream:
                results = json.load(stream)
            if results.get("error"):
                detail = results.get("traceback")
                raise RuntimeError(
                    f"{results['error']}\n{detail}" if detail else results["error"]
                )
            return results

    def run_simulation(self, params: dict, analysis_profile: AnalysisProfile | None = None) -> dict:
        """
        Run a full simulation cycle: update params -> mesh -> solve -> extract results.

        Args:
            params: Parameter dict, e.g. {"Height": 10.5}

        Returns:
            {"stress": float, "mass": float, "displacement": float, "error": str | None}
        """
        if analysis_profile is None:
            raise ValueError("An FEM analysis profile is required for a simulation.")
        errors = analysis_profile.validate()
        if errors:
            raise ValueError("; ".join(errors))
        with tempfile.TemporaryDirectory() as tmp:
            params_file = os.path.join(tmp, "params.json")
            results_file = os.path.join(tmp, "results.json")

            with open(params_file, "w") as f:
                json.dump(params, f)
            profile_file = os.path.join(tmp, "analysis_profile.json")
            if analysis_profile is not None:
                analysis_profile.save(profile_file)

            logger.debug("Running FreeCAD worker for %s", self.model_path)
            environment = {
                "FEA_CAD_PARAMS": params_file,
                "FEA_CAD_RESULTS": results_file,
                "FEA_CAD_MODEL": self.model_path,
                "FEA_CAD_CCX": self.ccx_binary,
                "TMPDIR": tmp,
            }
            if analysis_profile is not None:
                environment["FEA_CAD_PROFILE"] = profile_file
            self._run_worker([], environment)

            if not os.path.exists(results_file):
                raise RuntimeError("Worker did not produce results file")

            with open(results_file) as f:
                results = json.load(f)

            if results.get("error"):
                detail = results.get("traceback")
                raise RuntimeError(
                    f"Worker error: {results['error']}\n{detail}"
                    if detail else f"Worker error: {results['error']}"
                )

            return results

    def run_many(self, params_list: list[dict], analysis_profile: AnalysisProfile | None = None) -> list:
        """Solve several designs; each FreeCAD worker is its own process, so threads suffice."""
        def run(params):
            try:
                return self.run_simulation(params, analysis_profile)
            except Exception as exc:
                return exc
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(run, params_list))

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def save_design(self, params: dict, output_path: str | Path,
                    analysis_profile: AnalysisProfile) -> Path:
        """Save the parameterized model and applied FEM setup without opening the GUI."""
        errors = analysis_profile.validate()
        if errors:
            raise ValueError("; ".join(errors))
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            params_file = os.path.join(tmp, "params.json")
            results_file = os.path.join(tmp, "results.json")
            profile_file = os.path.join(tmp, "analysis_profile.json")
            with open(params_file, "w") as stream:
                json.dump(params, stream)
            analysis_profile.save(profile_file)
            environment = {
                "FEA_CAD_PARAMS": params_file,
                "FEA_CAD_RESULTS": results_file,
                "FEA_CAD_MODEL": self.model_path,
                "FEA_CAD_PROFILE": profile_file,
                "FEA_CAD_EXPORT": str(output),
                "TMPDIR": tmp,
            }
            self._run_worker([], environment)
            if not os.path.isfile(output):
                raise RuntimeError("FreeCAD worker reported success but did not save the optimized model.")
            if os.path.exists(results_file):
                with open(results_file) as stream:
                    results = json.load(stream)
                if results.get("error"):
                    detail = results.get("traceback")
                    raise RuntimeError(
                        f"Worker error: {results['error']}\n{detail}"
                        if detail else f"Worker error: {results['error']}"
                    )
        return output

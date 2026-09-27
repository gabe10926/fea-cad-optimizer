"""
FEA CAD Optimizer - Main Entry Point
"""

import argparse
import sys
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# build123d logs every fillet call at INFO, which floods the terminal during a run.
logging.getLogger("build123d").setLevel(logging.WARNING)



def preflight():
    """Check that Python dependencies and the CalculiX solver are present."""
    for module in ("PyQt6", "numpy", "matplotlib", "scipy", "build123d", "gmsh"):
        try:
            __import__(module)
        except ImportError:
            logger.error(f"Missing Python package: {module}  (pip install -r requirements.txt)")
            return False
    if shutil.which("ccx") is None:
        logger.error("CalculiX 'ccx' not found on PATH (e.g. sudo apt install calculix-ccx)")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="FEA CAD Optimizer GUI")
    parser.add_argument("model", nargs="?", help="Part to open on start (.py or .FCStd)")
    parser.add_argument("--profile", help="FEM profile JSON to load on start")
    args, qt_args = parser.parse_known_args()
    if not preflight():
        return 1

    from PyQt6.QtWidgets import QApplication
    from gui import OptimizerGUI

    app = QApplication([sys.argv[0], *qt_args])
    window = OptimizerGUI()
    window.show()
    if args.profile:
        window.load_profile_path(args.profile)
    if args.model:
        window.open_path(args.model)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

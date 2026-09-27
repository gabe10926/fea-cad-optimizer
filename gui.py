"""
GUI for the FEA Gradient Descent Optimizer.
"""

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)
from fem_profile import AnalysisProfile
from pipeline import OptimizerSettings, design_space, open_model, run_optimization
from progress_plots import ProgressPlots


class OptimizationWorker(QThread):
    """Runs the shared pipeline off the main thread so the GUI stays responsive."""
    progress = pyqtSignal(int, object, float, float, object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, session, profile, initial, steps, bounds, settings):
        super().__init__()
        self.session = session
        self.profile = profile
        self.initial = initial
        self.steps = steps
        self.bounds = bounds
        self.settings = settings

    def run(self):
        try:
            result = run_optimization(
                self.session, self.profile, self.initial, self.steps, self.bounds, self.settings,
                callback=lambda it, p, obj, gn, info: self.progress.emit(it, p, obj, gn, info),
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class OptimizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEA Gradient Descent Optimizer")
        self.setMinimumWidth(700)
        self.session = None
        self.worker = None
        self.analysis_profile = AnalysisProfile()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout()

        self.open_btn = QPushButton("Open Part (build123d .py or FreeCAD .FCStd)")
        self.open_btn.clicked.connect(self._open_model)
        root.addWidget(self.open_btn)
        self.model_label = QLabel("No model selected")
        root.addWidget(self.model_label)

        param_group = QGroupBox("Design Parameters (mm)")
        self.param_form = QFormLayout()
        self.param_inputs, self.step_inputs, self.min_inputs, self.max_inputs = {}, {}, {}, {}
        param_group.setLayout(self.param_form)
        root.addWidget(param_group)

        profile_group = QGroupBox("FEM Setup Profile (JSON)")
        profile_layout = QVBoxLayout()
        self.profile_editor = QPlainTextEdit(self.analysis_profile.to_json())
        self.profile_editor.setMinimumHeight(220)
        profile_layout.addWidget(self.profile_editor)
        profile_buttons = QHBoxLayout()
        for label, handler in (("Load Profile", self._load_profile), ("Save Profile", self._save_profile),
                               ("Check Faces", self._check_faces)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            profile_buttons.addWidget(button)
        profile_layout.addLayout(profile_buttons)
        profile_group.setLayout(profile_layout)
        root.addWidget(profile_group)

        opt_group = QGroupBox("Optimizer Settings")
        opt_form = QFormLayout()
        self.lr_input = QLineEdit("0.1")
        self.tol_input = QLineEdit("1e-4")
        self.max_iter_input = QLineEdit("30")
        self.workers_input = QLineEdit("4")
        self.method_input = QComboBox()
        self.method_input.addItems(["slsqp", "gradient"])
        opt_form.addRow("Method:", self.method_input)
        opt_form.addRow("Learning Rate (gradient):", self.lr_input)
        opt_form.addRow("Tolerance:", self.tol_input)
        opt_form.addRow("Max Iterations:", self.max_iter_input)
        opt_form.addRow("Parallel Workers:", self.workers_input)
        opt_group.setLayout(opt_form)
        root.addWidget(opt_group)

        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self._start)
        root.addWidget(self.start_btn)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        root.addWidget(self.log)

        self.plots = ProgressPlots()
        self.plots.setMinimumHeight(460)
        root.addWidget(self.plots)

        self.result_label = QLabel("")
        root.addWidget(self.result_label)
        self.setLayout(root)

    def _add_parameter(self, name, value, step, minimum, maximum):
        row = QHBoxLayout()
        edits = [QLineEdit(f"{number:.6g}") for number in (value, step, minimum, maximum)]
        for label, widget in zip(("Value:", "Step:", "Min:", "Max:"), edits):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.param_form.addRow(f"{name}:", row)
        for store, widget in zip((self.param_inputs, self.step_inputs, self.min_inputs, self.max_inputs), edits):
            store[name] = widget

    def _profile_from_editor(self) -> AnalysisProfile:
        profile = AnalysisProfile.from_json(self.profile_editor.toPlainText())
        errors = profile.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return profile

    def _workers(self) -> int:
        return max(1, int(self.workers_input.text()))

    def _open_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Part", "", "Parametric parts (*.py *.FCStd *.fcstd)"
        )
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            profile = self._profile_from_editor()
            if self.session is not None:
                self.session.close()
                self.session = None
            session = open_model(path, profile, workers=self._workers())
            errors = session.metadata.validate()
            if errors:
                session.close()
                raise ValueError("; ".join(errors))
            self.session = session
            self.model_label.setText(f"Model: {path}  ({session.metadata.backend})")
            while self.param_form.rowCount():
                self.param_form.removeRow(0)
            for store in (self.param_inputs, self.step_inputs, self.min_inputs, self.max_inputs):
                store.clear()
            initial, bounds, steps = design_space(session.metadata)
            for index, parameter in enumerate(session.metadata.parameters):
                self._add_parameter(parameter.alias, initial[index], steps[index], *bounds[index])
            self.log.append(f"Loaded {len(session.metadata.parameters)} design variables.")
            self._show_faces()
        except Exception as exc:
            self.log.append(f"ERROR loading model: {exc}")

    def _show_faces(self):
        metadata = self.session.metadata
        for name, faces in metadata.selections.items():
            self.log.append(f"  {name} -> {', '.join(faces)}")
        for warning in metadata.warnings:
            self.log.append(f"  WARNING: {warning}")

    def _check_faces(self):
        if self.session is None:
            self.log.append("Open a model first.")
            return
        try:
            profile = self._profile_from_editor()
            from model_intake import ModelMetadata
            self.session.metadata = ModelMetadata.from_dict(self.session.controller.inspect_model(profile))
            self.log.append("Faces (number, center mm, area mm2, bounding-box planes):")
            for face in self.session.metadata.faces:
                center = ", ".join(f"{value:.1f}" for value in face["center"])
                self.log.append(f"  {face['face']}: ({center})  {face['area']:.1f}  {' '.join(face['on'])}")
            self._show_faces()
        except Exception as exc:
            self.log.append(f"ERROR checking faces: {exc}")

    def _load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load FEM Profile", "", "JSON (*.json)")
        if path:
            self.load_profile_path(path)

    def load_profile_path(self, path):
        try:
            profile = AnalysisProfile.load(path)
            errors = profile.validate()
            if errors:
                raise ValueError("; ".join(errors))
            self.analysis_profile = profile
            self.profile_editor.setPlainText(profile.to_json())
            self.log.append(f"Loaded FEM profile: {path}")
        except Exception as exc:
            self.log.append(f"ERROR loading FEM profile: {exc}")

    def _save_profile(self):
        try:
            profile = self._profile_from_editor()
            path, _ = QFileDialog.getSaveFileName(self, "Save FEM Profile", "analysis_profile.json",
                                                  "JSON (*.json)")
            if path:
                profile.save(path)
                self.analysis_profile = profile
                self.log.append(f"Saved FEM profile: {path}")
        except Exception as exc:
            self.log.append(f"ERROR saving FEM profile: {exc}")

    def _start(self):
        if self.session is None:
            self.log.setText("ERROR: Open a parametric part first.")
            return
        self.log.clear()
        self.result_label.setText("")
        names = list(self.param_inputs)
        try:
            self.analysis_profile = self._profile_from_editor()
            initial = np.array([float(self.param_inputs[n].text()) for n in names])
            steps = np.array([float(self.step_inputs[n].text()) for n in names])
            bounds = np.array([
                [float(self.min_inputs[n].text()), float(self.max_inputs[n].text())] for n in names
            ])
            if self._workers() != getattr(self.session.controller, "workers", 1):
                self.log.append("Restarting model with the new worker count...")
                path = self.session.job.source_path
                self.session.close()
                self.session = open_model(path, self.analysis_profile, workers=self._workers())
            settings = OptimizerSettings(
                float(self.lr_input.text()), float(self.tol_input.text()), int(self.max_iter_input.text()),
                self.method_input.currentText(),
            )
        except (ValueError, TypeError) as exc:
            self.log.append(f"ERROR in settings: {exc}")
            return
        self.plots.reset(names, bounds)
        self.worker = OptimizationWorker(self.session, self.analysis_profile, initial, steps, bounds, settings)
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_finished)
        self.worker.failed.connect(self._on_error)
        self.start_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.worker.start()

    def _on_progress(self, iteration, params, objective, grad_norm, info):
        self.log.append(f"Iter {iteration:3d} | J: {objective:.6g} kg | |grad|: {grad_norm:.4g} | x: "
                        + ", ".join(f"{value:.3f}" for value in params))
        self.plots.add(iteration, params, objective, info)

    def _on_finished(self, result):
        names = ", ".join(f"{name}={value:.4f}" for name, value in result["optimized_parameters"].items())
        evaluation = result["evaluation"]
        self.result_label.setText(f"Optimized: {names}")
        self.log.append(f"\nDone. Final params: {names}")
        self.log.append(
            f"Mass: {evaluation['mass']:.6g} kg | Stress: {evaluation['stress'] / 1e6:.4g} MPa | "
            f"Displacement: {evaluation['displacement'] * 1e3:.4g} mm | Feasible: {evaluation['feasible']}"
        )
        self.log.append(f"Status: {result['optimizer_status']} | {result['solves']} solves in "
                        f"{result['wall_seconds']} s")
        self.log.append(f"Exported to {result['export_dir']}")
        self.start_btn.setEnabled(True)
        self.open_btn.setEnabled(True)

    def _on_error(self, msg):
        self.log.append(f"\nERROR: {msg}")
        self.start_btn.setEnabled(True)
        self.open_btn.setEnabled(True)

    def closeEvent(self, event):
        if self.session is not None:
            self.session.close()
        super().closeEvent(event)

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QTextEdit
)
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from optimizer import GDOptimizer
from simulation import objective_function


# =========================
# Worker Thread
# =========================
class OptimizationWorker(QThread):
    progress = pyqtSignal(int, object, float, float)
    finished = pyqtSignal(object)

    def __init__(self, initial, steps, lr, tol):
        super().__init__()
        self.initial = initial
        self.steps = steps
        self.lr = lr
        self.tol = tol

    def run(self):

        def callback(iteration, params, objective, grad_norm):
            self.progress.emit(iteration, params, objective, grad_norm)

        optimizer = GDOptimizer(
            initial_params=self.initial,
            step_sizes=self.steps,
            learning_rate=self.lr,
            tolerance=self.tol,
            max_iterations=50,
            callback=callback
        )

        result = optimizer.optimize(objective_function)
        self.finished.emit(result)


# =========================
# GUI
# =========================
class OptimizerGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FEA Gradient Descent Optimizer")
        self.setMinimumWidth(600)

        layout = QVBoxLayout()
        form = QFormLayout()

        # Inputs
        self.param1 = QLineEdit("1.0")
        self.param2 = QLineEdit("1.0")
        self.step1 = QLineEdit("0.01")
        self.step2 = QLineEdit("0.01")
        self.lr = QLineEdit("0.1")
        self.tol = QLineEdit("1e-4")

        form.addRow("Parameter 1:", self.param1)
        form.addRow("Parameter 2:", self.param2)
        form.addRow("Step Size 1:", self.step1)
        form.addRow("Step Size 2:", self.step2)
        form.addRow("Learning Rate:", self.lr)
        form.addRow("Tolerance:", self.tol)

        layout.addLayout(form)

        # Start Button
        self.start_button = QPushButton("Start Optimization")
        self.start_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.start_button)

        # Log Box
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        # Live Plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Convergence Plot")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Objective Value")

        self.iterations = []
        self.objectives = []

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def start_optimization(self):

        self.iterations = []
        self.objectives = []
        self.ax.clear()
        self.ax.set_title("Convergence Plot")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Objective Value")
        self.canvas.draw()

        initial = np.array([
            float(self.param1.text()),
            float(self.param2.text())
        ])

        steps = np.array([
            float(self.step1.text()),
            float(self.step2.text())
        ])

        self.worker = OptimizationWorker(
            initial,
            steps,
            float(self.lr.text()),
            float(self.tol.text())
        )

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.optimization_finished)

        self.start_button.setEnabled(False)
        self.log.clear()
        self.worker.start()

    def update_progress(self, iteration, params, objective, grad_norm):

        self.log.append(
            f"Iter {iteration} | Obj: {objective:.6f} | Grad Norm: {grad_norm:.6f}"
        )

        self.iterations.append(iteration)
        self.objectives.append(objective)

        self.ax.clear()
        self.ax.plot(self.iterations, self.objectives)
        self.ax.set_title("Convergence Plot")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Objective Value")
        self.canvas.draw()

    def optimization_finished(self, result):
        self.result_label.setText(f"Optimized Parameters: {result}")
        self.start_button.setEnabled(True)
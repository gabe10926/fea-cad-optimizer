"""Live optimization progress charts for the GUI.

Four small multiples share the iteration axis (never two y-scales on one plot):
mass, each limit as a percentage of its allowed value, each design variable as
a percentage of its bound range, and the cumulative number of solves. Hovering
any panel marks that iteration in all four and prints its values underneath.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

# Validated categorical order (light surface); assigned by entity, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"
LIMIT_LABELS = {"stress": "Stress", "displacement": "Deflection", "mass": "Mass"}


class ProgressPlots(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 5.2), facecolor=SURFACE, layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.readout = QLabel("Hover a chart to see an iteration's values.")
        self.readout.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.readout)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.reset([], np.zeros((0, 2)))

    # ------------------------------------------------------------------ data

    def reset(self, names, bounds):
        self.names = list(names)
        self.bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
        self.rows: list[dict] = []
        self._draw()

    def add(self, iteration, params, objective, info):
        evaluation = (info or {}).get("evaluation") or {}
        span = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        self.rows.append({
            "iteration": int(iteration),
            "params": np.asarray(params, dtype=float),
            "percent_of_range": (np.asarray(params, dtype=float) - self.bounds[:, 0]) / span * 100.0,
            "mass": evaluation.get("mass"),
            "objective": objective,
            "feasible": evaluation.get("feasible"),
            "limits": {name: (ratio + 1.0) * 100.0 for name, ratio in evaluation.get("constraint_ratios", {}).items()},
            "stress": evaluation.get("stress"),
            "displacement": evaluation.get("displacement"),
            "solves": (info or {}).get("solves"),
        })
        self._draw()

    # --------------------------------------------------------------- drawing

    @staticmethod
    def _style(ax, title, ylabel):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=TEXT_PRIMARY)
        ax.set_ylabel(ylabel, fontsize=8.5, color=TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    @staticmethod
    def _end_labels(ax, x, items, gap_fraction=0.09):
        """Dot at each series end plus its name, nudged apart so labels never overlap."""
        low, high = ax.get_ylim()
        gap = (high - low) * gap_fraction
        order = sorted(range(len(items)), key=lambda index: items[index][0])
        placed = {}
        previous = None
        for index in order:
            y = items[index][0]
            placed[index] = y if previous is None else max(y, previous + gap)
            previous = placed[index]
        for index, (y, text, color) in enumerate(items):
            ax.plot([x], [y], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=4)
            ax.annotate(text, (x, placed[index]), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_PRIMARY, annotation_clip=False,
                        bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"})

    def _draw(self):
        self.figure.clear()
        grid = self.figure.subplots(2, 2, sharex=True)
        self.axes = grid.ravel()
        mass_ax, limit_ax, design_ax, solve_ax = self.axes
        self._style(mass_ax, "Mass", "kg")
        self._style(limit_ax, "Limits used (incl. safety factor)", "% of limit")
        self._style(design_ax, "Design variables", "% of bound range")
        self._style(solve_ax, "Solves used", "cumulative")
        for ax in (design_ax, solve_ax):
            ax.set_xlabel("Iteration", fontsize=8.5, color=TEXT_SECONDARY)
        self.cursors = [ax.axvline(0, color=REFERENCE, lw=1, alpha=0, zorder=1) for ax in self.axes]

        if not self.rows:
            for ax in self.axes:
                ax.text(0.5, 0.5, "waiting for first iteration", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color=TEXT_SECONDARY)
            self.canvas.draw_idle()
            return

        its = [row["iteration"] for row in self.rows]
        last = its[-1]

        # Mass: filled markers are feasible designs, hollow ones break a limit.
        masses = [row["mass"] if row["mass"] is not None else np.nan for row in self.rows]
        mass_ax.plot(its, masses, color=SERIES[0], lw=1.5, zorder=2)
        for it, mass, row in zip(its, masses, self.rows):
            feasible = row["feasible"] is True
            mass_ax.plot([it], [mass], "o", ms=6, zorder=3, color=SERIES[0],
                         mfc=SERIES[0] if feasible else SURFACE, mec=SERIES[0], mew=1.5)
        if np.isfinite(masses[-1]):
            start = masses[0]
            change = f" ({(masses[-1] / start - 1) * 100:+.0f}%)" if np.isfinite(start) and start else ""
            mass_ax.annotate(f"{masses[-1]:.4g} kg{change}", (last, masses[-1]), xytext=(0, 8),
                             textcoords="offset points", ha="right", fontsize=8, color=TEXT_PRIMARY)
        mass_ax.text(0.99, 0.97, "\u25cb hollow = breaks a limit", transform=mass_ax.transAxes,
                     ha="right", va="top", fontsize=7.5, color=TEXT_SECONDARY)

        # Limits as % of allowed, with the 100% line every design must stay under.
        limit_names = [name for name in LIMIT_LABELS if any(name in row["limits"] for row in self.rows)]
        limit_ax.axhline(100, color=REFERENCE, lw=1, ls=(0, (4, 3)), zorder=1)
        limit_ax.annotate("limit", (0, 100), xycoords=("axes fraction", "data"), xytext=(2, 3),
                          textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY)
        ends = []
        for slot, name in enumerate(limit_names):
            values = [row["limits"].get(name, np.nan) for row in self.rows]
            limit_ax.plot(its, values, color=SERIES[slot], lw=1.5, label=LIMIT_LABELS[name], zorder=2)
            ends.append((values[-1], f"{LIMIT_LABELS[name]} {values[-1]:.0f}%", SERIES[slot]))
        top = max([100.0] + [value for row in self.rows for value in row["limits"].values()])
        limit_ax.set_ylim(0, top * 1.12)
        self._end_labels(limit_ax, last, ends)
        if limit_names:
            limit_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(limit_names),
                            fontsize=7.5, frameon=False, labelcolor=TEXT_PRIMARY)

        # Design variables on one common scale: 0% = lower bound, 100% = upper bound.
        design_ax.set_ylim(-5, 105)
        percents = np.array([row["percent_of_range"] for row in self.rows])
        for slot, name in enumerate(self.names[:len(SERIES)]):
            design_ax.plot(its, percents[:, slot], color=SERIES[slot], lw=1.5, label=name, zorder=2)
        if len(self.names) <= 4:
            self._end_labels(design_ax, last, [
                (percents[-1, slot], name, SERIES[slot]) for slot, name in enumerate(self.names)
            ])
        design_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=min(4, len(self.names)),
                         fontsize=7.5, frameon=False, labelcolor=TEXT_PRIMARY)

        # Solves: the cost of each iteration.
        solves = [row["solves"] if row["solves"] is not None else np.nan for row in self.rows]
        solve_ax.plot(its, solves, color=SERIES[0], lw=1.5, marker="o", ms=5, mec=SURFACE, mew=1.5)
        if np.isfinite(solves[-1]):
            solve_ax.annotate(f"{solves[-1]:.0f}", (last, solves[-1]), xytext=(0, 8),
                              textcoords="offset points", ha="right", fontsize=8, color=TEXT_PRIMARY)

        # Room on the right for end labels.
        for ax in self.axes:
            ax.set_xlim(min(its) - 0.3, max(its) + max(0.8, 0.35 * max(1, max(its) - min(its))))
        self.canvas.draw_idle()

    # ----------------------------------------------------------------- hover

    def _on_hover(self, event):
        if not self.rows or event.inaxes not in getattr(self, "axes", []) or event.xdata is None:
            return
        row = min(self.rows, key=lambda item: abs(item["iteration"] - event.xdata))
        for cursor in self.cursors:
            cursor.set_xdata([row["iteration"], row["iteration"]])
            cursor.set_alpha(0.8)
        parts = [f"<b>Iteration {row['iteration']}</b>"]
        if row["mass"] is not None:
            parts.append(f"mass {row['mass']:.5g} kg ({'feasible' if row['feasible'] else 'over a limit'})")
        if row["stress"] is not None:
            parts.append(f"stress {row['stress'] / 1e6:.4g} MPa")
        if row["displacement"] is not None:
            parts.append(f"deflection {row['displacement'] * 1e3:.4g} mm")
        parts += [f"{LIMIT_LABELS[name]} {value:.1f}% of limit" for name, value in row["limits"].items()]
        parts += [f"{name} = {value:.4g} mm" for name, value in zip(self.names, row["params"])]
        if row["solves"] is not None:
            parts.append(f"{row['solves']} solves so far")
        self.readout.setText(" · ".join(parts))
        self.canvas.draw_idle()

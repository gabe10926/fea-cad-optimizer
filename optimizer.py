import numpy as np

class GDOptimizer:
    
    def __init__(self,
                 initial_params,
                 step_sizes,
                 learning_rate=0.01,
                 tolerance=1e-4,
                 max_iterations=50,
                 callback=None):

        self.params = np.array(initial_params, dtype=float)
        self.step_sizes = np.array(step_sizes, dtype=float)
        self.alpha = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.callback = callback  # used to update GUI

    def compute_gradient(self, objective_function):
        grad = np.zeros_like(self.params)

        for i in range(len(self.params)):
            step = self.step_sizes[i]

            params_plus = self.params.copy()
            params_minus = self.params.copy()

            params_plus[i] += step
            params_minus[i] -= step

            f_plus = objective_function(params_plus)
            f_minus = objective_function(params_minus)

            grad[i] = (f_plus - f_minus) / (2 * step)

        return grad

    def optimize(self, objective_function):

        for iteration in range(self.max_iterations):

            f_current = objective_function(self.params)
            gradient = self.compute_gradient(objective_function)

            grad_norm = np.linalg.norm(gradient)

            if self.callback:
                self.callback(iteration, self.params, f_current, grad_norm)

            if grad_norm < self.tolerance:
                return self.params

            self.params -= self.alpha * gradient

        return self.params
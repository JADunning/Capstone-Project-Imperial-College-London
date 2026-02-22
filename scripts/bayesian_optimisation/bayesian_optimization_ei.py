"""
Bayesian Optimization with Gaussian Process and Expected Improvement (EI).

This module provides a BO implementation using:
- Gaussian Process (GP) as the surrogate model
- Expected Improvement (EI) as the acquisition function

Compared with mean-only UCB, EI directly optimizes expected improvement over
the best observed value.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


class BayesianOptimizationEI:
    """
    Bayesian Optimization using GP + EI.

    Parameters
    ----------
    bounds : array-like, shape (n_dims, 2)
        Bounds for each dimension in the format [[min, max], ...].
    n_initial : int, default=5
        Number of random initial points to sample in `optimize`.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, bounds, n_initial=5, random_state=42):
        self.bounds = np.array(bounds, dtype=float)
        self.n_dims = len(bounds)
        self.n_initial = n_initial
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.X_observed = np.empty((0, self.n_dims), dtype=float)
        self.y_observed = np.array([], dtype=float)

        # Safer kernel bounds: avoid length_scale → 0 (nearly singular K) and
        # constant_value → 1000 (numerical blow-up). normalize_y stabilizes fit.
        kernel = C(1.0, (1e-2, 100.0)) * RBF(1.0, (0.1, 10.0))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state,
            alpha=1e-6,
            normalize_y=True,
        )

    def fit(self, X, y):
        """Fit the GP surrogate model to observed data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X_observed = X
        self.y_observed = y
        self.gp.fit(X, y)

    def predict(self, X, return_std=True):
        """Predict mean and standard deviation using the fitted GP."""
        X = np.asarray(X, dtype=float)
        mean, std = self.gp.predict(X, return_std=True)
        if return_std:
            return mean, std
        return mean

    def acquisition_ei(self, X, xi=0.01):
        """
        Expected Improvement (EI) acquisition function.

        EI(x) = E[max(f(x) - f_best - xi, 0)]
        """
        if self.y_observed.size == 0:
            raise RuntimeError("Call fit before acquisition_ei.")

        mean, std = self.predict(X)
        best = np.max(self.y_observed)
        std = np.maximum(std, 1e-12)
        z = (mean - best - xi) / std
        ei = (mean - best - xi) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def _is_duplicate(self, x, min_distance):
        """Return True if x is too close to an observed point."""
        if self.X_observed.size == 0:
            return False
        distances = np.linalg.norm(self.X_observed - x.reshape(1, -1), axis=1)
        return np.min(distances) <= min_distance

    def suggest_next_point(
        self,
        xi=0.01,
        n_restarts=25,
        n_candidates=5000,
        min_distance=1e-6,
    ):
        """
        Suggest next point by maximizing EI with robust search.

        Strategy:
        1. Global random candidate scan for a strong EI seed.
        2. Multi-start L-BFGS-B local refinement on EI.
        3. Duplicate rejection against observed points.
        """
        if self.y_observed.size == 0:
            raise RuntimeError("Call fit before suggest_next_point.")

        def negative_ei(x):
            return -self.acquisition_ei(x.reshape(1, -1), xi=xi)[0]

        candidates = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(max(1, n_candidates), self.n_dims),
        )
        ei_vals = self.acquisition_ei(candidates, xi=xi)

        order = np.argsort(-ei_vals)
        best_point = None
        best_value = -np.inf

        for idx in order:
            x = candidates[idx]
            val = ei_vals[idx]
            if self._is_duplicate(x, min_distance):
                continue
            best_point = x
            best_value = val
            break

        seed_count = min(n_restarts, len(order))
        for rank in range(seed_count):
            x0 = candidates[order[rank]]
            result = minimize(
                negative_ei,
                x0=x0,
                bounds=self.bounds,
                method="L-BFGS-B",
            )
            if not result.success:
                continue

            x = np.asarray(result.x, dtype=float)
            if self._is_duplicate(x, min_distance):
                continue

            val = -result.fun
            if val > best_value:
                best_value = val
                best_point = x

        if best_point is not None:
            return best_point

        # Fallback: sample until we find a non-duplicate point.
        for _ in range(1000):
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.n_dims)
            if not self._is_duplicate(x, min_distance):
                return x

        # Final fallback: best observed point + small bounded jitter.
        x_best = self.X_observed[np.argmax(self.y_observed)].copy()
        span = self.bounds[:, 1] - self.bounds[:, 0]
        jitter = self.rng.normal(0.0, 1e-3, size=self.n_dims) * span
        x = np.clip(x_best + jitter, self.bounds[:, 0], self.bounds[:, 1])
        return x

    def optimize(self, objective_func, n_iterations=10, xi=0.01, verbose=True):
        """Run BO loop using EI acquisition."""
        X_init = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.n_initial, self.n_dims),
        )
        y_init = np.array([objective_func(x) for x in X_init], dtype=float)

        X_all = X_init.copy()
        y_all = y_init.copy()

        if verbose:
            print(f"Initial random sampling: {self.n_initial} points")
            print(f"Best initial value: {y_all.max():.6f}")

        for i in range(n_iterations):
            self.fit(X_all, y_all)
            next_x = self.suggest_next_point(xi=xi)
            next_y = objective_func(next_x)

            X_all = np.vstack([X_all, next_x.reshape(1, -1)])
            y_all = np.append(y_all, float(next_y))

            if verbose:
                best_idx = np.argmax(y_all)
                best_x = X_all[best_idx]
                best_y = y_all[best_idx]
                print(
                    f"Iteration {i+1}/{n_iterations}: "
                    f"Next point: {next_x}, "
                    f"Value: {next_y:.6f}, "
                    f"Best so far: {best_y:.6f}"
                )

        best_idx = np.argmax(y_all)
        best_x = X_all[best_idx]
        best_y = y_all[best_idx]
        return X_all, y_all, best_x, best_y

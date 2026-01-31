"""
Noise-aware Bayesian Optimization (re-purposed from exploitative variant).

This module is adapted from bayesian_optimization_exploitative.py for use with
objectives that have significant output noise (e.g. stochastic simulations,
noisy evaluations, or algorithms with random seeds). Below is what we changed
and why it is better for noisy algorithms.

Changes from the exploitative script
------------------------------------
1. GP kernel: C * RBF + WhiteKernel (instead of C * RBF only)
   - The original GP assumes observations are essentially noiseless, so it
     overfits: it tries to pass through every point, giving near-zero
     predictive variance near data. With noisy objectives this is wrong and
     makes the acquisition function overconfident.
   - Adding WhiteKernel lets the surrogate learn the observation noise level.
     The GP then smooths through noise instead of overfitting, and predictive
     uncertainty σ(x) stays meaningful. That makes UCB (and other acquisition
     functions) behave correctly.

2. Default beta = 0.5 (instead of 0.0)
   - The exploitative script uses beta=0 (pure exploitation of the mean). For
     noisy objectives, the mean at any point can be misleading due to noise;
     pure exploitation can lock onto spuriously good values.
   - Using a moderate beta (e.g. 0.2–1.0) adds exploration via σ(x), reducing
     over-commitment to noisy peaks and improving robustness. We default to
     0.5 as a balanced choice for noisy settings.

3. Local restarts near the best observed point (in addition to global restarts)
   - The original only uses global random restarts to maximize UCB. We add
     n_local restarts initialized near the current best observed point (with
     small Gaussian perturbations scaled by the domain).
   - This balances exploitation (refining around the best so far) with
     exploration (global restarts), which is especially useful when the
     objective is noisy and you still want to refine promising regions.

4. GP settings: normalize_y=True, alpha=1e-8, n_restarts_optimizer=20
   - normalize_y improves numerical stability when observation scales are
     large or vary a lot (common with noisy outputs).
   - alpha is a small jitter for numerical stability; we use 1e-8.
   - More kernel hyperparameter restarts (20 vs 10) give more robust fits
     when including the WhiteKernel (extra hyperparameters to optimize).

Summary
-------
For noisy objectives, use this script (WhiteKernel + moderate beta + local
restarts) instead of the purely exploitative variant (no WhiteKernel, beta=0,
global restarts only). The exploitative variant is better suited to
deterministic or very low-noise objectives where pure mean exploitation is
desired.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize

from .bayesian_optimization_exploitative import SimpleBayesianOptimization


class BayesianOptimizationForNoise(SimpleBayesianOptimization):
    """
    Noise-aware Bayesian Optimization for objectives with significant output noise.

    - Noise-aware GP: C * RBF + WhiteKernel (don't assume near-zero noise).
    - Local restarts near best observed point + global random restarts.
    - Default beta=0.5 (use ~0.2–1.0 for noisy objectives, not 0).

    Parameters
    ----------
    bounds : array-like, shape (n_dims, 2)
        Bounds for each dimension.
    n_initial : int, default=5
        Number of random initial points.
    random_state : int, default=42
        Random seed.
    """

    def __init__(self, bounds, n_initial=5, random_state=42):
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_initial = n_initial
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.X_observed = []
        self.y_observed = []

        # Noise-aware kernel: signal (C * RBF) + WhiteKernel for observation noise
        kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(self.n_dims), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e-1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=20,
            random_state=random_state,
            alpha=1e-8,  # tiny jitter for numerical stability
        )

    def suggest_next_point(self, beta=0.5, n_restarts=25, n_local=15, local_scale=0.05):
        """
        Suggest next point by maximizing UCB, with local restarts near best observed
        and global random restarts. Default beta=0.5 for noisy objectives.
        """
        def negative_ucb(x):
            return -self.acquisition_ucb(x.reshape(1, -1), beta=beta)[0]

        best_point = None
        best_value = -np.inf

        restart_points = []

        # Local restarts near current best observed point (exploitation)
        if len(self.X_observed) > 0:
            x_best = self.X_observed[np.argmax(self.y_observed)]
            span = self.bounds[:, 1] - self.bounds[:, 0]
            for _ in range(n_local):
                x0 = x_best + self.rng.normal(0, local_scale, size=self.n_dims) * span
                x0 = np.clip(x0, self.bounds[:, 0], self.bounds[:, 1])
                restart_points.append(x0)

        # Global random restarts (safety)
        for _ in range(n_restarts):
            x0 = self.rng.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=self.n_dims
            )
            restart_points.append(x0)

        for x0 in restart_points:
            result = minimize(
                negative_ucb, x0=x0, bounds=self.bounds, method="L-BFGS-B"
            )
            if result.success and -result.fun > best_value:
                best_value = -result.fun
                best_point = result.x

        if best_point is None:
            best_point = self.rng.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=self.n_dims
            )

        return best_point

    def optimize(self, objective_func, n_iterations=10, beta=0.5, verbose=True):
        """Run BO with default beta=0.5 for noisy objectives."""
        return super().optimize(
            objective_func, n_iterations=n_iterations, beta=beta, verbose=verbose
        )

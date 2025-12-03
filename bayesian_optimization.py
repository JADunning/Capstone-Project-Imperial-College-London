"""
Simple Bayesian Optimization with Gaussian Process and Upper Confidence Bound (UCB)

This module provides a simple implementation of Bayesian Optimization using:
- Gaussian Process (GP) as the surrogate model
- Upper Confidence Bound (UCB) as the acquisition function
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize


class SimpleBayesianOptimization:
    """
    Simple Bayesian Optimization using GP + UCB
    
    Parameters:
    -----------
    bounds : array-like, shape (n_dims, 2)
        Bounds for each dimension in the format [[min, max], ...]
    n_initial : int, default=5
        Number of random initial points to sample
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, bounds, n_initial=5, random_state=42):
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_initial = n_initial
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        
        # Initialize GP with RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state,
            alpha=1e-6
        )
    
    def fit(self, X, y):
        """
        Fit the GP surrogate model to observed data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_dims)
            Input points
        y : array-like, shape (n_samples,)
            Output values
        """
        X = np.array(X)
        y = np.array(y).ravel()
        
        self.X_observed = X
        self.y_observed = y
        self.gp.fit(X, y)
    
    def predict(self, X, return_std=True):
        """
        Predict mean and standard deviation using the GP
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_dims)
            Input points to predict
        
        Returns:
        --------
        mean : array, shape (n_samples,)
            Predicted mean
        std : array, shape (n_samples,)
            Predicted standard deviation (if return_std=True)
        """
        X = np.array(X)
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std
    
    def acquisition_ucb(self, X, beta=2.0):
        """
        Upper Confidence Bound (UCB) acquisition function
        
        UCB(x) = μ(x) + β * σ(x)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_dims)
            Input points to evaluate
        beta : float, default=2.0
            Exploration-exploitation trade-off parameter
        
        Returns:
        --------
        ucb : array, shape (n_samples,)
            UCB values (higher is better)
        """
        mean, std = self.predict(X)
        ucb = mean + beta * std
        return ucb
    
    def suggest_next_point(self, beta=2.0, n_restarts=25):
        """
        Suggest the next point to evaluate by maximizing UCB
        
        Parameters:
        -----------
        beta : float, default=2.0
            Exploration-exploitation trade-off parameter
        n_restarts : int, default=25
            Number of random restarts for optimization
        
        Returns:
        --------
        next_point : array, shape (n_dims,)
            Next point to evaluate
        """
        def negative_ucb(x):
            """Negative UCB for minimization"""
            return -self.acquisition_ucb(x.reshape(1, -1), beta=beta)[0]
        
        best_point = None
        best_value = -np.inf
        
        # Try multiple random starting points
        for _ in range(n_restarts):
            # Random starting point within bounds
            x0 = self.rng.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=self.n_dims
            )
            
            # Optimize
            result = minimize(
                negative_ucb,
                x0=x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_value:
                best_value = -result.fun
                best_point = result.x
        
        # Fallback to random point if optimization fails
        if best_point is None:
            best_point = self.rng.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=self.n_dims
            )
        
        return best_point
    
    def optimize(self, objective_func, n_iterations=10, beta=2.0, verbose=True):
        """
        Run Bayesian Optimization
        
        Parameters:
        -----------
        objective_func : callable
            Function to optimize (takes X and returns y)
        n_iterations : int, default=10
            Number of BO iterations
        beta : float, default=2.0
            Exploration-exploitation trade-off parameter
        verbose : bool, default=True
            Whether to print progress
        
        Returns:
        --------
        X_all : array, shape (n_total, n_dims)
            All evaluated points
        y_all : array, shape (n_total,)
            All function values
        best_x : array, shape (n_dims,)
            Best point found
        best_y : float
            Best function value found
        """
        # Initial random sampling
        X_init = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.n_initial, self.n_dims)
        )
        
        y_init = np.array([objective_func(x) for x in X_init])
        
        X_all = X_init.copy()
        y_all = y_init.copy()
        
        if verbose:
            print(f"Initial random sampling: {self.n_initial} points")
            print(f"Best initial value: {y_all.max():.6f}")
        
        # BO iterations
        for i in range(n_iterations):
            # Fit GP to current observations
            self.fit(X_all, y_all)
            
            # Suggest next point
            next_x = self.suggest_next_point(beta=beta)
            
            # Evaluate objective
            next_y = objective_func(next_x)
            
            # Add to observations
            X_all = np.vstack([X_all, next_x.reshape(1, -1)])
            y_all = np.append(y_all, next_y)
            
            if verbose:
                best_idx = np.argmax(y_all)
                best_x = X_all[best_idx]
                best_y = y_all[best_idx]
                print(f"Iteration {i+1}/{n_iterations}: "
                      f"Next point: {next_x}, "
                      f"Value: {next_y:.6f}, "
                      f"Best so far: {best_y:.6f}")
        
        # Find best point
        best_idx = np.argmax(y_all)
        best_x = X_all[best_idx]
        best_y = y_all[best_idx]
        
        return X_all, y_all, best_x, best_y


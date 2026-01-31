"""
Non-linear Logistic Regression helpers for the Capstone Project

This module provides:
- Binary label creation from continuous outputs (same as linear version)
- A non-linear logistic regression wrapper with polynomial features
- Creates curved decision boundaries instead of linear ones
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import minimize, differential_evolution


def make_binary_labels(y, threshold=None, threshold_percentile=75):
    """
    Convert continuous outputs into binary classes using a threshold.
    
    (Same function as in linear version - included for convenience)

    Parameters:
    -----------
    y : array-like, shape (n_samples,)
        Continuous outputs
    threshold : float, optional
        Explicit threshold for class 1
    threshold_percentile : float, default=75
        Percentile to use when threshold is None

    Returns:
    --------
    y_class : array, shape (n_samples,)
        Binary labels (0/1)
    threshold_used : float
        Threshold applied to create labels
    """
    y = np.array(y).ravel()
    if threshold is None:
        threshold_used = float(np.percentile(y, threshold_percentile))
    else:
        threshold_used = float(threshold)
    y_class = (y >= threshold_used).astype(int)
    return y_class, threshold_used


class NonLinearLogisticRegression:
    """
    Non-linear logistic regression wrapper with polynomial features.
    
    Creates curved decision boundaries by using polynomial feature transformations.
    For 2D input with degree=2, creates features: [1, x1, x2, x1^2, x1*x2, x2^2]

    Parameters:
    -----------
    degree : int, default=2
        Degree of polynomial features (2 = quadratic, 3 = cubic, etc.)
    penalty : str or None, default='l2'
        Regularization penalty ('l1', 'l2', or None)
    C : float, default=1.0
        Inverse regularization strength (smaller = stronger regularization)
    random_state : int, default=42
        Random seed for reproducibility
    max_iter : int, default=1000
        Max solver iterations
    solver : str, default='lbfgs'
        Solver to use for LogisticRegression
    include_bias : bool, default=True
        Whether to include bias term in polynomial features
    """
    
    def __init__(
        self, 
        degree=2, 
        penalty='l2', 
        C=1.0, 
        random_state=42, 
        max_iter=1000, 
        solver='lbfgs',
        include_bias=True
    ):
        self.degree = degree
        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        self.include_bias = include_bias
        
        # Feature transformers
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            include_bias=include_bias,
            interaction_only=False  # Include x1^2, x2^2, etc.
        )
        
        # Logistic regression model
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            random_state=random_state,
            max_iter=max_iter,
            solver=solver,
        )
    
    def fit_scaler(self, X):
        """
        Fit the scaler and polynomial transformer, return transformed features.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_poly_features)
            Transformed features (polynomial + scaled)
        """
        # First scale the original features
        X_scaled = self.scaler.fit_transform(X)
        # Then create polynomial features from scaled data
        X_poly = self.poly_features.fit_transform(X_scaled)
        return X_poly
    
    def transform(self, X):
        """
        Transform features using fitted scaler and polynomial transformer.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_poly_features)
            Transformed features (polynomial + scaled)
        """
        # Scale first, then polynomial transform
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly_features.transform(X_scaled)
        return X_poly
    
    def fit(self, X, y):
        """
        Fit the non-linear logistic regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Binary class labels (0/1)
        """
        X_transformed = self.fit_scaler(X)
        self.model.fit(X_transformed, y)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels (0/1)
        """
        X_transformed = self.transform(X)
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        probabilities : array, shape (n_samples, 2)
            Probabilities for each class [P(Class 0), P(Class 1)]
        """
        X_transformed = self.transform(X)
        return self.model.predict_proba(X_transformed)
    
    def confusion_matrix(self, y_true, y_pred):
        """Compute the confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def accuracy(self, y_true, y_pred):
        """Compute accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    def find_next_point_near_boundary(
        self,
        bounds,
        n_candidates=1000,
        method="grid_search",
        random_state=None,
    ):
        """
        Find the next point closest to the decision boundary (probability ≈ 0.5).
        
        This is useful for active learning - points near the boundary are most
        uncertain and provide the most information.
        
        Parameters:
        -----------
        bounds : array-like, shape (n_features, 2)
            Bounds for each feature: [[min1, max1], [min2, max2], ...]
        n_candidates : int, default=1000
            Number of random candidates to evaluate (for random/grid search)
        method : str, default='grid_search'
            Method to use: 'grid_search', 'random', or 'optimize'
        random_state : int or None, default=None
            Random seed for reproducibility
            
        Returns:
        --------
        next_point : array, shape (n_features,)
            Next point to evaluate (closest to decision boundary)
        uncertainty : float
            Distance from decision boundary (|prob - 0.5|)
        probability : float
            Predicted probability at this point
        """
        bounds = np.array(bounds)
        n_features = bounds.shape[0]
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if method == "grid_search":
            # Generate grid of candidates
            n_per_dim = int(np.ceil(n_candidates ** (1.0 / n_features)))
            grids = [np.linspace(bounds[i, 0], bounds[i, 1], n_per_dim) 
                    for i in range(n_features)]
            mesh = np.meshgrid(*grids)
            candidates = np.column_stack([g.ravel() for g in mesh])
            
        elif method == "random":
            # Generate random candidates
            candidates = np.random.uniform(
                low=bounds[:, 0],
                high=bounds[:, 1],
                size=(n_candidates, n_features)
            )
            
        elif method == "optimize":
            # Use optimization to find point closest to decision boundary
            def objective(x):
                x = x.reshape(1, -1)
                prob = self.predict_proba(x)[0, 1]  # Probability of class 1
                # Minimize distance from 0.5 (decision boundary)
                return abs(prob - 0.5)
            
            # Use differential evolution for global optimization
            result = differential_evolution(
                objective,
                bounds=list(bounds),
                seed=random_state,
                maxiter=100,
                popsize=15,
                atol=1e-6,
            )
            
            next_point = result.x
            prob = self.predict_proba(next_point.reshape(1, -1))[0, 1]
            uncertainty = abs(prob - 0.5)
            
            return next_point, uncertainty, prob
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'grid_search', 'random', or 'optimize'")
        
        # Evaluate all candidates
        probabilities = self.predict_proba(candidates)[:, 1]
        
        # Find point closest to decision boundary (prob = 0.5)
        uncertainties = np.abs(probabilities - 0.5)
        best_idx = np.argmin(uncertainties)
        
        next_point = candidates[best_idx]
        uncertainty = uncertainties[best_idx]
        probability = probabilities[best_idx]
        
        return next_point, uncertainty, probability
    
    def get_uncertainty(self, X):
        """
        Compute uncertainty (distance from decision boundary) for given points.
        
        Uncertainty is defined as |P(y=1|x) - 0.5|, where lower values
        indicate higher uncertainty (points closer to decision boundary).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input points
            
        Returns:
        --------
        uncertainties : array, shape (n_samples,)
            Uncertainty scores (lower = more uncertain)
        probabilities : array, shape (n_samples,)
            Predicted probabilities of class 1
        """
        probabilities = self.predict_proba(X)[:, 1]
        uncertainties = np.abs(probabilities - 0.5)
        return uncertainties, probabilities
    
    def get_polynomial_feature_names(self, feature_names=None):
        """
        Get names of polynomial features (useful for interpretation).
        
        Parameters:
        -----------
        feature_names : list of str, optional
            Names of original features (e.g., ['x1', 'x2'])
            If None, uses ['x0', 'x1', ...]
            
        Returns:
        --------
        poly_feature_names : list of str
            Names of polynomial features
        """
        if feature_names is None:
            n_original = self.poly_features.n_features_in_
            feature_names = [f'x{i}' for i in range(n_original)]
        
        return self.poly_features.get_feature_names_out(feature_names)

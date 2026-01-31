"""
Logistic regression: linear and nonlinear (polynomial) classifiers.
"""

from scripts.logistic_regression.logistic_regression import (
    SimpleLogisticRegression,
    make_binary_labels as make_binary_labels_linear,
)
from scripts.logistic_regression.nonlinear_logistic_regression import (
    NonLinearLogisticRegression,
    make_binary_labels,
)

__all__ = [
    "SimpleLogisticRegression",
    "NonLinearLogisticRegression",
    "make_binary_labels",
    "make_binary_labels_linear",
]

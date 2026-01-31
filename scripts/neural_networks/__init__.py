"""
Neural networks: PyTorch MLP regressor and next-point suggestion.
"""

from scripts.neural_networks.pytorch_simple_nn_next_point import (
    SimpleRegressor,
    train_model,
    suggest_next_point,
    suggest_next_point_gradient,
)

__all__ = [
    "SimpleRegressor",
    "train_model",
    "suggest_next_point",
    "suggest_next_point_gradient",
]

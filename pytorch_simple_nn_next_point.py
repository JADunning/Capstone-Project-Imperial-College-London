"""
Minimal PyTorch MLP regressor + next-point suggestion.

Notebook usage:
    model, y_mean, y_std = train_model(X, y)
    x_next, y_pred = suggest_next_point(model, X.shape[1], y_mean, y_std)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(
    x_np: np.ndarray,
    y_np: np.ndarray,
    hidden: int = 64,
    epochs: int = 1000,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[SimpleRegressor, torch.Tensor, torch.Tensor]:
    """
    Train a simple MLP regressor.

    Returns the model plus (y_mean, y_std) for de-normalizing predictions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    y_mean = y.mean()
    y_std = y.std()
    if y_std.item() == 0.0:
        y_std = torch.tensor(1.0)
    y_norm = (y - y_mean) / y_std

    model = SimpleRegressor(x.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, y_mean, y_std


def suggest_next_point(
    model: SimpleRegressor,
    in_dim: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    samples: int = 50000,
    seed: int = 123,
    batch_size: int = 2048,
) -> tuple[np.ndarray, float]:
    """
    Sample random candidate points in [0, 1] and return the best prediction.
    """
    rng = np.random.default_rng(seed)
    best_x = None
    best_pred = -np.inf

    model.eval()
    remaining = samples
    while remaining > 0:
        current = min(batch_size, remaining)
        remaining -= current
        x_np = rng.random((current, in_dim), dtype=np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(x_np)) * y_std + y_mean
        max_idx = int(torch.argmax(pred).item())
        max_val = float(pred[max_idx].item())
        if max_val > best_pred:
            best_pred = max_val
            best_x = x_np[max_idx]

    return best_x, best_pred


def suggest_next_point_gradient(
    model: SimpleRegressor,
    in_dim: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    n_restarts: int = 25,
    max_iter: int = 100,
    seed: int = 123,
) -> tuple[np.ndarray, float]:
    """
    Use gradient ascent to find the maximum.
    Multiple random restarts to avoid local optima.
    
    This is more efficient than random sampling, especially in high dimensions.
    """
    torch.manual_seed(seed)
    model.eval()
    best_x = None
    best_pred = -np.inf
    
    for _ in range(n_restarts):
        # Random starting point
        x = torch.rand(1, in_dim, requires_grad=True)
        optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            pred = model(x) * y_std + y_mean
            loss = -pred  # Negative because we want to maximize
            loss.backward()
            # Project to stay in [0, 1] bounds
            with torch.no_grad():
                x.clamp_(0, 1)
            return loss
        
        optimizer.step(closure)
        
        # Final evaluation
        with torch.no_grad():
            x.clamp_(0, 1)
            pred = model(x) * y_std + y_mean
            if pred.item() > best_pred:
                best_pred = pred.item()
                best_x = x.numpy()[0]
    
    return best_x, best_pred

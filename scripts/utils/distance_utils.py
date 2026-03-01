"""
Distance utilities for the Capstone Project.

Provides functions to compute Euclidean distances from each input to the
input that has the maximum output (the incumbent best), and to build
comparison tables for inputs/outputs across initial + weekly data.
"""

import numpy as np
import pandas as pd


def distances_to_best_input(X, y, relative=False):
    """
    Find the input with maximum output, then compute Euclidean distance from
    every input to that best input.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input points.
    y : array-like, shape (n_samples,)
        Output (objective) values.
    relative : bool, default False
        If True, normalize distances by the maximum distance so they lie
        in [0, 1] (relative to the spread of the data).

    Returns
    -------
    distances : np.ndarray, shape (n_samples,)
        Euclidean distance from each row of X to the best input.
    idx_best : int
        Index of the row with maximum output.
    x_best : np.ndarray, shape (n_features,)
        The input that has the maximum output.
    y_best : float
        The maximum output value.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    idx_best = np.argmax(y)
    x_best = X[idx_best]
    y_best = float(y[idx_best])

    # Euclidean distance from each input to the best input
    distances = np.linalg.norm(X - x_best, axis=1)

    if relative and distances.size > 0:
        d_max = distances.max()
        if d_max > 0:
            distances = distances / d_max

    return distances, idx_best, x_best, y_best


def distances_to_best_input_per_function(data_dict, relative=False):
    """
    Run distances_to_best_input on each function in a data dictionary.

    Expects data_dict to have keys like 'function_1', 'function_2', ... and
    each value to be a dict with 'inputs' and 'outputs' arrays (e.g. from
    load_initial_data or combined week data).

    Parameters
    ----------
    data_dict : dict
        Mapping function_name -> {'inputs': X, 'outputs': y}.
    relative : bool, default False
        If True, normalize distances by max distance within each function.

    Returns
    -------
    results : dict
        For each function_name, a dict with:
        - 'distances': array of distances for each input
        - 'idx_best': index of best input
        - 'x_best': best input vector
        - 'y_best': best output value
    """
    results = {}
    for name, data in data_dict.items():
        X = data["inputs"]
        y = data["outputs"]
        dist, idx_best, x_best, y_best = distances_to_best_input(X, y, relative=relative)
        results[name] = {
            "distances": dist,
            "idx_best": idx_best,
            "x_best": x_best,
            "y_best": y_best,
        }
    return results


def make_inputs_outputs_table(X, y, n_initial, x_proposed=None, y_proposed=None):
    """
    Build a comparison table: input number, output value, distance to best
    input, and summary of progress (improving / not improving vs previous).

    Designed for data ordered as: initial inputs (first n_initial rows), then
    one row per week (week 1, week 2, ...). Progress is "Improving" when the
    output is higher than the previous week (we assume we want to maximise).

    Optionally append a row for "proposed this week" with an
    estimated output and computed distance/progress.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        All inputs (initial + weekly).
    y : array-like, shape (n_samples,)
        All outputs.
    n_initial : int
        Number of initial points (rows 0 .. n_initial-1). Remaining rows are
        treated as week 1, week 2, etc.
    x_proposed : array-like, shape (n_features,), optional
        Proposed next input. If provided, a row with
        Source "proposed this week" is appended.
    y_proposed : float, optional
        Estimated output at x_proposed. If x_proposed is given and y_proposed
        is None, the output value for the proposed row is NaN.

    Returns
    -------
    pd.DataFrame
        Columns: Source, Input number, Output value, Distance to maximum input,
        Summary of progress. Source is the data folder (e.g. "initial", "week1",
        "week7") plus optionally "proposed this week".
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n = len(y)

    distances, idx_best, x_best, _y_best = distances_to_best_input(X, y, relative=False)

    # Source folder: "initial" for initial points, "week1", "week2", ... for weekly
    source = ["initial"] * n_initial + [f"week{k}" for k in range(1, n - n_initial + 1)]

    progress = []
    for i in range(n):
        if i < n_initial:
            progress.append("Initial")
        elif i == n_initial:
            # First week: compare to best of initial
            best_initial = np.max(y[:n_initial])
            progress.append("Improving" if y[i] > best_initial else "Not improving")
        else:
            # Later weeks: compare to previous week
            progress.append("Improving" if y[i] > y[i - 1] else "Not improving")

    # Optional proposed this week row
    if x_proposed is not None:
        x_proposed = np.asarray(x_proposed).ravel()
        dist_proposed = float(np.linalg.norm(x_proposed - x_best))
        source.append("proposed this week")
        n += 1
        distances = np.append(distances, dist_proposed)
        if y_proposed is not None:
            y_last = float(y[-1])
            progress.append("Improving" if y_proposed > y_last else "Not improving")
            y = np.append(y, float(y_proposed))
        else:
            progress.append("—")
            y = np.append(y, np.nan)

    return pd.DataFrame({
        "Source": source,
        "Input number": np.arange(1, n + 1),
        "Output value": y,
        "Distance to maximum input": distances,
        "Summary of progress": progress,
    })

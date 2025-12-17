"""
Data loading and utility functions for the Capstone Project

This module provides functions for:
- Loading initial data
- Loading week 1 data
- Combining datasets
- Visualization helpers
"""

import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_initial_data(base_dir=None):
    """
    Load initial data for all functions.
    
    Parameters:
    -----------
    base_dir : str or Path, optional
        Base directory path. If None, uses default path.
    
    Returns:
    --------
    initial_data : dict
        Dictionary with keys 'function_1' through 'function_8',
        each containing 'inputs' and 'outputs' arrays.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent / 'data' / 'initial_data'
    else:
        base_dir = Path(base_dir)
    
    initial_data = {}
    
    for function_num in range(1, 9):
        function_name = f'function_{function_num}'
        function_dir = base_dir / function_name
        
        inputs_path = function_dir / 'initial_inputs.npy'
        outputs_path = function_dir / 'initial_outputs.npy'
        
        if inputs_path.exists() and outputs_path.exists():
            inputs = np.load(inputs_path)
            outputs = np.load(outputs_path)
            
            initial_data[function_name] = {
                'inputs': inputs,
                'outputs': outputs
            }
        else:
            print(f"Warning: Files not found for {function_name}")
    
    return initial_data


def load_week1_data(week1_dir=None):
    """
    Load week 1 data for all functions.
    
    Parameters:
    -----------
    week1_dir : str or Path, optional
        Week 1 data directory path. If None, uses default path.
    
    Returns:
    --------
    week1_data : dict
        Dictionary with keys 'function_1' through 'function_8',
        each containing 'input' and 'output' arrays.
    """
    if week1_dir is None:
        week1_dir = Path(__file__).parent / 'data' / 'week1'
    else:
        week1_dir = Path(week1_dir)
    
    inputs_path = week1_dir / 'inputs.txt'
    outputs_path = week1_dir / 'outputs.txt'
    
    # Read and parse the text files
    with open(inputs_path, 'r') as f:
        inputs_text = f.read()
    
    with open(outputs_path, 'r') as f:
        outputs_text = f.read()
    
    # Parse the arrays
    week1_inputs_list = eval(inputs_text, {'array': np.array, 'np': np})
    week1_outputs_list = eval(outputs_text, {'np': np, 'np.float64': np.float64})
    
    # Convert to numpy arrays and organize by function
    week1_data = {}
    
    for function_num in range(1, 9):
        function_name = f'function_{function_num}'
        week1_data[function_name] = {
            'input': np.array(week1_inputs_list[function_num - 1]),
            'output': np.array(week1_outputs_list[function_num - 1])
        }
    
    return week1_data


def load_week2_data(week2_dir=None):
    """
    Load week 2 data for all functions.
    
    Parameters:
    -----------
    week2_dir : str or Path, optional
        Week 2 data directory path. If None, uses default path.
        Files may contain multiple lines; the last non-empty line is treated
        as the latest week of inputs/outputs.
    
    Returns:
    --------
    week2_data : dict
        Dictionary with keys 'function_1' through 'function_8',
        each containing 'input' and 'output' arrays for week 2.
    """
    if week2_dir is None:
        week2_dir = Path(__file__).parent / 'data' / 'week2'
    else:
        week2_dir = Path(week2_dir)
    
    inputs_path = week2_dir / 'inputs.txt'
    outputs_path = week2_dir / 'outputs.txt'
    
    def _split_blocks(lines):
        """Group lines into blocks that start with '[' to allow multi-line arrays."""
        blocks = []
        current = []
        for line in lines:
            if line.strip().startswith('[') and current:
                blocks.append(''.join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append(''.join(current))
        # Keep only non-empty trimmed blocks
        return [b.strip() for b in blocks if b.strip()]
    
    with open(inputs_path, 'r') as f:
        input_blocks = _split_blocks(f.readlines())
    with open(outputs_path, 'r') as f:
        output_blocks = _split_blocks(f.readlines())
    
    if not input_blocks or not output_blocks:
        raise ValueError("Week2 inputs/outputs files are empty or malformed.")
    
    # Use the last evaluable block to allow files that append previous weeks
    latest_inputs_list = None
    latest_outputs_list = None
    
    for in_block, out_block in zip(reversed(input_blocks), reversed(output_blocks)):
        try:
            candidate_inputs = eval(in_block, {'array': np.array, 'np': np})
            candidate_outputs = eval(out_block, {'np': np, 'np.float64': np.float64})
        except SyntaxError:
            # Skip malformed block and try the previous one
            continue
        latest_inputs_list = candidate_inputs
        latest_outputs_list = candidate_outputs
        break
    
    if latest_inputs_list is None or latest_outputs_list is None:
        raise ValueError("Could not parse a valid block from week2 inputs/outputs.")
    
    week2_data = {}
    for function_num in range(1, 9):
        function_name = f'function_{function_num}'
        week2_data[function_name] = {
            'input': np.array(latest_inputs_list[function_num - 1]),
            'output': np.array(latest_outputs_list[function_num - 1])
        }
    
    return week2_data


def combine_data(original_inputs, original_outputs, new_input, new_output):
    """
    Combine original data with new week data.
    
    Parameters:
    -----------
    original_inputs : array-like, shape (n_samples, n_dims)
        Original input data
    original_outputs : array-like, shape (n_samples,)
        Original output data
    new_input : array-like, shape (n_dims,)
        New input point
    new_output : float
        New output value
    
    Returns:
    --------
    combined_inputs : array, shape (n_samples+1, n_dims)
        Combined input data
    combined_outputs : array, shape (n_samples+1,)
        Combined output data
    """
    original_inputs = np.array(original_inputs)
    original_outputs = np.array(original_outputs)
    new_input = np.array(new_input)
    
    combined_inputs = np.vstack([original_inputs, new_input.reshape(1, -1)])
    combined_outputs = np.concatenate([original_outputs, np.array([new_output])])
    
    return combined_inputs, combined_outputs


def plot_2d_scatter_with_distribution(original_inputs, original_outputs, 
                                       new_input, new_output, 
                                       function_name, figsize=(16, 6),
                                       new_point_label="New point",
                                       prior_input=None, prior_output=None,
                                       prior_point_label="Prior point",
                                       connect_points=False,
                                       connect_color="tab:gray"):
    """
    Plot 2D scatter plot with output distribution for 2D functions.
    
    Parameters:
    -----------
    original_inputs : array-like, shape (n_samples, 2)
        Original input data
    original_outputs : array-like, shape (n_samples,)
        Original output data
    new_input : array-like, shape (2,)
        New input point
    new_output : float
        New output value
    function_name : str
        Name of the function (for titles)
    figsize : tuple, default=(16, 6)
        Figure size
    new_point_label : str, default="New point"
        Label for the new point in legends/text
    prior_input : array-like, optional
        Optional prior point to highlight (e.g., Week 1)
    prior_output : float, optional
        Output value for the prior point
    prior_point_label : str, default="Prior point"
        Label for the prior point in legends/text
    connect_points : bool, default=False
        If True, draw a line between prior and new points
    connect_color : str, default="tab:gray"
        Color for the connecting line
    """
    original_inputs = np.array(original_inputs)
    original_outputs = np.array(original_outputs)
    new_input = np.array(new_input)
    if prior_input is not None:
        prior_input = np.array(prior_input)
    if prior_output is not None:
        prior_output = float(prior_output)
    
    # Combine for color scale
    all_outputs = [original_outputs, np.array([new_output])]
    if prior_output is not None:
        all_outputs.append(np.array([prior_output]))
    all_outputs = np.concatenate(all_outputs)
    vmin = all_outputs.min()
    vmax = all_outputs.max()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Scatter plot
    x1_original = original_inputs[:, 0]
    x2_original = original_inputs[:, 1]
    scatter1 = axes[0].scatter(x1_original, x2_original, 
                               c=original_outputs, cmap='viridis', 
                               s=100, alpha=0.7, edgecolors='black', 
                               linewidth=1.5, label='Original points',
                               vmin=vmin, vmax=vmax)
    
    x1_new = new_input[0]
    x2_new = new_input[1]
    scatter2 = axes[0].scatter(x1_new, x2_new, c=[new_output], 
                              cmap='viridis', s=200, alpha=0.9, 
                              edgecolors='black', linewidth=2, marker='*',
                              label=new_point_label, vmin=vmin, vmax=vmax)
    
    if prior_input is not None and prior_output is not None:
        axes[0].scatter(prior_input[0], prior_input[1], c=[prior_output],
                        cmap='viridis', s=180, alpha=0.9,
                        edgecolors='black', linewidth=2, marker='*',
                        label=prior_point_label, vmin=vmin, vmax=vmax)
        if connect_points:
            axes[0].plot([prior_input[0], new_input[0]],
                         [prior_input[1], new_input[1]],
                         linestyle='--', color=connect_color, linewidth=2,
                         alpha=0.8)
    
    plt.colorbar(scatter1, ax=axes[0], label='Output value')
    axes[0].set_xlabel('x1 (Input 1)', fontsize=12)
    axes[0].set_ylabel('x2 (Input 2)', fontsize=12)
    axes[0].set_title(f'{function_name} - Scatter Plot of Combined Data', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Output distribution
    axes[1].hist(original_outputs, bins=min(10, len(original_outputs)), 
                edgecolor='black', alpha=0.7, label='Original outputs')
    axes[1].axvline(new_output, color='red', linestyle='--', linewidth=2, 
                   label=f'{new_point_label}: {new_output:.2e}')
    if prior_output is not None:
        axes[1].axvline(prior_output, color='orange', linestyle='-.', linewidth=2,
                       label=f'{prior_point_label}: {prior_output:.2e}')
    axes[1].set_xlabel('Output Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{function_name} - Output Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_3d_scatter_with_distribution(original_inputs, original_outputs,
                                      new_input, new_output,
                                      function_name, figsize=(18, 7),
                                      new_point_label="New point",
                                      prior_input=None, prior_output=None,
                                      prior_point_label="Prior point",
                                      connect_points=False,
                                      connect_color="tab:gray"):
    """
    Plot 3D scatter plot with output distribution for 3D functions.
    
    Parameters:
    -----------
    original_inputs : array-like, shape (n_samples, 3)
        Original input data
    original_outputs : array-like, shape (n_samples,)
        Original output data
    new_input : array-like, shape (3,)
        New input point
    new_output : float
        New output value
    function_name : str
        Name of the function (for titles)
    figsize : tuple, default=(18, 7)
        Figure size
    new_point_label : str, default="New point"
        Label for the new point in legends/text
    prior_input : array-like, optional
        Optional prior point to highlight (e.g., Week 1)
    prior_output : float, optional
        Output value for the prior point
    prior_point_label : str, default="Prior point"
        Label for the prior point in legends/text
    connect_points : bool, default=False
        If True, draw a line between prior and new points
    connect_color : str, default="tab:gray"
        Color for the connecting line
    """
    original_inputs = np.array(original_inputs)
    original_outputs = np.array(original_outputs)
    new_input = np.array(new_input)
    if prior_input is not None:
        prior_input = np.array(prior_input)
    if prior_output is not None:
        prior_output = float(prior_output)
    
    # Combine for color scale
    all_outputs = [original_outputs, np.array([new_output])]
    if prior_output is not None:
        all_outputs.append(np.array([prior_output]))
    all_outputs = np.concatenate(all_outputs)
    vmin = all_outputs.min()
    vmax = all_outputs.max()
    
    fig = plt.figure(figsize=figsize)
    
    # Left plot: 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    x1_original = original_inputs[:, 0]
    x2_original = original_inputs[:, 1]
    x3_original = original_inputs[:, 2]
    
    scatter1 = ax1.scatter(x1_original, x2_original, x3_original, 
                          s=80, c=original_outputs, cmap='viridis', 
                          alpha=0.8, edgecolors='black', linewidth=1,
                          vmin=vmin, vmax=vmax)
    
    x1_new = new_input[0]
    x2_new = new_input[1]
    x3_new = new_input[2]
    scatter2 = ax1.scatter(x1_new, x2_new, x3_new, s=300, c=[new_output], 
                          cmap='viridis', alpha=0.9, edgecolors='black', 
                          linewidth=2, marker='*', vmin=vmin, vmax=vmax,
                          label=new_point_label)
    
    if prior_input is not None and prior_output is not None:
        ax1.scatter(prior_input[0], prior_input[1], prior_input[2], s=260,
                   c=[prior_output], cmap='viridis', alpha=0.9, edgecolors='black',
                   linewidth=2, marker='*', vmin=vmin, vmax=vmax,
                   label=prior_point_label)
        if connect_points:
            ax1.plot([prior_input[0], new_input[0]],
                     [prior_input[1], new_input[1]],
                     [prior_input[2], new_input[2]],
                     linestyle='--', color=connect_color, linewidth=2,
                     alpha=0.8)
    
    ax1.set_xlabel('x1', fontsize=12)
    ax1.set_ylabel('x2', fontsize=12)
    ax1.set_zlabel('x3', fontsize=12)
    ax1.set_title(f'{function_name} - 3D Scatter Plot of Combined Data', fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='Output', shrink=0.8)
    ax1.grid(True)
    ax1.view_init(elev=25, azim=135)
    
    # Right plot: Output distribution
    ax2 = fig.add_subplot(122)
    ax2.hist(original_outputs, bins=min(15, len(original_outputs)), 
            edgecolor='black', alpha=0.7, label='Original outputs')
    ax2.axvline(new_output, color='red', linestyle='--', linewidth=2, 
               label=f'{new_point_label}: {new_output:.4f}')
    if prior_output is not None:
        ax2.axvline(prior_output, color='orange', linestyle='-.', linewidth=2,
                   label=f'{prior_point_label}: {prior_output:.4f}')
    ax2.set_xlabel('Output Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{function_name} - Output Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_tsne_with_distribution(original_inputs, original_outputs,
                                new_input, new_output,
                                function_name, figsize=(16, 6), random_state=42,
                                new_point_label="New point",
                                prior_input=None, prior_output=None,
                                prior_point_label="Prior point",
                                connect_points=False,
                                connect_color="tab:gray"):
    """
    Plot t-SNE visualization with output distribution for high-dimensional functions.
    
    Parameters:
    -----------
    original_inputs : array-like, shape (n_samples, n_dims)
        Original input data
    original_outputs : array-like, shape (n_samples,)
        Original output data
    new_input : array-like, shape (n_dims,)
        New input point
    new_output : float
        New output value
    function_name : str
        Name of the function (for titles)
    figsize : tuple, default=(16, 6)
        Figure size
    random_state : int, default=42
        Random seed for t-SNE
    new_point_label : str, default="New point"
        Label for the new point in legends/text
    prior_input : array-like, optional
        Optional prior point to highlight (e.g., Week 1)
    prior_output : float, optional
        Output value for the prior point
    prior_point_label : str, default="Prior point"
        Label for the prior point in legends/text
    connect_points : bool, default=False
        If True, draw a line between prior and new points (in t-SNE space)
    connect_color : str, default="tab:gray"
        Color for the connecting line
    """
    original_inputs = np.array(original_inputs)
    original_outputs = np.array(original_outputs)
    new_input = np.array(new_input)
    if prior_input is not None:
        prior_input = np.array(prior_input)
    if prior_output is not None:
        prior_output = float(prior_output)
    
    # Combine inputs for t-SNE
    combined_list = [original_inputs]
    if prior_input is not None:
        combined_list.append(prior_input.reshape(1, -1))
    combined_list.append(new_input.reshape(1, -1))
    combined_inputs = np.vstack(combined_list)
    
    # Combine for color scale
    all_outputs = [original_outputs, np.array([new_output])]
    if prior_output is not None:
        all_outputs.append(np.array([prior_output]))
    all_outputs = np.concatenate(all_outputs)
    vmin = all_outputs.min()
    vmax = all_outputs.max()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, 
               perplexity=min(30, len(combined_inputs)-1))
    inputs_2d = tsne.fit_transform(combined_inputs)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: t-SNE visualization
    original_indices = range(len(original_inputs))
    scatter1 = axes[0].scatter(inputs_2d[original_indices, 0], 
                              inputs_2d[original_indices, 1], 
                              c=original_outputs, cmap='viridis', 
                              s=100, alpha=0.7, edgecolors='black', 
                              linewidth=1.5, label='Original points',
                              vmin=vmin, vmax=vmax)
    
    prior_index = None
    if prior_input is not None:
        prior_index = len(original_inputs)
        axes[0].scatter(inputs_2d[prior_index, 0],
                        inputs_2d[prior_index, 1],
                        c=[prior_output], cmap='viridis',
                        s=180, alpha=0.9, edgecolors='black',
                        linewidth=2, marker='*',
                        label=prior_point_label, vmin=vmin, vmax=vmax)
    
    new_index = len(combined_inputs) - 1
    scatter2 = axes[0].scatter(inputs_2d[new_index, 0], 
                              inputs_2d[new_index, 1], 
                              c=[new_output], cmap='viridis', 
                              s=200, alpha=0.9, edgecolors='black', 
                              linewidth=2, marker='*',
                              label=new_point_label, vmin=vmin, vmax=vmax)
    
    if connect_points and prior_index is not None:
        axes[0].plot([inputs_2d[prior_index, 0], inputs_2d[new_index, 0]],
                     [inputs_2d[prior_index, 1], inputs_2d[new_index, 1]],
                     linestyle='--', color=connect_color, linewidth=2,
                     alpha=0.8)
    
    plt.colorbar(scatter1, ax=axes[0], label='Output value')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title(f'{function_name} - t-SNE Visualization of Combined Data', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Output distribution
    axes[1].hist(original_outputs, bins=min(15, len(original_outputs)), 
                edgecolor='black', alpha=0.7, label='Original outputs')
    axes[1].axvline(new_output, color='red', linestyle='--', linewidth=2, 
                   label=f'{new_point_label}: {new_output:.4f}')
    if prior_output is not None:
        axes[1].axvline(prior_output, color='orange', linestyle='-.', linewidth=2,
                       label=f'{prior_point_label}: {prior_output:.4f}')
    axes[1].set_xlabel('Output Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{function_name} - Output Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_bo_visualization_2d(bo, X_obs, y_obs, next_point, function_name, 
                            n_grid=50, beta=2.0, figsize=(18, 5)):
    """
    Visualize GP predictions and UCB for 2D functions.
    
    Parameters:
    -----------
    bo : SimpleBayesianOptimization
        Fitted BO object
    X_obs : array-like, shape (n_samples, 2)
        Observed input points
    y_obs : array-like, shape (n_samples,)
        Observed output values
    next_point : array-like, shape (2,)
        BO-suggested next point
    function_name : str
        Name of the function
    n_grid : int, default=50
        Grid resolution for contour plots
    beta : float, default=2.0
        UCB beta parameter
    figsize : tuple, default=(18, 5)
        Figure size
    """
    X_obs = np.array(X_obs)
    y_obs = np.array(y_obs)
    next_point = np.array(next_point)
    
    # Create grid
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.c_[X1_grid.ravel(), X2_grid.ravel()]
    
    # Get GP predictions
    mean, std = bo.predict(X_grid)
    mean_grid = mean.reshape(n_grid, n_grid)
    std_grid = std.reshape(n_grid, n_grid)
    
    # Get UCB values
    ucb = bo.acquisition_ucb(X_grid, beta=beta)
    ucb_grid = ucb.reshape(n_grid, n_grid)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: GP Mean
    im1 = axes[0].contourf(X1_grid, X2_grid, mean_grid, levels=20, cmap='viridis')
    axes[0].scatter(X_obs[:, 0], X_obs[:, 1], c=y_obs, s=100, 
                   cmap='viridis', edgecolors='black', linewidth=2, 
                   marker='o', label='Observed', zorder=5)
    axes[0].scatter(next_point[0], next_point[1], s=200, 
                   c='red', edgecolors='black', linewidth=2, 
                   marker='*', label='BO suggestion', zorder=6)
    axes[0].set_xlabel('x1', fontsize=12)
    axes[0].set_ylabel('x2', fontsize=12)
    axes[0].set_title(f'{function_name} - GP Mean Prediction', fontsize=14)
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0], label='Predicted mean')
    
    # Plot 2: GP Uncertainty (std)
    im2 = axes[1].contourf(X1_grid, X2_grid, std_grid, levels=20, cmap='plasma')
    axes[1].scatter(X_obs[:, 0], X_obs[:, 1], c='white', s=100, 
                   edgecolors='black', linewidth=2, marker='o', 
                   label='Observed', zorder=5)
    axes[1].scatter(next_point[0], next_point[1], s=200, 
                   c='red', edgecolors='black', linewidth=2, 
                   marker='*', label='BO suggestion', zorder=6)
    axes[1].set_xlabel('x1', fontsize=12)
    axes[1].set_ylabel('x2', fontsize=12)
    axes[1].set_title(f'{function_name} - GP Uncertainty (std)', fontsize=14)
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='Standard deviation')
    
    # Plot 3: UCB Acquisition Function
    im3 = axes[2].contourf(X1_grid, X2_grid, ucb_grid, levels=20, cmap='hot')
    axes[2].scatter(X_obs[:, 0], X_obs[:, 1], c='white', s=100, 
                   edgecolors='black', linewidth=2, marker='o', 
                   label='Observed', zorder=5)
    axes[2].scatter(next_point[0], next_point[1], s=200, 
                   c='cyan', edgecolors='black', linewidth=2, 
                   marker='*', label='BO suggestion', zorder=6)
    axes[2].set_xlabel('x1', fontsize=12)
    axes[2].set_ylabel('x2', fontsize=12)
    axes[2].set_title(f'{function_name} - UCB Acquisition Function', fontsize=14)
    axes[2].legend()
    plt.colorbar(im3, ax=axes[2], label='UCB value')
    
    plt.tight_layout()
    plt.show()


def plot_function_summary_subplot(ax_left, ax_right, original_inputs, original_outputs,
                                  week2_input, week2_output, bo_next_point,
                                  function_name, n_dims, random_state=42,
                                  week1_input=None, week1_output=None):
    """
    Plot function visualization in subplot axes (for summary grid).
    Shows input space visualization on left and output distribution on right.
    Matches the style of individual function plots.
    
    Parameters:
    -----------
    ax_left : matplotlib.axes
        Left axes for input space visualization
    ax_right : matplotlib.axes
        Right axes for output distribution
    original_inputs : array-like
        Original input data
    original_outputs : array-like
        Original output data
    week2_input : array-like
        Latest input point (e.g., Week 2)
    week2_output : float
        Output value for the latest input point
    bo_next_point : array-like
        BO-suggested next point
    function_name : str
        Name of the function
    n_dims : int
        Number of input dimensions
    random_state : int, default=42
        Random seed for t-SNE
    week1_input : array-like, optional
        Previous input point (e.g., Week 1)
    week1_output : float, optional
        Output value for the previous point
    """
    original_inputs = np.array(original_inputs)
    original_outputs = np.array(original_outputs)
    week2_input = np.array(week2_input)
    week2_output = float(week2_output)
    bo_next_point = np.array(bo_next_point)
    if week1_input is not None:
        week1_input = np.array(week1_input)
    if week1_output is not None:
        week1_output = float(week1_output)
    
    # Determine color scale
    all_outputs = [original_outputs, np.array([week2_output])]
    if week1_output is not None:
        all_outputs.append(np.array([week1_output]))
    all_outputs = np.concatenate(all_outputs)
    vmin = all_outputs.min()
    vmax = all_outputs.max()
    
    # Left plot: Input space visualization
    if n_dims == 2:
        # 2D scatter plot
        scatter = ax_left.scatter(original_inputs[:, 0], original_inputs[:, 1], 
                                 c=original_outputs, cmap='viridis', 
                                 s=80, alpha=0.7, edgecolors='black', 
                                 linewidth=1.5, vmin=vmin, vmax=vmax)
        # Week 2 point (colored by output)
        ax_left.scatter(week2_input[0], week2_input[1], s=150, 
                      c=[week2_output], cmap='viridis', alpha=0.9, 
                      edgecolors='black', linewidth=2, marker='*', 
                      zorder=5, vmin=vmin, vmax=vmax)
        if week1_input is not None and week1_output is not None:
            ax_left.scatter(week1_input[0], week1_input[1], s=140,
                          c=[week1_output], cmap='viridis', alpha=0.9,
                          edgecolors='black', linewidth=2, marker='*',
                          zorder=5, vmin=vmin, vmax=vmax)
            ax_left.plot([week1_input[0], week2_input[0]],
                         [week1_input[1], week2_input[1]],
                         linestyle='--', color='tab:orange', linewidth=2, alpha=0.8)
        # BO suggestion (black star)
        ax_left.scatter(bo_next_point[0], bo_next_point[1], s=150, 
                      c='black', edgecolors='white', linewidth=2, 
                      marker='*', zorder=6)
        ax_left.set_xlabel('x1', fontsize=9)
        ax_left.set_ylabel('x2', fontsize=9)
        plt.colorbar(scatter, ax=ax_left, label='Output', shrink=0.8)
    elif n_dims == 3:
        # 3D scatter plot - use t-SNE for consistency in summary grid
        # (3D plots are harder to display in small subplots)
        combined_list = [original_inputs]
        if week1_input is not None:
            combined_list.append(week1_input.reshape(1, -1))
        combined_list.append(week2_input.reshape(1, -1))
        combined_list.append(bo_next_point.reshape(1, -1))
        combined_inputs = np.vstack(combined_list)
        tsne = TSNE(n_components=2, random_state=random_state, 
                   perplexity=min(30, len(combined_inputs)-1))
        inputs_2d = tsne.fit_transform(combined_inputs)
        
        original_count = len(original_inputs)
        week1_idx = original_count if week1_input is not None else None
        week2_idx = original_count + (1 if week1_idx is not None else 0)
        bo_idx = week2_idx + 1
        
        scatter = ax_left.scatter(inputs_2d[:original_count, 0], inputs_2d[:original_count, 1], 
                                 c=original_outputs, cmap='viridis', 
                                 s=80, alpha=0.7, edgecolors='black', 
                                 linewidth=1.5, vmin=vmin, vmax=vmax)
        if week1_idx is not None:
            ax_left.scatter(inputs_2d[week1_idx, 0], inputs_2d[week1_idx, 1], s=150, 
                          c=[week1_output], cmap='viridis', alpha=0.9, 
                          edgecolors='black', linewidth=2, marker='*', 
                          zorder=5, vmin=vmin, vmax=vmax,
                          label='Week 1 point')
        ax_left.scatter(inputs_2d[week2_idx, 0], inputs_2d[week2_idx, 1], s=150, 
                      c=[week2_output], cmap='viridis', alpha=0.9, 
                      edgecolors='black', linewidth=2, marker='*', 
                      zorder=5, vmin=vmin, vmax=vmax,
                      label='Week 2 point')
        if week1_idx is not None:
            ax_left.plot([inputs_2d[week1_idx, 0], inputs_2d[week2_idx, 0]],
                         [inputs_2d[week1_idx, 1], inputs_2d[week2_idx, 1]],
                         linestyle='--', color='tab:orange', linewidth=2, alpha=0.8)
        # BO suggestion (black star)
        ax_left.scatter(inputs_2d[bo_idx, 0], inputs_2d[bo_idx, 1], s=150, 
                      c='black', edgecolors='white', linewidth=2, 
                      marker='*', zorder=6, label='BO point')
        ax_left.set_xlabel('t-SNE 1', fontsize=9)
        ax_left.set_ylabel('t-SNE 2', fontsize=9)
        plt.colorbar(scatter, ax=ax_left, label='Output', shrink=0.8)
    else:
        # t-SNE for higher dimensions
        combined_list = [original_inputs]
        if week1_input is not None:
            combined_list.append(week1_input.reshape(1, -1))
        combined_list.append(week2_input.reshape(1, -1))
        combined_list.append(bo_next_point.reshape(1, -1))
        combined_inputs = np.vstack(combined_list)
        tsne = TSNE(n_components=2, random_state=random_state, 
                   perplexity=min(30, len(combined_inputs)-1))
        inputs_2d = tsne.fit_transform(combined_inputs)
        
        original_count = len(original_inputs)
        week1_idx = original_count if week1_input is not None else None
        week2_idx = original_count + (1 if week1_idx is not None else 0)
        bo_idx = week2_idx + 1
        
        scatter = ax_left.scatter(inputs_2d[:original_count, 0], inputs_2d[:original_count, 1], 
                                 c=original_outputs, cmap='viridis', 
                                 s=80, alpha=0.7, edgecolors='black', 
                                 linewidth=1.5, vmin=vmin, vmax=vmax)
        if week1_idx is not None:
            ax_left.scatter(inputs_2d[week1_idx, 0], inputs_2d[week1_idx, 1], s=150, 
                          c=[week1_output], cmap='viridis', alpha=0.9, 
                          edgecolors='black', linewidth=2, marker='*', 
                          zorder=5, vmin=vmin, vmax=vmax,
                          label='Week 1 point')
        ax_left.scatter(inputs_2d[week2_idx, 0], inputs_2d[week2_idx, 1], s=150, 
                      c=[week2_output], cmap='viridis', alpha=0.9, 
                      edgecolors='black', linewidth=2, marker='*', 
                      zorder=5, vmin=vmin, vmax=vmax,
                      label='Week 2 point')
        if week1_idx is not None:
            ax_left.plot([inputs_2d[week1_idx, 0], inputs_2d[week2_idx, 0]],
                         [inputs_2d[week1_idx, 1], inputs_2d[week2_idx, 1]],
                         linestyle='--', color='tab:orange', linewidth=2, alpha=0.8)
        # BO suggestion (black star)
        ax_left.scatter(inputs_2d[bo_idx, 0], inputs_2d[bo_idx, 1], s=150, 
                      c='black', edgecolors='white', linewidth=2, 
                      marker='*', zorder=6, label='BO point')
        ax_left.set_xlabel('t-SNE 1', fontsize=9)
        ax_left.set_ylabel('t-SNE 2', fontsize=9)
        plt.colorbar(scatter, ax=ax_left, label='Output', shrink=0.8)
    
    ax_left.set_title(f'{function_name} ({n_dims}D) - with BO', fontsize=10)
    ax_left.grid(True, alpha=0.3)
    
    # Right plot: Same visualization WITHOUT BO point (matching individual plot)
    if n_dims == 2:
        # 2D scatter plot without BO
        scatter_right = ax_right.scatter(original_inputs[:, 0], original_inputs[:, 1], 
                                        c=original_outputs, cmap='viridis', 
                                        s=80, alpha=0.7, edgecolors='black', 
                                        linewidth=1.5, vmin=vmin, vmax=vmax)
        # Week 2 point (colored by output)
        ax_right.scatter(week2_input[0], week2_input[1], s=150, 
                       c=[week2_output], cmap='viridis', alpha=0.9, 
                       edgecolors='black', linewidth=2, marker='*', 
                       zorder=5, vmin=vmin, vmax=vmax)
        if week1_input is not None and week1_output is not None:
            ax_right.scatter(week1_input[0], week1_input[1], s=140,
                           c=[week1_output], cmap='viridis', alpha=0.9,
                           edgecolors='black', linewidth=2, marker='*',
                           zorder=5, vmin=vmin, vmax=vmax)
            ax_right.plot([week1_input[0], week2_input[0]],
                          [week1_input[1], week2_input[1]],
                          linestyle='--', color='tab:orange', linewidth=2, alpha=0.8)
        ax_right.set_xlabel('x1', fontsize=9)
        ax_right.set_ylabel('x2', fontsize=9)
        plt.colorbar(scatter_right, ax=ax_right, label='Output', shrink=0.8)
    elif n_dims == 3:
        # t-SNE without BO point (matching individual plot)
        combined_list_no_bo = [original_inputs]
        if week1_input is not None:
            combined_list_no_bo.append(week1_input.reshape(1, -1))
        combined_list_no_bo.append(week2_input.reshape(1, -1))
        combined_inputs_no_bo = np.vstack(combined_list_no_bo)
        tsne_no_bo = TSNE(n_components=2, random_state=random_state, 
                         perplexity=min(30, len(combined_inputs_no_bo)-1))
        inputs_2d_no_bo = tsne_no_bo.fit_transform(combined_inputs_no_bo)
        
        orig_count = len(original_inputs)
        week1_idx = orig_count if week1_input is not None else None
        week2_idx = orig_count + (1 if week1_idx is not None else 0)
        
        scatter_right = ax_right.scatter(inputs_2d_no_bo[:orig_count, 0], inputs_2d_no_bo[:orig_count, 1], 
                                        c=original_outputs, cmap='viridis', 
                                        s=80, alpha=0.7, edgecolors='black', 
                                        linewidth=1.5, vmin=vmin, vmax=vmax)
        if week1_idx is not None:
            ax_right.scatter(inputs_2d_no_bo[week1_idx, 0], inputs_2d_no_bo[week1_idx, 1], s=150, 
                           c=[week1_output], cmap='viridis', alpha=0.9, 
                           edgecolors='black', linewidth=2, marker='*', 
                           zorder=5, vmin=vmin, vmax=vmax)
        ax_right.scatter(inputs_2d_no_bo[week2_idx, 0], inputs_2d_no_bo[week2_idx, 1], s=150, 
                       c=[week2_output], cmap='viridis', alpha=0.9, 
                       edgecolors='black', linewidth=2, marker='*', 
                       zorder=5, vmin=vmin, vmax=vmax)
        if week1_idx is not None:
            ax_right.plot([inputs_2d_no_bo[week1_idx, 0], inputs_2d_no_bo[week2_idx, 0]],
                          [inputs_2d_no_bo[week1_idx, 1], inputs_2d_no_bo[week2_idx, 1]],
                          linestyle='--', color='tab:orange', linewidth=2, alpha=0.8)
        ax_right.set_xlabel('t-SNE 1', fontsize=9)
        ax_right.set_ylabel('t-SNE 2', fontsize=9)
        plt.colorbar(scatter_right, ax=ax_right, label='Output', shrink=0.8)
    else:
        # t-SNE for higher dimensions without BO point (matching individual plot)
        combined_list_no_bo = [original_inputs]
        if week1_input is not None:
            combined_list_no_bo.append(week1_input.reshape(1, -1))
        combined_list_no_bo.append(week2_input.reshape(1, -1))
        combined_inputs_no_bo = np.vstack(combined_list_no_bo)
        tsne_no_bo = TSNE(n_components=2, random_state=random_state, 
                         perplexity=min(30, len(combined_inputs_no_bo)-1))
        inputs_2d_no_bo = tsne_no_bo.fit_transform(combined_inputs_no_bo)
        
        orig_count = len(original_inputs)
        week1_idx = orig_count if week1_input is not None else None
        week2_idx = orig_count + (1 if week1_idx is not None else 0)
        
        scatter_right = ax_right.scatter(inputs_2d_no_bo[:orig_count, 0], inputs_2d_no_bo[:orig_count, 1], 
                                        c=original_outputs, cmap='viridis', 
                                        s=80, alpha=0.7, edgecolors='black', 
                                        linewidth=1.5, vmin=vmin, vmax=vmax)
        if week1_idx is not None:
            ax_right.scatter(inputs_2d_no_bo[week1_idx, 0], inputs_2d_no_bo[week1_idx, 1], s=150, 
                           c=[week1_output], cmap='viridis', alpha=0.9, 
                           edgecolors='black', linewidth=2, marker='*', 
                           zorder=5, vmin=vmin, vmax=vmax)
        ax_right.scatter(inputs_2d_no_bo[week2_idx, 0], inputs_2d_no_bo[week2_idx, 1], s=150, 
                       c=[week2_output], cmap='viridis', alpha=0.9, 
                       edgecolors='black', linewidth=2, marker='*', 
                       zorder=5, vmin=vmin, vmax=vmax)
        if week1_idx is not None:
            ax_right.plot([inputs_2d_no_bo[week1_idx, 0], inputs_2d_no_bo[week2_idx, 0]],
                          [inputs_2d_no_bo[week1_idx, 1], inputs_2d_no_bo[week2_idx, 1]],
                          linestyle='--', color='tab:orange', linewidth=2, alpha=0.8)
        ax_right.set_xlabel('t-SNE 1', fontsize=9)
        ax_right.set_ylabel('t-SNE 2', fontsize=9)
        plt.colorbar(scatter_right, ax=ax_right, label='Output', shrink=0.8)
    
    ax_right.set_title(f'{function_name} ({n_dims}D) - without BO', fontsize=10)
    ax_right.grid(True, alpha=0.3)

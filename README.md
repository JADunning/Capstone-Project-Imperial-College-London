# Black-Box Optimization (BBO) Capstone Project
**Imperial College London - Machine Learning Capstone**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Inputs and Outputs](#inputs-and-outputs)
3. [Challenge Objectives](#challenge-objectives)
4. [Technical Approach](#technical-approach)
5. [Setup Instructions](#setup-instructions)

---

## 1. Project Overview

### What is Black-Box Optimization?

Black-Box Optimization (BBO) is the problem of optimizing an unknown function where we have no knowledge of its internal structure, gradient information, or mathematical form. We can only observe input-output pairs by querying the function. This project tackles the challenge of **efficiently finding optimal parameter configurations** across eight different black-box functions, each with varying dimensionality (2D to 8D) and characteristics.

### Real-World Relevance

BBO is fundamental to many real-world machine learning and engineering applications for example: hyperparameter tuning where relationships between parameters and performance is unknown, or drug discovery where testing compound combinations is expensive and time-consuming.

The key challenge is that **function evaluations are expensive** (limited to ~10 queries per week), and we must be strategic about where to sample to find optimal solutions efficiently.

### Career Relevance

This project directly applies to my career goals in data science and machine learning engineering because it demonstrates:
- Systematic problem-solving
- Model selection and evaluation
- Working with limited data
- Documentation and communication


## 2. Inputs and Outputs

### Input Format

Each function accepts a **continuous input vector** of fixed dimensionality, with all values bounded in **[0, 1]**:

| Function | Dimensions | Input Format | Example Input |
|----------|-----------|--------------|---------------|
| Function 1 | 2D | `[x₁, x₂]` | `[0.5, 0.5]` |
| Function 2 | 2D | `[x₁, x₂]` | `[0.65, 0.5]` |
| Function 3 | 3D | `[x₁, x₂, x₃]` | `[0.3, 0.7, 0.4]` |
| Function 4 | 4D | `[x₁, x₂, x₃, x₄]` | `[0.25, 0.5, 0.75, 0.4]` |
| Function 5 | 4D | `[x₁, x₂, x₃, x₄]` | `[0.6, 0.3, 0.8, 0.2]` |
| Function 6 | 5D | `[x₁, ..., x₅]` | `[0.2, 0.4, 0.6, 0.8, 0.5]` |
| Function 7 | 6D | `[x₁, ..., x₆]` | `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` |
| Function 8 | 8D | `[x₁, ..., x₈]` | `[0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6]` |

**Constraints:**
- All input values must be in the range `[0, 1]`
- Each function has a fixed dimensionality
- Inputs are submitted as text files with one query per line

### Output Format

Each function returns a **single scalar value** representing the performance at that input location:

```python
# Example: Function 2
input = [0.65, 0.5]
output = 0.427  # Higher is better (maximization)
```

**Output Characteristics:**
- **Type**: Continuous scalar (float)
- **Optimization Goal**: Maximize all functions
- **Noise**: Some functions exhibit noisy outputs (e.g., Function 3)
- **Scale**: Output scales vary significantly across functions (from ~0 to thousands)
- **Sparsity**: Some functions (e.g., Function 1) return 0 for most of the input space


## 3. Challenge Objectives

### Primary Goal

**Maximize the output** of eight black-box functions with minimal function evaluations. The competition runs for 10 weeks with approximately **1 query per function per week** (~80 total queries across all functions).

### Constraints and Limitations

1. **Limited Queries**: Only ~10 evaluations per function over the entire project duration
2. **Unknown Function Structure**: 
   - May contain multiple local optima
   - May have discontinuities or flat regions
   - Different levels of noise and smoothness
3. **Evaluation Delay**: Weekly submission cadence requires planning ahead


## 4. Technical Approach

This section documents my evolving optimization strategy across the first three weeks of submissions. The approach emphasizes **early exploration** before transitioning to **exploitation** in later weeks.

### Overall Strategy (10-Week Plan)

**Weeks 1-3: Exploration Phase**
- Focus on sampling diverse regions of the input space
- Build understanding of function behavior across all dimensions
- Collect data for training surrogate models

**Weeks 4-6: Transition Phase**
- Begin incorporating exploitation near promising regions
- Tune surrogate model hyperparameters
- Experiment with different functions

**Weeks 7-10: Exploitation Phase**
- Focus on refining best-known regions
- Consider function-specific strategies (e.g., logistic regression for Function 1)
- Potentially employ neural networks for high-dimensional functions (7, 8)


---

## 5. Setup Instructions

### Activate Virtual Environment

Before installing packages or running code, activate the virtual environment:

```bash
source .venv/bin/activate
```

### Install Requirements

Once the virtual environment is activated, install all required packages:

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `numpy>=1.24.0` - Array operations
- `scipy>=1.10.0` - Optimization routines
- `matplotlib>=3.7.0` - Visualization
- `pandas>=2.0.0` - Data handling
- `scikit-learn>=1.3.0` - Gaussian Processes
- `scikit-optimize>=0.9.0` - Bayesian Optimization tools
- `jupyter>=1.0.0` - Notebook interface

**Note:** In Jupyter notebooks, you can use `%pip install -r requirements.txt` directly in a cell.

### Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── function_description_and_details.md # Function specifications
├── scripts/                           # Python modules
│   ├── bayesian_optimisation/         # BO (standard, exploitative, noise-aware)
│   ├── logistic_regression/           # Linear and nonlinear logistic regression
│   ├── neural_networks/               # PyTorch MLP next-point suggestion
│   └── utils/                         # Data loading and visualization (data_utils)
├── data/                              # Query data by week
│   ├── initial_data/
│   ├── week1/
│   ├── week2/
│   └── week3/
├── weeklyDiary/                       # Weekly reflections
│   ├── week1.md
│   ├── week2.md
│   └── week3.md
├── week1-*.ipynb                      # Analysis notebooks
├── week2.ipynb
└── week3.ipynb
```

---

## References and Resources

- **Bayesian Optimization**: Shahriari, B., et al. (2016). "Taking the human out of the loop: A review of Bayesian optimization." *Proceedings of the IEEE*.
- **Gaussian Processes**: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- **Acquisition Functions**: Srinivas, N., et al. (2010). "Gaussian process optimization in the bandit setting: No regret and experimental design." *ICML*.

---

## Contact

**Jack Dunning**  
Imperial College London  
MSc Machine Learning

*This README is a living document and will be updated as the project progresses.*

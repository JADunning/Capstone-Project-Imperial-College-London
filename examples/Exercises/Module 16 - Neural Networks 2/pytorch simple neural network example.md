 Mini-lesson 16.7: Building your own neural networks
Neural networks are powerful tools for classification, but understanding how they compute outputs can feel abstract. In this lesson, you’ll demystify the process by building a simple feedforward neural network in PyTorch, inspecting its layers, and visualizing its architecture. You’ll also explore how activations and weights shape predictions — both algebraically and visually.

If you'd like a refresher on vector operations, norms and activation functions, refer to Mini-lessons 1.2–1.8 from Module 1: Mathematical Concepts in ML/AI.

What is a decision boundary?
A decision boundary is a hypersurface that divides the feature space into regions corresponding to different predicted classes.

In a binary classification problem, points on one side of the boundary are classified as Class 0 and on the other side as Class 1.
Neural networks learn these boundaries from training data by adjusting weights and biases.
Key factors affecting decision boundaries
Number of neurons per layer: more neurons increase the network’s capacity to model complex shapes.
Number of hidden layers: additional layers allow the model to capture more abstract features.
Activation function: functions such as tanh, ReLU and sigmoid control how neurons transform input signals and influence the smoothness or sharpness of the boundaries
Steps to visualise decision boundaries in Python
Neural networks may seem complex, but they’re built from simple building blocks: linear transformations, activation functions and layer compositions. In this lesson, you’ll construct a feedforward neural network using PyTorch, inspect its weights and biases and understand how data flows through each layer.

Select each tab to learn more.

Step 1: define a linear layer.
Start by importing PyTorch and creating a basic linear layer:

import torch
import torch.nn as nn

# Create a linear layer with two input features and two output features
F_1 = nn.Linear(in_features=2, out_features=2, bias=True)

# Inspect the randomly initialised weights and biases
print("Weights:", F_1.weight)
print("Biases:", F_1.bias)
This layer performs the transformation: 

Step 2: add an activation function.
Define common activation functions manually:

import numpy as np
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
These functions introduce non-linearity, allowing the network to model complex patterns.

Step 3: pass input through a layer.
Create an input tensor and pass it through a second layer with ReLU activation:

x = torch.from_numpy(np.array([1., 2.], dtype=np.float64)).float()

L_2 = nn.Linear(in_features=2, out_features=3)
F_2 = lambda x: nn.functional.relu(L_2(x))

output = F_2(x)
print("Activated output:", output)
Step 4: manually compute the forward pass.
To understand what PyTorch is doing, extract the weights and biases, and compute the transformation manually:

weights = L_2.weight.detach().numpy()
biases = L_2.bias.detach().numpy()
x_vect = x.detach().numpy().reshape(2, 1)

# Affine transformation
input_of_L2 = weights.dot(x_vect) + biases.reshape(3, 1)

# Apply ReLU activation
output_of_L2 = relu(input_of_L2)
print("Manual output:", output_of_L2)
Step 5: add a final layer and compose the network.
Define a third layer and compose the full network:

L_3 = nn.Linear(in_features=3, out_features=2)
F_3 = lambda x: L_3(x)

# Compose all layers
F = lambda x: F_3(F_2(F_1(x)))

# Evaluate the network
x = torch.from_numpy(np.array([1., 2.], dtype=np.float64)).float()
y = F(x)
print("Final output:", y)
Step 6: visualise the network architecture.
Use nn.Sequential and torchviz to visualise the computation graph:

from torch.autograd import Variable
from torchviz import make_dot

model = nn.Sequential()
model.add_module('W1', nn.Linear(2, 2))
model.add_module('W2', nn.Linear(2, 3))
model.add_module('relu', nn.ReLU())
model.add_module('W3', nn.Linear(3, 2))

x = Variable(torch.randn(1, 2))
y = model(x)

make_dot(y, params=dict(model.named_parameters()))
This graph shows how data flows through the network and how each layer transforms the input.

Step 7: model logistic regression with a neural network.
Use a single-layer network with a sigmoid activation to model logistic regression:

L_1 = nn.Linear(in_features=2, out_features=1, bias=True)

# Set the weights and bias manually
L_1.weight.data = torch.tensor([[0.3, -0.1]])
L_1.bias.data = torch.tensor([1.0])

F_1 = lambda x: nn.functional.sigmoid(L_1(x))

x = torch.from_numpy(np.array([1, -2], dtype=np.float64)).float()
y = F_1(x)
print("Estimated probability:", round(y.item(), 4))
This mirrors the logistic regression formula: 

Summary
This mini-lesson demonstrated how to construct a neural network using PyTorch, inspect its internal parameters and visualise its architecture. It also illustrated how logistic regression can be represented within a neural network framework. These foundational techniques support effective model design, interpretation and debugging – skills that are essential for applied ML tasks such as those you have encountered in the capstone project.
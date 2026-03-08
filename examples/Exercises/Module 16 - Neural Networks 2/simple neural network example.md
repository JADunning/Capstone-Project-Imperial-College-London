Mini-lesson 16.5: A simple neural network example
 Mini-lesson 16.5: A simple neural network example
In Video 16.6, you explored a simple neural network example coded from scratch, where each step – from forward propagation to backpropagation – was implemented manually. This mini-lesson builds on that foundation by showing you how the exact same network can be expressed in two more concise and powerful ways using PyTorch. These examples not only streamline the code but also highlight PyTorch features that automate repetitive tasks, allowing you to focus more on model design and experimentation.

You’ll see three versions of the same network side by side:

Select each tab to learn more.

Original version
As you saw in Video 16.6, this consists of a fully manual forward pass, loss computation and weight updates using Python and NumPy.

The vanilla approach of building a neural network manually using Python and NumPy involves explicitly coding every step of the network operations without relying on deep learning frameworks. This approach helps in understanding the fundamental mechanics of neural networks. The key steps include the following:

Initialise weights randomly for each layer.
Perform forward propagation to compute predictions:
Multiply inputs with weights (matrix multiplication).
Apply an activation function (e.g. ReLU).
Compute the loss by comparing predictions with true outputs (e.g. mean squared error).
Backpropagate errors manually:
Compute gradients of the loss with respect to weights using the chain rule.
Update weights using gradient descent with the computed gradients.
This manual method uses NumPy arrays for matrix operations and shows explicitly how all the computations flow through the network without automation. The vanilla approach excels as an educational tool to learn the workings of neural networks but is impractical for real-world or larger network tasks, which benefit from automation, optimisation and modular frameworks such as PyTorch.

PyTorch version 1: manual backprop with tensors
Uses PyTorch tensors instead of NumPy arrays but still implements forward and backward passes manually

This step demonstrates how PyTorch can integrate seamlessly with the manual approach while offering GPU acceleration and tensor utilities.

The refactoring in this version involves transitioning from pure NumPy arrays to PyTorch tensors but still manually implementing every step of backpropagation.

Example code snippet:

import torch
 dtype = torch.float
 device = torch.device("cpu")


#Numpy arrays are replaced with PyTorch tensors
 N, D_in, H, D_out = 64, 1000, 100, 10
 x = torch.randn(N, D_in, device=device, dtype=dtype)
 y = torch.randn(N, D_out, device=device, dtype=dtype)
 w1 = torch.randn(D_in, H, device=device, dtype=dtype)
 w2 = torch.randn(H, D_out, device=device, dtype=dtype)
 learning_rate = 1e-6

#Manual forward pass using tensor operations
 for t in range(500):
      h = x.mm(w1) #Matrix multiplication
      h_relu = h.clamp(min=0) #ReLU activation
      y_pred = h_relu.mm(w2) #Hidden → output

      loss = (y_pred - y).pow(2).mean().item()
      print(t, loss)

  #Manual backward pass computations
      grad_y_pred = 2.0 * (y_pred - y) #Grad of loss with respect to predictions
      grad_w2 = h_relu.t().mm(grad_y_pred) #Grad of loss with respect to w2
      grad_h_relu = grad_y_pred.mm(w2.t()) #Grad of loss with respect to hidden layer after ReLU
      grad_h = grad_h_relu.clone()
      grad_h[h < 0] = 0 #Apply ReLU derivative
      grad_w1 = x.t().mm(grad_h) #Grad of loss wrt w1

      w1 -= learning_rate * grad_w1
      w2 -= learning_rate * grad_w2
The main benefit of this approach is that PyTorch tensor operations support hardware acceleration (e.g. GPU), unlike NumPy. It has access to more efficient and flexible tensor computations with built-in functions. But it still requires manual gradient calculations, which are prone to errors. 

PyTorch version 2: using autograd
Leverages PyTorch’s built-in autograd to automatically compute gradients and update weights

This eliminates manual backpropagation code, reducing errors and improving development speed.

import torch
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") #Uncomment this to run on GPU

#N is batch size; D_in is input dimension
#H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
      y_pred = x.mm(w1).clamp(min=0).mm(w2)

      #Compute and print loss using operations on tensors
      #Now, loss is a tensor of shape (1,)
      #loss.item() gets the scalar value held in the loss
      loss = (y_pred - y).pow(2).sum()
      print(t, loss.item())
      loss.backward()
      with torch.no_grad():
      w1 -= learning_rate * w1.grad
      w2 -= learning_rate * w2.grad

      w1.grad.zero_()
      w2.grad.zero_()
In this version, PyTorch's autograd is introduced to automate gradient computations. requires_grad=True is set on weight tensors to track operations. The forward pass is performed normally. The gradients are automatically computed using .backward(). Weight updates are performed manually using .grad(). The forward pass is combined in a single step, y_pred = x.mm(w1).clamp(min=0).mm(w2), making the code more concise while preserving the same computation. loss.backward() automatically computes gradients for all tensors with requires_grad=True. This removes the need for manual gradient derivations. This new version includes clearer comments explaining the significance of requires_grad and the use of autograd as well as a hint that torch.optim.SGD could be used instead for updating weights more cleanly.

The main benefit of this approach is that it removes error-prone manual gradient code, increasing reliability and speeding up experimentation.

Download the accompanying Python files for the above versions.Links to an external site.

Why this matters
While manually coding each step is invaluable for learning, most real-world deep learning projects use libraries such as PyTorch to simplify repetitive tasks. Understanding both the manual and automated approaches equips you to debug effectively, optimise performance and choose the right level of abstraction for each project.

If you'd like a refresher on the mathematical foundations of backpropagation and matrix operations, refer to Mini-lessons 1.2–1.11 and 1.16 from Module 1: Mathematical Concepts in ML/AI.
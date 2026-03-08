 Mini-lesson 15.9: Applying gradient descent and backpropagation in Python
When training neural networks, the goal is to adjust the model’s parameters – weights and biases – so that the network learns to map inputs to outputs with minimal error. This learning process is powered by two key techniques: gradient descent, which optimises the model by minimising the loss function, and backpropagation, which efficiently computes gradients for all parameters using the chain rule.

Gradient descent is an optimisation algorithm used to minimise the loss function, which quantifies how far the network's predictions are from the true values. It works by iteratively updating the weights in the direction that reduces this loss, following the slope (gradient) of the loss function. The step size of these updates is controlled by the learning rate, which determines how large each step should be.

Step-by-step procedure for implementing gradient descent
Select each tab to learn more. 

Step 1: define the function 
.
def fn(x1, x2):
    # return the value of the function at (x1, x2)
    return ...
Step 2: define the gradient function 
.
Define a function that returns the gradient vector 
.

def grad(x1, x2):
    # return numpy array of partial derivatives
    return np.array([... , ...])
Step 3: initialise the starting point, step size, iteration counter and stopping condition.
Set the starting point 
, learning rate 
, iteration count as zero and a variable to track stopping condition.

x0 = np.zeros(2)
alpha = 0.01
iteration = 0
condition = 0
xk = x0.copy()   # current point
Step 4: set up the gradient descent loop.
Use a while loop that runs until the stopping criterion is met (based on gradient magnitude near zero).

Inside the loop:

Increment the iteration counter.

Every 50 iterations, print the current iterate 
.

Update 
.

Check whether the norm of the gradient at the new point 
 is less than or equal to 10− 6. If yes, set the stopping condition to the end of the loop.

Update the current point 
 to the new point.

while condition == 0:
    iteration += 1
    if iteration % 50 == 0:
        print(xk)
    xk1 = xk - alpha * grad(xk[0], xk[1]) # The update step moves against the gradient to reduce the function value
if np.linalg.norm(grad(xk1[0], xk1[1])) <= 10**-6:
        condition = 1
    xk = xk1
Step 5: after the loop terminates, print the results.
Print the total number of iterations required for convergence and the final value of 
.

print(f"Converged in {iteration} iterations")
print(f"Final x = {xk}")
Backpropagation is the algorithm that efficiently computes these gradients in multilayer neural networks. It applies the chain rule of calculus to propagate the error from the output layer backward through the network, layer by layer, allowing you to calculate how each weight contributes to the overall error.

Step-by-step procedure for optimising neural networks using backpropagation
For a simple neural network with one hidden layer with two neurons and one output layer, let’s see the steps in computing the forward and backward passes. 

Select each tab to learn more. 

Step 1: define the input data and target output and initialisation.
Initialise the weights to small random numbers.
Define the trainable weight variables for TensorFlow. This is done by using the tf.Variable().
Step 2: perform a forward pass.
The forward pass refers to the computational phase in which input data is propagated through the layers of a neural network to generate output predictions. The forward_pass(x) function specifically implements forward propagation for a feedforward neural network with a single hidden layer.

In this implementation:

The network applies a ReLU activation function after the affine transformation of the input layer, introducing nonlinearity between the input and hidden layers.

The sigmoid activation function is then applied between the hidden layer and the output layer, mapping the output to a probability-like value in the range (0, 1).

Mathematically, the forward pass performs the following steps:

Hidden layer computation:



Output layer computation:



where 
 denotes the sigmoid function.

Here, 
 and 
 are the weight matrices for the input-to-hidden and hidden-to-output layers respectively, while 
 and 
 are the corresponding bias terms.

Calling forward_pass(x_train)executes this sequence on the training data set x_train, returning the network's current output predictions based on the learned parameters 
 and 
.

def forward_pass(x):
    bias = tf.ones((tf.shape(x)[0], 1))
    x_with_bias = tf.concat([x, bias], axis=1)
    hidden_linear = tf.matmul(x_with_bias, W1)
    hidden_relu = tf.nn.relu(hidden_linear)
    output_linear = tf.matmul(hidden_relu, W2)
    output_sigmoid = tf.sigmoid(output_linear)
    return output_sigmoid
Step 3: use the loss function and optimiser.
The loss function and optimiser enable the neural network to learn from data by gradually improving the model’s predictions through iterative refinement. TensorFlow uses the loss function tf.keras.losses.MeanSquaredError() and the optimiser function tf.optimizers.SGD(learning_rate = 0.1) where SGD is a stochastic gradient descent.

Step 4: perform forward pass after one gradient descent step.
The train_step() performs one gradient descent step: it computes predictions, evaluates the loss, calculates gradients and updates the weights using the optimiser.

def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = forward_pass(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, [W1, W2])
    optimizer.apply_gradients(zip(gradients, [W1, W2]))
    return loss
tf.GradientTape() is a key tool for implementing custom training loops and performing backpropagation in TensorFlow, giving you explicit control over gradient computation and updates.

Step 5: run the training loop.
This training loop is run repeatedly for a set number of iterations. It updates the model’s weights to minimise the loss and improve prediction accuracy over time.

for epoch in range(n):
    loss = train_step(x_train, y_train)
Step 6: make predictions after training.
After training a neural network, the model has learned patterns from the training data by updating its weights to minimise errors. The next important step is to use the trained model to make predictions on new, unseen data.

y_pred = forward_pass(x_test).numpy()
A test sample is passed through the forward_pass() and the output is stored in y_pred.

Summing up, gradient descent and backpropagation are core techniques used to train neural networks by iteratively adjusting weights and biases to minimise prediction errors via gradients of the loss function. The process involves computing forward passes to generate predictions, calculating loss and then using backpropagation with gradient descent to update parameters until convergence, enabling the network to learn accurate input–output mappings.
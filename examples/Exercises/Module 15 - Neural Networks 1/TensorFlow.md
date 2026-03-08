 Mini-lesson 15.5: Introduction to TensorFlow
TensorFlow is a powerful open-source library commonly used for developing and training neural networks in deep learning applications. It provides a flexible and efficient platform for building a variety of neural network architectures, from simple feedforward networks to complex deep learning models such as CNNs and recurrent neural networks (RNNs). TensorFlow’s popularity stems in part from its user-friendly APIs, especially Keras, which makes the process of creating, training and evaluating models straightforward for both beginners and experts.

Basic concepts of Tensorflow
Tensors: multi-dimensional arrays used as the basic data structure
Computational graph: a network of operations that define the data flow
Model: a structured network of layers (composed of neurons) used to learn patterns from data
Training: the process of feeding labeled data to the model so it can learn from it
Inference: using the trained model to make predictions on new or unseen data
Keras is integrated within TensorFlow as tf.keras. It allows users to build neural networks easily by stacking layers. Keras abstracts much of the complexity, making it especially accessible for beginners.

Below are the core steps for developing a neural network with TensorFlow. Each step follows industry-standard practices and includes concise code examples that are easy to adapt for practical use.

Let’s start with a simple sequential and practical example to classify digits using Modified National Institute of Standards and Technology (MNIST) data. Once this is clear, in the required assignment the focus would be on the working of the different layers.

Step 1: import the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
Step 2: load the data
# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
Step 3: preprocess the data
# Preprocess data: Flatten each 28×28 image (2D) into a 784-length (1D) vector, then normalize pixel values to the range [0, 1] for input into the dense layer.
train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32') / 255
Step 4: build the model
The model can be built by manually specifying the number of neurons in each of the layers (inputs, hidden and output). The code below defines a simple two-layer feedforward neural network for multi-class classification, with a flattened input layer, one hidden layer (128 ReLU units) and a softmax output layer for probabilities.

# Build a simple feedforward neural network (one hidden layer with 128 neurons and ReLU activation, output layer with 10 neurons for classification)
model = Sequential([
 Dense(128, activation='relu', input_shape=(28 * 28,)),
 Dense(10, activation='softmax')
])
Step 5 (optional): set or inspect weights and biases
You can inspect or set a model’s weights and biases for debugging or reproducibility. Use .get_weights() to retrieve them, and .set_weights() to assign new values:

model.layers[1].set_weights([new_weights, new_biases])
You can obtain the weights and biases by using the .get_weights().

Step 6: compile the model with optimiser and loss function
# Compile with optimizer and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
Step 7: train the model
# Train the model
model.fit(train_images, train_labels, epochs=5)
Step 8: evaluate the model
# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# The layers can be inspected using model.summary()
Developing a neural network in TensorFlow begins by importing the necessary libraries and preparing the data – typically by converting images or numbers into tensors suitable for model input. The model architecture is then built using Keras, commonly with the Sequential API. Optionally, weights and biases can be inspected or initialised manually. After compiling the model with a chosen optimiser and loss function, training is performed to optimise the model's parameters. Finally, the model is evaluated on test data or used for inference. This streamlined workflow enables efficient and scalable development of deep learning models using TensorFlow.
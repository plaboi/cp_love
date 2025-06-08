import numpy as np
import nnfs 
from nnfs.datasets import spiral_data #example dataset

nnfs.init()  # Initialize nnfs library


class Layer_Dense: #layer to layer with weights and biases 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # outputting random weights 
        self.biases = np.zeros((1, n_neurons))  # initializing biases to zero
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_ReLU: #activation function ess - negative goes to zero 
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #prevents overflow 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalises the val 
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)  # Number of samples in the batch
        # Clip data to prevent inf
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) #~bias 
        # Calculate loss
        if len(y_true.shape) == 1:  # passed scalar values 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  #one-hot encoded values 
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences ) 
        return negative_log_likelihoods


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3) #input has to match the input size obvs
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3) #output side is 3 because of the classes = 3 
activation2 = Activation_Softmax()


dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])  # first 5 ouputs 

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)
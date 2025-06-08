import numpy as np
import nnfs 
from nnfs.datasets import spiral_data #example dataset
import matplotlib.pyplot as plt
nnfs.init()  # Initialize nnfs library

#Dense layer class 
class Layer_Dense: #layer to layer with weights and biases 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # outputting random weights 
        self.biases = np.zeros((1, n_neurons))  # initializing biases to zero
    def forward(self, inputs): #forward pass 
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases 

    def backward(self, dvalues): #backward pass
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU Activation
class Activation_ReLU: #activation function ess - negative goes to zero 
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0  

#Softmax Activation 
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #prevents overflow 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalises the val 
        self.output = probabilities
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1) 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)  # Jacobian matrix
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

#Combined softmax activation and corss-entropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)
    
    def backward(self,dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2: #flag 
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
    
    def __init__(self):
        self.loss = Loss_CategoricalCrossentropy()  # Initialize the loss function

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

#Get input of dataset 
X, y = spiral_data(samples=100, classes=3)


#Create layers
dense1 = Layer_Dense(2,9) #input has to match the input size obvs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(9,9) #output side is 3 because of the classes = 3 
activation2 = Activation_ReLU()
dense3 = Layer_Dense(9,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#dense1.forward(X)
#activation1.forward(dense1.output)
#dense2.forward(activation1.output)
#activation2.forward(dense2.output)
#dense3.forward(activation2.output)


# Training loop
for epoch in range(50000):
    # Forward pass (edit with differnet layers config) 
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y)

    # Accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Loss: {float(np.mean(loss)):.3f}, Accuracy: {accuracy:.3f}')


    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases (SGD)
    for layer in [dense1, dense2, dense3]:
        layer.weights += -0.05 * layer.dweights
        layer.biases += -0.05 * layer.dbiases

np.savez("model_weights.npz", 
         dense1_weights=dense1.weights, dense1_biases=dense1.biases,
         dense2_weights=dense2.weights, dense2_biases=dense2.biases,
         dense3_weights=dense3.weights, dense3_biases=dense3.biases)







#print(activation2.output[:5])  # first 5 ouputs 

#loss_function = Loss_CategoricalCrossentropy()
#loss = loss_function.calculate(activation2.output, y)
#print("Loss:", loss) 
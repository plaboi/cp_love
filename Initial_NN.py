import numpy as np

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]] #Data sets

#weights should be minimal -1 ~ 1 

np.random.seed(0)  


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # outputting random weights 
        self.biases = np.zeros((1, n_neurons))  # initializing biases to zero
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

layer1 = Layer_Dense(4, 5) # 4 inputs, 5 neurons
layer2 = Layer_Dense(5, 2) # 5 inputs from layer1, 2 neurons in layer2 

layer1.forward(X)
#print(layer1.output)  # Output from layer1
layer2.forward(layer1.output)
print(layer2.output)  # Output from layer2
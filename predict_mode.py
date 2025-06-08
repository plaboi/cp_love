import numpy as np

# Define your classes exactly as in training (only forward pass needed)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = None
        self.biases = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Load saved model weights
data = np.load("model_weights.npz")

# Create layers with correct sizes
dense1 = Layer_Dense(2, 9)
dense2 = Layer_Dense(9, 9)
dense3 = Layer_Dense(9, 3)

# Assign weights and biases from saved file
dense1.weights = data["dense1_weights"]
dense1.biases = data["dense1_biases"]
dense2.weights = data["dense2_weights"]
dense2.biases = data["dense2_biases"]
dense3.weights = data["dense3_weights"]
dense3.biases = data["dense3_biases"]

# Create activation objects
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Softmax()

def predict(X_input):
    dense1.forward(X_input)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    return np.argmax(activation3.output, axis=1)

# Example usage
if __name__ == "__main__":
    # Example data to predict - 2 features per sample
    X_new = np.array([[0.5, -0.2],
                      [1.0, 1.0],
                      [-1.5, 2.0]])
    predictions = predict(X_new)
    print("Predicted classes:", predictions)

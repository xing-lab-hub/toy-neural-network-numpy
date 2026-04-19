import numpy as np

# 1. Define the Activation Function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ToyNeuralNetwork:
    def __init__(self):
        # Set a random seed for reproducibility
        np.random.seed(42)
        
        # We are building a simple 2-layer network (Input -> Hidden -> Output)
        # 2 inputs, 3 hidden neurons, 1 output neuron
        self.weights_input_hidden = np.random.uniform(-1, 1, (2, 3))
        self.weights_hidden_output = np.random.uniform(-1, 1, (3, 1))

    def train(self, inputs, expected_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            # --- Forward Pass (Prediction) ---
            hidden_layer_activation = np.dot(inputs, self.weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output, self.weights_hidden_output)
            predicted_output = sigmoid(output_layer_activation)

            # --- Backpropagation (Learning from mistakes) ---
            # Calculate the error (Difference between expected and predicted)
            error = expected_outputs - predicted_output
            
            # Calculate gradients for the output layer
            d_predicted_output = error * sigmoid_derivative(predicted_output)
            
            # Calculate gradients for the hidden layer
            error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

            # Update weights
            self.weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
            self.weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate

    def predict(self, inputs):
        hidden_layer_output = sigmoid(np.dot(inputs, self.weights_input_hidden))
        return sigmoid(np.dot(hidden_layer_output, self.weights_hidden_output))

if __name__ == "__main__":
    # Training data: The classic XOR logic problem
    # Inputs: [0,0], [0,1], [1,0], [1,1]
    X_train = np.array([[0, 0],[0, 1], [1, 0], [1, 1]])
    
    # Expected Outputs for XOR: 0, 1, 1, 0
    y_train = np.array([[0], [1], [1], [0]])

    # Initialize and train the neural network
    nn = ToyNeuralNetwork()
    print("Training the Neural Network for 10,000 epochs...")
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.1)

    # Test the network
    print("\n--- Testing the trained network ---")
    for test_input in X_train:
        prediction = nn.predict(test_input)
        print(f"Input: {test_input} -> Prediction: {prediction[0]:.4f} (Rounded: {round(prediction[0])})")

# Backpropogation
import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, weights, X, y, lr=1):    # lr is the learning rate
        # Weights between input layer and hidden layer
        self.X = X
        self.y = y
        self.lr = lr
        self.w13 = weights['w13']
        self.w14 = weights['w14']
        self.w23 = weights['w23']
        self.w24 = weights['w24']
        # Weights between hidden layer and output layer
        self.w35 = weights['w35']
        self.w45 = weights['w45']

    def forward(self):            # Forward pass
        a3 = self.X[0]*self.w13 + self.X[1]*self.w23 # Weighted sum of inputs
        self.y3 = sigmoid(a3)                        # Activation function
        a4 = self.X[0]*self.w14 + self.X[1]*self.w24 
        self.y4 = sigmoid(a4)
        a5 = self.y3*self.w35 + self.y4*self.w45
        self.y5 = sigmoid(a5)
        return self.y5            # Return the output

    def backward(self, output):   # Backward pass
        error = self.y - output   # Calculate the error
        d5 = error * sigmoid_derivative(output) # Calculate the derivative of the error

        # Calculate error for hidden layer
        d3 = d5*self.w35*sigmoid_derivative(self.y3)
        d4 = d5*self.w45*sigmoid_derivative(self.y4)

        # Update weights
        self.w35 += self.y3*d5*self.lr      # Update the weights between hidden layer and output layer
        self.w45 += self.y4*d5*self.lr      # self.lr is the learning rate
        # hidden layer
        self.w13 += self.X[0]*d3*self.lr    # Update the weights between input layer and hidden layer
        self.w14 += self.X[0]*d4*self.lr    # self.X[0] is the first input
        self.w23 += self.X[1]*d3*self.lr    # self.X[1] is the second input
        self.w24 += self.X[1]*d4*self.lr

    def train(self, epochs=500):  # Train the neural network
        for epoch in range(epochs):
            # Forward pass
            output = self.forward()

            # Backward pass
            self.backward(output)

            if epoch % 50 == 0:
                # Print loss and weights
                loss = np.mean((self.y - output) ** 2)  # Mean Squared Error
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
                print(f"Weights: w13: {self.w13}, w14: {self.w14}, w23: {self.w23}, w24: {
                    self.w24}, w35: {self.w35}, w45: {self.w45}, \n{output}")

# Example usage
if __name__ == "__main__":
    X = [0.35, 0.7] # Input
    y = 0.5         # Target output

    # Define initial weights
    initial_weights = {
        'w13': 0.2,     # w13 is the weight between the first input and the first hidden node
        'w14': 0.3,     # w14 is the weight between the first input and the second hidden node
        'w23': 0.2,     # w23 is the weight between the second input and the first hidden node
        'w24': 0.3,     # w24 is the weight between the second input and the second hidden node
        'w35': 0.3,     # w35 is the weight between the first hidden node and the output node 
        'w45': 0.9      # w45 is the weight between the second hidden node and the output node
    }

    nn = SimpleNeuralNetwork(initial_weights, X, y, 1)
    nn.train()

    # Test the network, print predictions
    print("Predictions:")
    print(nn.forward()) 
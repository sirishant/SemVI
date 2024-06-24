#Single Layer Perceptron for OR gate
import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.2, epochs=3):  # Default values for learning_rate and epochs
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.array([-0.2, 0.4, 0])  # -0.2 is the bias weight, 0.4 is the weight for the first feature, 0 is the weight for the second feature

    def activation_function(self, x):   # Step function
        return 1 if x >= 0 else 0

    def predict(self, inputs):  # Calculate the weighted sum of inputs and weights
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]   # self.weights[1:] is used to exclude the bias weight, self.weights[0] is the bias weight
        return self.activation_function(summation)  # Apply the activation function

    def fit(self, X, y):    # Train the perceptron
        mse_history = []    # Store the Mean Squared Error for each epoch
        print("Initial Weights:", self.weights) # Print the initial weights
        for epoch in range(self.epochs):        # Iterate through the epochs
            total_error = 0                     # Initialize the total error for the epoch
            predictions = []                    # Store the predictions for each input
            for inputs, label in zip(X, y):     # Iterate through the training data
                prediction = self.predict(inputs)                   # Get the prediction
                predictions.append((inputs, prediction))            # Store the prediction
                error = label - prediction                          # Calculate the error
                total_error += error**2                             # Add the squared error to the total error
                self.weights[1:] += self.learning_rate * error * inputs # Update the weights
                self.weights[0] += self.learning_rate * error       # Update the bias weight
            mse = total_error / len(y)          # Calculate the Mean Squared Error for the epoch
            mse_history = mse                   # Store the MSE in the history
            print(f"Epoch {epoch + 1}:")        
            print(f"  Weights: {self.weights}")
            print("  Predictions:")
            for input, prediction in predictions:
                print(f"    Input: {input}, Predicted Output: {prediction}")
            print(f"  Mean Squared Error: {mse}\n")
        return mse_history

# Define the training data for the OR function
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1])  # OR logic gate outputs

# Create a Perceptron instance
perceptron = Perceptron(num_features=2) # 2 features for the OR gate (x1 and x2)

# Train the perceptron and get MSE history
mse_history = perceptron.fit(X, y) 

# Test the perceptron with the training data
print("\nPredictions on the training data:")
for inputs in X:
    print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")

# Print the final state of the neural network
print("\nFinal Neural Network Weights:")
print(perceptron.weights)
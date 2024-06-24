# Naive Bayes Classifier
'''
in the datset we have
Outlook: Rainy = 1, Overcast = 0, Sunny = 2
Temperature: Hot = 1, Mild = 2, Cool = 0
Humidity: High = 1, Normal = 0
Windy: False = 0, True = 1
Play Golf: No = 0, Yes = 1
'''
import csv
import numpy as np

# Load the dataset
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(value) for value in row])
    data = np.array(data)               # Convert to NumPy array
    return data[:, :-1], data[:, -1]    # Return X, y

# Calculate prior probabilities
def calculate_priors(y):
    classes, counts = np.unique(y, return_counts=True)  # Count occurrences of each class
    priors = dict(zip(classes, counts / len(y)))        # Calculate priors
    return priors

# Calculate conditional probabilities
def calculate_conditional_probabilities(X, y):
    conditional_probs = {}
    for cls in np.unique(y):                # Iterate over each class
        X_cls = X[y == cls]                 # Filter samples of class cls
        feature_probs = []                  # List to store feature probabilities
        for col in range(X.shape[1]):       # Iterate over each feature
            values, counts = np.unique(X_cls[:, col], return_counts=True)   # Count occurrences of feature values
            probs = dict(zip(values, counts / len(X_cls)))                  # Calculate probabilities
            feature_probs.append(probs)                                     # Append probabilities to the list
        conditional_probs[cls] = feature_probs  # Store feature probabilities for the class
    return conditional_probs

# Classify a new sample
def classify(sample, priors, conditional_probs):
    probabilities = {}
    for cls in priors:
        prior = priors[cls]             # Prior probability of the class
        likelihood = 1.0                # Initialize likelihood to 1
        for col, value in enumerate(sample):    # Iterate over each feature value
            feature_probs = conditional_probs[cls][col] # Get feature probabilities for the class
            # Smoothing for unseen feature values       
            likelihood *= feature_probs.get(value, 1e-6)# Multiply by the probability of the feature value
        probabilities[cls] = prior * likelihood         # Multiply by the prior
    return max(probabilities, key=probabilities.get), probabilities # Return the class with the highest probability

# Evaluate the classifier on a dataset
def evaluate(X, y, priors, conditional_probs):
    predictions = [classify(sample, priors, conditional_probs)[0] for sample in X]  # Classify each sample
    accuracy = np.mean(predictions == y) # Calculate accuracy
    return accuracy

# Predict function for interactive use
def predict(input_features, priors, conditional_probs):
    input_array = np.array(input_features, dtype=float) # Convert input to NumPy array
    prediction, probabilities = classify(input_array, priors, conditional_probs) # Classify the input
    for cls in probabilities:                           # Print probabilities for each class
        print(f'P({cls}) = {probabilities[cls]:.6f}')
    return prediction

# Main function
def main(file_path):
    X, y = load_data(file_path)

    # Discretize data (optional, if features are continuous)
    X_discrete = np.floor(X)  # Discretize features

    priors = calculate_priors(y)                # Calculate prior probabilities
    conditional_probs = calculate_conditional_probabilities(X_discrete, y)  # Calculate conditional probabilities

    # Evaluate the model on the dataset
    accuracy = evaluate(X_discrete, y, priors, conditional_probs)   
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Prediction loop
    while True:
        try:
            input_str = input("\n Enter new sample (comma-separated values) or 'exit' to quit: ")
            if input_str.lower() == 'exit':
                break
            input_features = [float(x) for x in input_str.split(',')]       # Parse input
            if len(input_features) != X.shape[1]:                           # Check number of features
                print(f"Please enter exactly {X.shape[1]} feature values.")
                continue
            input_features_discrete = np.floor(input_features)              # Discretize input
            prediction = predict(input_features_discrete, priors, conditional_probs) # Predict class
            print(f'Predicted class: {prediction}')
        except ValueError:
            print("Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    main('naive_bayes_dataset.csv')
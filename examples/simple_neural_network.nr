# Create a simple neural network for binary classification
model = NeuralNetwork(input_size=3, output_size=1) {
    Dense(units=64, activation="relu")
    Dense(units=32, activation="relu")
    Dense(units=1, activation="sigmoid")
}

# Configure loss function and optimizer
loss(type="bce")
optimizer(type="adam", learning_rate=0.001)

# Load training data
data = load_matrix("examples/training_data.nrm")

# Train the model
model.train(data, epochs=100)

# Load and evaluate test data
test_data = load_matrix("examples/test_data.nrm")
accuracy = model.evaluate(test_data)

print("Test accuracy: ", accuracy) 
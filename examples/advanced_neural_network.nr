# Create a neural network with custom architecture
model = NeuralNetwork(input_size=3, output_size=1) {
    Dense(units=64, activation="relu")
    Dropout(rate=0.2)
    Dense(units=32, activation="relu")
    normalize(features=32)
    Dense(units=16, activation="relu")
}

# Load training data
data = load_matrix("examples/training_data.nrm")

# Train the model
model.train(data, epochs=100, learning_rate=0.01)

# Save the trained model
save_model(model, "trained_model.pt")

# Load the model back
loaded = load_model("trained_model.pt")

# Evaluate both models
test = load_matrix("examples/test_data.nrm")
original = model.evaluate(test)
loaded_acc = loaded.evaluate(test)

# Print results
print("Original model accuracy: ", original)
print("Loaded model accuracy: ", loaded_acc) 
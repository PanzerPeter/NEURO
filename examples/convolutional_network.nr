# Create a CNN for image classification
model = NeuralNetwork(input_size=784, output_size=10) {
    # Reshape input to 1x28x28
    Conv2D(in_channels=1, out_channels=32, kernel_size=3)
    MaxPool(kernel_size=2)
    Conv2D(in_channels=32, out_channels=64, kernel_size=3)
    MaxPool(kernel_size=2)
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Dense(units=64, activation="relu")
}

# Load MNIST-like dataset
data = load_matrix("examples/mnist_train.nrm")

# Train the model
model.train(data, epochs=10, learning_rate=0.001)

# Evaluate on test data
test = load_matrix("examples/mnist_test.nrm")
accuracy = model.evaluate(test)

# Print results
print("Model accuracy on MNIST: ", accuracy)

# Save the trained model
save_model(model, "mnist_model.pt") 
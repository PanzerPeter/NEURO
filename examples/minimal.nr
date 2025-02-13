# Minimal example to test NEURO functionality
model = NeuralNetwork(input_size=2, output_size=1) {
    Dense(units=64, activation="relu");
    Dense(units=32, activation="relu");
    Dense(units=1, activation="sigmoid");
}

print("Created a simple neural network"); 
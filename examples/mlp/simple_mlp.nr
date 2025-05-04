# examples/mlp/simple_mlp.nr
# Defines a simple Multi-Layer Perceptron (MLP).

print("Defining a simple MLP model...");

# Define the model and assign it to a variable
# input_size/output_size are informational
mlp_model = NeuralNetwork(input_size=4, output_size=1) {
    # Layer definitions require semicolons
    Dense(units=16, activation="relu");
    Dense(units=8, activation="relu");
    Dense(units=1, activation="sigmoid"); # For binary classification output
};

print("MLP Model defined:");
print(mlp_model); # Print the model structure (representation might be basic) 
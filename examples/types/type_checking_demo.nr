# Example demonstrating Neuro Type System Features

# --- Correctly Typed Model ---

# Define an MLP for MNIST-like data (28x28 = 784 features)
# Note: Requires parser to handle numeric params like input_features
# Assuming input shape is inferred or handled by interpreter later if not specified
# For type checker, input shape helps validate first layer.
mnist_mlp = NeuralNetwork() { # Input implicitly Any -> Tensor(shape=(None, 784)) assumed by first layer maybe?
    Dense(units=128, activation="relu"); # Output: (None, 128)
    Dense(units=64, activation="relu");  # Output: (None, 64)
    Dense(units=10, activation="softmax");# Output: (None, 10)
};

# --- Simple CNN Example ---
# Requires parser to handle tuple values for kernel_size/stride/padding
cifar_cnn = NeuralNetwork(input_shape=(None, 3, 32, 32)) { # Explicit input shape
    Conv2D(out_channels=32, kernel_size=3, padding=1); # Output: (None, 32, 32, 32)
    MaxPool2D(kernel_size=2, stride=2);               # Output: (None, 32, 16, 16)
    Flatten();                                        # Output: (None, 32*16*16 = 8192)
    Dense(units=128, activation="relu");              # Output: (None, 128)
    Dense(units=10, activation="softmax");           # Output: (None, 10)
};


# --- Variable Assignments ---
learning_rate = 0.001;
epoch_count = 20;
model_path = "./saved_models/mlp_v1.nrm";

# --- Loss and Optimizer ---
Loss(type="BCE"); # Expects model output compatible with TensorFloat32(shape=(None,)) or (None,1)
                 # mnist_mlp output is (None, 10) - Type Error expected here if training!
                 # cifar_cnn output is (None, 10) - Type Error expected here if training!
                 # Need CrossEntropyLoss for multi-class

Optimizer(type="Adam", lr=0.001); # Use literal instead of variable for type checking


# --- Function/Method Calls ---
# data = load_data(path="./data/cifar.nrm"); # Returns DataType
# print(data);


# --- Training / Evaluation ---
# train(mnist_mlp, data=data, epochs=epoch_count); # Expects Loss compatible with model output
# save(mnist_mlp, filepath=model_path);


# --- Examples of Type Errors (Commented Out) ---

# Error: Incompatible layer dimensions
# incompatible_mlp = NeuralNetwork() {
#     Dense(units=100); # Output: (None, 100)
#     # Next Dense layer input features (implicitly 100) match.
#     Dense(units=50);  # Output: (None, 50)
#     # Conv2D expects 4D input, gets 2D (None, 50)
#     Conv2D(out_channels=16, kernel_size=3); # << TYPE ERROR HERE
# };

# Error: Passing wrong type to function
# count = 5;
# print(count); # OK: print accepts AnyType (including Int)
# Needs String
# data = load_data(path=count); # << TYPE ERROR HERE (Expected String, got Int)

# Error: Using undefined variable
# result = undefined_var * 10; # << TYPE ERROR HERE (undefined_var not defined)

# Error: Loss/Optimizer not defined before train
# train(mnist_mlp, data=data, epochs=10); # << TYPE ERROR HERE (if Loss/Optimizer weren't defined above)

# Error: Loss incompatible with Model Output
# multi_class_model = NeuralNetwork() { Dense(units=10); }; # Output (None, 10)
# Loss(type="BCE"); # BCE expects single output (or compatible)
# train(multi_class_model, data=data, epochs=5); # << TYPE ERROR HERE (Model output (None, 10) incompatible with BCE loss) 
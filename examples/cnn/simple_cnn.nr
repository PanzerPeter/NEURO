# examples/cnn/simple_cnn.nr
# Defines a simple Convolutional Neural Network (CNN).

print("Defining a simple CNN model...");

# Define the model and assign it to a variable
# input_size/output_size are informational
# Assuming input is like a 1x8x8 image -> 64 features when flattened
# Outputting 3 classes
cnn_model = NeuralNetwork(input_size=[1, 8, 8], output_size=3) {
    # Layer definitions require semicolons
    
    # First Conv layer MUST specify in_channels (e.g., 1 for grayscale)
    Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1, activation="relu");
    BatchNorm(); # Infers dim=2, features=8 from previous Conv2D
    
    # Second Conv layer infers in_channels=8 from previous
    Conv2D(out_channels=16, kernel_size=3, padding=1, activation="relu"); 
    BatchNorm(); # Infers dim=2, features=16

    # MaxPool2D() # Not implemented yet - would go here
    
    Flatten(); # Flatten the 2D feature map for Dense layers
    
    Dense(units=32, activation="relu");
    BatchNorm(); # Infers dim=1 after Dense
    # Dropout(rate=0.5); # Example if Dropout was implemented
    
    Dense(units=3, activation="softmax"); # Output layer for 3 classes
};

print("CNN Model defined:");
print(cnn_model); # Print the model structure 